"""
Clean LightGCN implementation for the ncLeakCheck-Disease benchmark.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BipartiteLightGCN(nn.Module):
    """LightGCN for a miRNA-disease bipartite graph."""

    def __init__(
        self,
        n_mirnas: int,
        n_diseases: int,
        embed_dim: int = 32,
        n_layers: int = 2,
    ) -> None:
        super().__init__()

        if n_mirnas <= 0 or n_diseases <= 0:
            raise ValueError("n_mirnas and n_diseases must be positive.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if n_layers < 0:
            raise ValueError("n_layers must be non-negative.")

        self.n_mirnas = n_mirnas
        self.n_diseases = n_diseases
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.mirna_embedding = nn.Embedding(n_mirnas, embed_dim)
        self.disease_embedding = nn.Embedding(n_diseases, embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mirna_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.disease_embedding.weight, mean=0.0, std=0.1)

    def _validate_edge_index(self, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges].")

        edge_index = edge_index.to(
            device=self.mirna_embedding.weight.device,
            dtype=torch.long,
        )

        mirna_idx = edge_index[0]
        disease_idx = edge_index[1]

        if torch.any(mirna_idx < 0) or torch.any(
            mirna_idx >= self.n_mirnas
        ):
            raise IndexError("miRNA index out of range.")

        if torch.any(disease_idx < 0) or torch.any(
            disease_idx >= self.n_diseases
        ):
            raise IndexError("disease index out of range.")

        return edge_index

    def _normalized_edges(
        self,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = self._validate_edge_index(edge_index)

        mirna_idx = edge_index[0]
        disease_idx = edge_index[1] + self.n_mirnas

        src = torch.cat([mirna_idx, disease_idx], dim=0)
        dst = torch.cat([disease_idx, mirna_idx], dim=0)

        n_nodes = self.n_mirnas + self.n_diseases

        degree = torch.bincount(src, minlength=n_nodes).to(
            dtype=self.mirna_embedding.weight.dtype
        )

        inv_sqrt_degree = degree.clamp_min(1.0).rsqrt()
        edge_weight = inv_sqrt_degree[src] * inv_sqrt_degree[dst]

        return src, dst, edge_weight


    def score_from_embeddings(
        self,
        mirna_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        mirna_embeddings: torch.Tensor,
        disease_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score pairs using precomputed propagated embeddings."""
        if mirna_idx.shape != disease_idx.shape:
            raise ValueError(
                "mirna_idx and disease_idx must have equal shapes."
            )

        mirna_idx = mirna_idx.to(
            device=mirna_embeddings.device,
            dtype=torch.long,
        )
        disease_idx = disease_idx.to(
            device=disease_embeddings.device,
            dtype=torch.long,
        )

        logits = (
            mirna_embeddings[mirna_idx]
            * disease_embeddings[disease_idx]
        ).sum(dim=-1)

        return torch.sigmoid(logits)

    def propagate(
        self,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate over the fold-specific training graph.

        edge_index must contain training edges only.
        """
        src, dst, edge_weight = self._normalized_edges(edge_index)

        initial = torch.cat(
            [
                self.mirna_embedding.weight,
                self.disease_embedding.weight,
            ],
            dim=0,
        )

        all_embeddings = [initial]
        current = initial

        for _ in range(self.n_layers):
            next_embedding = torch.zeros_like(current)
            next_embedding.index_add_(
                dim=0,
                index=dst,
                source=current[src] * edge_weight.unsqueeze(-1),
            )
            current = next_embedding
            all_embeddings.append(current)

        final = torch.stack(all_embeddings, dim=0).mean(dim=0)

        return (
            final[: self.n_mirnas],
            final[self.n_mirnas :],
        )

    def score_pairs(
        self,
        mirna_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid link probabilities for candidate pairs."""
        if mirna_idx.shape != disease_idx.shape:
            raise ValueError(
                "mirna_idx and disease_idx must have equal shapes."
            )

        mirna_idx = mirna_idx.to(
            device=self.mirna_embedding.weight.device,
            dtype=torch.long,
        )
        disease_idx = disease_idx.to(
            device=self.mirna_embedding.weight.device,
            dtype=torch.long,
        )

        mirna_embeddings, disease_embeddings = self.propagate(edge_index)

        logits = (
            mirna_embeddings[mirna_idx]
            * disease_embeddings[disease_idx]
        ).sum(dim=-1)

        return torch.sigmoid(logits)

    def forward(
        self,
        mirna_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return probabilities.

        edge_index is required for a real benchmark. If omitted, initial
        embeddings are used only for isolated smoke tests.
        """
        if edge_index is None:
            mirna_idx = mirna_idx.to(
                device=self.mirna_embedding.weight.device,
                dtype=torch.long,
            )
            disease_idx = disease_idx.to(
                device=self.mirna_embedding.weight.device,
                dtype=torch.long,
            )

            logits = (
                self.mirna_embedding(mirna_idx)
                * self.disease_embedding(disease_idx)
            ).sum(dim=-1)

            return torch.sigmoid(logits)

        return self.score_pairs(
            mirna_idx=mirna_idx,
            disease_idx=disease_idx,
            edge_index=edge_index,
        )