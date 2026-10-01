"""
LightGCN-style bipartite miRNA-disease link predictor.

The model uses trainable node embeddings and normalized bipartite
message propagation. It returns probabilities so it is compatible
with the existing BCELoss-based training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LightGCNBipartite(nn.Module):
    """
    Lightweight graph collaborative-filtering baseline.

    Parameters
    ----------
    n_mirnas:
        Number of miRNA nodes.
    n_diseases:
        Number of disease nodes.
    embed_dim:
        Node embedding dimension.
    n_layers:
        Number of propagation layers.
    """

    def __init__(
        self,
        n_mirnas: int,
        n_diseases: int,
        embed_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        if n_mirnas <= 0 or n_diseases <= 0:
            raise ValueError("Node counts must be positive.")
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
        nn.init.normal_(self.mirna_embedding.weight, std=0.1)
        nn.init.normal_(self.disease_embedding.weight, std=0.1)

    def _propagate(
        self,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate embeddings over the training bipartite graph.

        edge_index has shape [2, n_edges]:
        row 0 contains miRNA indices and row 1 contains disease indices.
        """

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges].")

        edge_index = edge_index.to(device=device, dtype=torch.long)

        mirna_idx = edge_index[0]
        disease_idx = edge_index[1]

        if bool(torch.any(mirna_idx < 0) or torch.any(mirna_idx >= self.n_mirnas)):
            raise IndexError("miRNA index is out of range.")

        if bool(torch.any(disease_idx < 0) or torch.any(disease_idx >= self.n_diseases)):
            raise IndexError("Disease index is out of range.")

        n_total = self.n_mirnas + self.n_diseases

        left_nodes = mirna_idx
        right_nodes = disease_idx + self.n_mirnas

        src = torch.cat([left_nodes, right_nodes], dim=0)
        dst = torch.cat([right_nodes, left_nodes], dim=0)

        degree = torch.bincount(src, minlength=n_total).to(torch.float32)
        inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)

        edge_weight = inv_sqrt_degree[src] * inv_sqrt_degree[dst]

        initial = torch.cat(
            [
                self.mirna_embedding.weight,
                self.disease_embedding.weight,
            ],
            dim=0,
        )

        embeddings = [initial]
        current = initial

        for _ in range(self.n_layers):
            propagated = torch.zeros_like(current)
            propagated.index_add_(
                0,
                dst,
                current[src] * edge_weight.unsqueeze(1),
            )
            current = propagated
            embeddings.append(current)

        final = torch.stack(embeddings, dim=0).mean(dim=0)

        return (
            final[: self.n_mirnas],
            final[self.n_mirnas :],
        )

    def forward(
        self,
        mirna_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return pairwise association probabilities.

        If edge_index is provided, embeddings are propagated over the
        training graph. If it is omitted, the model uses initial
        trainable embeddings, which is useful for smoke tests only.
        """

        if mirna_idx.shape != disease_idx.shape:
            raise ValueError("mirna_idx and disease_idx must have equal shapes.")

        device = self.mirna_embedding.weight.device
        mirna_idx = mirna_idx.to(device=device, dtype=torch.long)
        disease_idx = disease_idx.to(device=device, dtype=torch.long)

        if edge_index is None:
            mirna_embeddings = self.mirna_embedding.weight
            disease_embeddings = self.disease_embedding.weight
        else:
            mirna_embeddings, disease_embeddings = self._propagate(
                edge_index=edge_index,
                device=device,
            )

        pair_logits = (
            mirna_embeddings[mirna_idx] * disease_embeddings[disease_idx]
        ).sum(dim=-1)

        return torch.sigmoid(pair_logits)