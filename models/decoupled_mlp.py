"""
Similarity-aware decoupled MLP baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _validate_similarity(
    similarity: torch.Tensor,
    expected_size: int,
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    if similarity.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")

    if similarity.shape != (expected_size, expected_size):
        raise ValueError(
            f"{name} must have shape "
            f"({expected_size}, {expected_size}), got {tuple(similarity.shape)}."
        )

    return similarity.to(device=reference.device, dtype=reference.dtype)


def _normalize_similarity(similarity: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(
        similarity.shape[0],
        device=similarity.device,
        dtype=similarity.dtype,
    )
    matrix = similarity + identity
    degree = matrix.sum(dim=1).clamp_min(torch.finfo(matrix.dtype).eps)
    inv_sqrt_degree = degree.rsqrt()
    return inv_sqrt_degree[:, None] * matrix * inv_sqrt_degree[None, :]


class DecoupledMLP(nn.Module):
    """
    MLP over trainable node embeddings.

    When similarity matrices are provided, normalized similarity propagation
    is applied to the complete node embedding tables before batch lookup.
    """

    def __init__(
        self,
        n_mirnas: int,
        n_diseases: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_mirnas = n_mirnas
        self.n_diseases = n_diseases
        self.embed_dim = embed_dim

        self.mirna_embed = nn.Embedding(n_mirnas, embed_dim)
        self.disease_embed = nn.Embedding(n_diseases, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mirna_embed.weight)
        nn.init.xavier_uniform_(self.disease_embed.weight)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _node_embeddings(
        self,
        K_mirna: torch.Tensor | None,
        K_disease: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mirna_table = self.mirna_embed.weight
        disease_table = self.disease_embed.weight

        if (K_mirna is None) != (K_disease is None):
            raise ValueError(
                "K_mirna and K_disease must either both be provided or both be None."
            )

        if K_mirna is None:
            return mirna_table, disease_table

        K_mirna = _validate_similarity(
            K_mirna,
            self.n_mirnas,
            "K_mirna",
            mirna_table,
        )
        K_disease = _validate_similarity(
            K_disease,
            self.n_diseases,
            "K_disease",
            disease_table,
        )

        K_mirna = _normalize_similarity(K_mirna)
        K_disease = _normalize_similarity(K_disease)

        return K_mirna @ mirna_table, K_disease @ disease_table

    def forward(
        self,
        mirna_idx: torch.Tensor,
        disease_idx: torch.Tensor,
        K_mirna: torch.Tensor | None = None,
        K_disease: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mirna_table, disease_table = self._node_embeddings(
            K_mirna,
            K_disease,
        )

        m_emb = mirna_table[mirna_idx]
        d_emb = disease_table[disease_idx]

        x = torch.cat([m_emb, d_emb], dim=-1)
        return self.mlp(x).squeeze(-1)