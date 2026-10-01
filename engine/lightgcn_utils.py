"""
Utilities for fold-specific LightGCN graph construction.
"""

from __future__ import annotations

import numpy as np
import torch


def adjacency_to_edge_index(
    A_train: np.ndarray,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert a binary training adjacency matrix to edge_index.

    Parameters
    ----------
    A_train:
        Binary array with shape (n_mirnas, n_diseases).
    device:
        Optional target device.

    Returns
    -------
    torch.Tensor:
        Tensor with shape [2, n_edges].
        Row 0 contains miRNA indices.
        Row 1 contains disease indices.
    """
    A_train = np.asarray(A_train)

    if A_train.ndim != 2:
        raise ValueError("A_train must be a 2D array.")

    mirna_idx, disease_idx = np.where(A_train > 0)

    edge_index = torch.from_numpy(
        np.stack([mirna_idx, disease_idx], axis=0)
    ).long()

    if device is not None:
        edge_index = edge_index.to(device)

    return edge_index