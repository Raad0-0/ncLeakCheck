"""
One-fold HMDD Regime B LightGCN smoke benchmark.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.lightgcn_training import train_lightgcn_fold
from engine.splitter import split_regime_b
from models.lightgcn_bipartite_clean import BipartiteLightGCN


def main() -> None:
    data_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "hmdd_v4_strict_adjacency.npy"
    )

    A_full = np.load(data_path)

    if A_full.ndim != 2:
        raise ValueError("Adjacency matrix must be two-dimensional.")

    n_mirnas, n_diseases = A_full.shape
    n_edges = int(np.sum(A_full > 0))

    print(f"Loaded adjacency: {A_full.shape}")
    print(f"Positive edges: {n_edges}")

    split = split_regime_b(
        A_full=A_full,
        test_ratio=0.2,
        seed=42,
    )

    model = BipartiteLightGCN(
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        embed_dim=16,
        n_layers=1,
    )

    device = torch.device("cpu")

    metrics = train_lightgcn_fold(
        model=model,
        A_train=split["A_train"],
        train_pairs=split["train_pairs"],
        test_pairs=split["test_pairs"],
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        neg_ratio=2.0,
        n_epochs=5,
        batch_size=512,
        lr=1e-3,
        seed=42,
        device=device,
    )

    print("\nLightGCN Regime B smoke result")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()