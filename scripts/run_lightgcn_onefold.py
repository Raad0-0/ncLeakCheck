"""
One-fold, full-configuration LightGCN benchmark for Regimes A/B/C.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.lightgcn_training import train_lightgcn_fold
from engine.splitter import (
    split_regime_a,
    split_regime_b,
    split_regime_c,
)
from models.lightgcn_bipartite_clean import BipartiteLightGCN


def load_clusters(data_dir: pathlib.Path) -> dict:
    with (
        data_dir / "hmdd_v4_strict_entity_index.json"
    ).open("r", encoding="utf-8") as f:
        entity_index = json.load(f)

    with (
        data_dir / "hmdd_v4_strict_precursor_family_clusters.json"
    ).open("r", encoding="utf-8") as f:
        raw_clusters = json.load(f)

    name_to_idx = entity_index["miRNA_index"]
    clusters = {}

    for cluster_id, names in raw_clusters["clusters"].items():
        indices = [
            name_to_idx[name]
            for name in names
            if name in name_to_idx
        ]

        if indices:
            clusters[cluster_id] = indices

    if not clusters:
        raise ValueError("No valid clusters were mapped.")

    return clusters


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"

    A_full = np.load(
        data_dir / "hmdd_v4_strict_adjacency.npy"
    )

    clusters = load_clusters(data_dir)

    print(f"Adjacency shape: {A_full.shape}")
    print(f"Positive edges: {int(np.sum(A_full > 0))}")

    for regime in ("A", "B", "C"):
        seed = 42

        if regime == "A":
            split = split_regime_a(
                A_full,
                test_ratio=0.2,
                seed=seed,
            )
        elif regime == "B":
            split = split_regime_b(
                A_full,
                test_ratio=0.2,
                seed=seed,
            )
        else:
            split = split_regime_c(
                A_full,
                precursor_clusters=clusters,
                test_ratio=0.2,
                seed=seed,
            )

        model = BipartiteLightGCN(
            n_mirnas=A_full.shape[0],
            n_diseases=A_full.shape[1],
            embed_dim=32,
            n_layers=2,
        )

        metrics = train_lightgcn_fold(
            model=model,
            A_train=split["A_train"],
            train_pairs=split["train_pairs"],
            test_pairs=split["test_pairs"],
            n_mirnas=A_full.shape[0],
            n_diseases=A_full.shape[1],
            neg_ratio=2.0,
            n_epochs=30,
            batch_size=512,
            lr=1e-3,
            seed=seed,
            device=torch.device("cpu"),
        )

        print(f"\n--- Regime {regime} ---")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        if regime == "C":
            print(f"n_test_mirnas: {len(split['test_mirnas'])}")
            print(
                f"n_test_clusters: {len(split['test_cluster_ids'])}"
            )


if __name__ == "__main__":
    main()