"""
One-fold LightGCN smoke test for Regimes A, B, and C.
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
from engine.splitter import split_regime_a, split_regime_b, split_regime_c
from models.lightgcn_bipartite_clean import BipartiteLightGCN


def load_precursor_clusters(
    entity_index_path: pathlib.Path,
    cluster_path: pathlib.Path,
) -> dict:
    with entity_index_path.open("r", encoding="utf-8") as f:
        entity_index = json.load(f)

    with cluster_path.open("r", encoding="utf-8") as f:
        precursor_clusters = json.load(f)

    mirna_name_to_idx = entity_index["miRNA_index"]

    clusters_by_index = {}

    for cluster_id, mirna_names in precursor_clusters["clusters"].items():
        indices = [
            mirna_name_to_idx[name]
            for name in mirna_names
            if name in mirna_name_to_idx
        ]

        if indices:
            clusters_by_index[cluster_id] = indices

    if not clusters_by_index:
        raise ValueError("No valid precursor clusters were mapped.")

    return clusters_by_index


def run_one_regime(
    regime: str,
    A_full: np.ndarray,
    clusters_by_index: dict | None,
) -> dict:
    seed = 42

    if regime == "A":
        split = split_regime_a(
            A_full=A_full,
            test_ratio=0.2,
            seed=seed,
        )
    elif regime == "B":
        split = split_regime_b(
            A_full=A_full,
            test_ratio=0.2,
            seed=seed,
        )
    elif regime == "C":
        if clusters_by_index is None:
            raise ValueError("Regime C requires precursor clusters.")

        split = split_regime_c(
            A_full=A_full,
            precursor_clusters=clusters_by_index,
            test_ratio=0.2,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    model = BipartiteLightGCN(
        n_mirnas=A_full.shape[0],
        n_diseases=A_full.shape[1],
        embed_dim=16,
        n_layers=1,
    )

    metrics = train_lightgcn_fold(
        model=model,
        A_train=split["A_train"],
        train_pairs=split["train_pairs"],
        test_pairs=split["test_pairs"],
        n_mirnas=A_full.shape[0],
        n_diseases=A_full.shape[1],
        neg_ratio=2.0,
        n_epochs=5,
        batch_size=512,
        lr=1e-3,
        seed=seed,
        device=torch.device("cpu"),
    )

    metrics["regime"] = regime

    if regime == "C":
        metrics["n_test_mirnas"] = int(len(split["test_mirnas"]))
        metrics["n_test_clusters"] = int(len(split["test_cluster_ids"]))

    return metrics


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"

    A_full = np.load(
        data_dir / "hmdd_v4_strict_adjacency.npy"
    )

    clusters_by_index = load_precursor_clusters(
        entity_index_path=(
            data_dir / "hmdd_v4_strict_entity_index.json"
        ),
        cluster_path=(
            data_dir
            / "hmdd_v4_strict_precursor_family_clusters.json"
        ),
    )

    print(f"Adjacency shape: {A_full.shape}")
    print(f"Positive edges: {int(np.sum(A_full > 0))}")
    print("\nLightGCN regime smoke results")
    print("=" * 60)

    for regime in ("A", "B", "C"):
        print(f"\n--- Regime {regime} ---")

        metrics = run_one_regime(
            regime=regime,
            A_full=A_full,
            clusters_by_index=clusters_by_index,
        )

        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()