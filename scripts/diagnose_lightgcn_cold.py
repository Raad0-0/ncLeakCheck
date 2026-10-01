"""
Diagnose LightGCN Regime C score behavior.
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

from diagnostics.metrics import compute_auprc, compute_auroc
from engine.lightgcn_training import (
    _evaluate_lightgcn,
    set_deterministic_seed,
)
from engine.lightgcn_utils import adjacency_to_edge_index
from engine.negative_sampler import sample_uniform_negatives
from engine.training_loop import _make_test_loader
from engine.splitter import split_regime_c
from models.lightgcn_bipartite_clean import BipartiteLightGCN
from engine.cold_start_negative_sampler import (
    sample_cold_mirna_negatives,
)


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

    return clusters


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"
    A_full = np.load(data_dir / "hmdd_v4_strict_adjacency.npy")
    clusters = load_clusters(data_dir)

    seed = 42
    set_deterministic_seed(seed)

    split = split_regime_c(
        A_full=A_full,
        precursor_clusters=clusters,
        test_ratio=0.2,
        seed=seed,
    )

    test_neg = sample_cold_mirna_negatives(
        train_pairs=split["train_pairs"],
        test_pairs=split["test_pairs"],
        test_mirnas=split["test_mirnas"],
        n_diseases=A_full.shape[1],
        ratio=1.0,
        seed=seed + 2,
    )

    test_loader = _make_test_loader(
        test_pairs=split["test_pairs"],
        neg_pairs=test_neg,
        batch_size=512,
    )

    model = BipartiteLightGCN(
        n_mirnas=A_full.shape[0],
        n_diseases=A_full.shape[1],
        embed_dim=32,
        n_layers=2,
    )

    device = torch.device("cpu")
    model = model.to(device)

    edge_index = adjacency_to_edge_index(
        split["A_train"],
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.BCELoss()

    from engine.lightgcn_training import (
        _train_lightgcn_epoch,
    )

    for _ in range(30):
        _train_lightgcn_epoch(
            model=model,
            train_loader=(
                __import__(
                    "engine.training_loop",
                    fromlist=["_make_train_loader"],
                )._make_train_loader(
                    train_pairs=split["train_pairs"],
                    neg_pairs=sample_uniform_negatives(
                        train_pairs=split["train_pairs"],
                        test_pairs=split["test_pairs"],
                        n_mirnas=A_full.shape[0],
                        n_diseases=A_full.shape[1],
                        ratio=2.0,
                        seed=seed + 1,
                    ),
                    batch_size=512,
                    seed=seed,
                )
            ),
            optimizer=optimizer,
            criterion=criterion,
            edge_index=edge_index,
            device=device,
        )

    y_true, y_score = _evaluate_lightgcn(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        device=device,
    )

    reversed_score = 1.0 - y_score

    positive_scores = y_score[y_true == 1]
    negative_scores = y_score[y_true == 0]

    print("Regime C score diagnostic")
    print("=" * 50)
    print(f"positive count: {len(positive_scores)}")
    print(f"negative count: {len(negative_scores)}")
    print(f"positive mean: {positive_scores.mean():.6f}")
    print(f"negative mean: {negative_scores.mean():.6f}")
    print(f"positive median: {np.median(positive_scores):.6f}")
    print(f"negative median: {np.median(negative_scores):.6f}")
    print(f"score min: {y_score.min():.6f}")
    print(f"score max: {y_score.max():.6f}")
    print(f"original AUPRC: {compute_auprc(y_true, y_score):.6f}")
    print(f"reversed AUPRC: {compute_auprc(y_true, reversed_score):.6f}")
    print(f"original AUROC: {compute_auroc(y_true, y_score):.6f}")
    print(f"reversed AUROC: {compute_auroc(y_true, reversed_score):.6f}")
    print(
        "test negatives use miRNAs:",
        sorted(set(test_neg[:, 0].tolist())),
    )

    print(
        "test positives use miRNAs:",
        sorted(set(split["test_pairs"][:, 0].tolist())),
    )


if __name__ == "__main__":
    main()