"""
Five-fold LightGCN benchmark for Regimes A, B, and C.

This is the corrected 1:1 evaluation run:
- A/B use uniform unobserved negatives.
- C uses negatives restricted to held-out miRNAs.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.metrics import (
    compute_auprc,
    compute_auroc,
    compute_ece,
    compute_macro_f1,
    compute_mcc,
)
from engine.cold_start_negative_sampler import (
    sample_cold_mirna_negatives,
)
from engine.lightgcn_training import (
    _evaluate_lightgcn,
    _train_lightgcn_epoch,
    set_deterministic_seed,
)
from engine.lightgcn_utils import adjacency_to_edge_index
from engine.negative_sampler import sample_uniform_negatives
from engine.splitter import (
    split_regime_a,
    split_regime_b,
    split_regime_c,
)
from engine.training_loop import (
    _make_test_loader,
    _make_train_loader,
)
from models.lightgcn_bipartite_clean import BipartiteLightGCN


def load_precursor_clusters(data_dir: pathlib.Path) -> dict:
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
        raise ValueError("No valid precursor clusters were mapped.")

    return clusters


def build_split(
    regime: str,
    A_full: np.ndarray,
    clusters: dict,
    seed: int,
) -> dict:
    if regime == "A":
        return split_regime_a(
            A_full=A_full,
            test_ratio=0.2,
            seed=seed,
        )

    if regime == "B":
        return split_regime_b(
            A_full=A_full,
            test_ratio=0.2,
            seed=seed,
        )

    if regime == "C":
        return split_regime_c(
            A_full=A_full,
            precursor_clusters=clusters,
            test_ratio=0.2,
            seed=seed,
        )

    raise ValueError(f"Unknown regime: {regime}")


def make_negatives(
    regime: str,
    split: dict,
    n_mirnas: int,
    n_diseases: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_neg = sample_uniform_negatives(
        train_pairs=split["train_pairs"],
        test_pairs=split["test_pairs"],
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        ratio=2.0,
        seed=seed + 1,
    )

    if regime == "C":
        test_neg = sample_cold_mirna_negatives(
            train_pairs=split["train_pairs"],
            test_pairs=split["test_pairs"],
            test_mirnas=split["test_mirnas"],
            n_diseases=n_diseases,
            ratio=1.0,
            seed=seed + 2,
        )
    else:
        test_neg = sample_uniform_negatives(
            train_pairs=split["train_pairs"],
            test_pairs=split["test_pairs"],
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            ratio=1.0,
            seed=seed + 2,
        )

    return train_neg, test_neg


def run_fold(
    A_full: np.ndarray,
    clusters: dict,
    regime: str,
    fold: int,
    seed: int,
) -> dict:
    device = torch.device("cpu")

    set_deterministic_seed(seed)

    split = build_split(
        regime=regime,
        A_full=A_full,
        clusters=clusters,
        seed=seed,
    )

    n_mirnas, n_diseases = A_full.shape

    train_neg, test_neg = make_negatives(
        regime=regime,
        split=split,
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        seed=seed,
    )

    train_loader = _make_train_loader(
        train_pairs=split["train_pairs"],
        neg_pairs=train_neg,
        batch_size=512,
        seed=seed,
    )

    test_loader = _make_test_loader(
        test_pairs=split["test_pairs"],
        neg_pairs=test_neg,
        batch_size=512,
    )

    model = BipartiteLightGCN(
        n_mirnas=n_mirnas,
        n_diseases=n_diseases,
        embed_dim=32,
        n_layers=2,
    ).to(device)

    edge_index = adjacency_to_edge_index(
        split["A_train"],
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.BCELoss()

    losses = []

    for _ in range(30):
        loss = _train_lightgcn_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            edge_index=edge_index,
            device=device,
        )
        losses.append(loss)

    y_true, y_score = _evaluate_lightgcn(
        model=model,
        test_loader=test_loader,
        edge_index=edge_index,
        device=device,
    )

    result = {
        "model": "lightgcn_clean",
        "regime": regime,
        "fold": fold,
        "seed": seed,
        "evaluation_negative_ratio": "1:1",
        "train_negative_ratio": "1:2",
        "n_epochs": 30,
        "embed_dim": 32,
        "n_layers": 2,
        "learning_rate": 1e-3,
        "batch_size": 512,
        "n_train_pos": int(len(split["train_pairs"])),
        "n_test_pos": int(len(split["test_pairs"])),
        "n_train_neg": int(len(train_neg)),
        "n_test_neg": int(len(test_neg)),
        "final_loss": float(losses[-1]),
        "auprc": compute_auprc(y_true, y_score),
        "auroc": compute_auroc(y_true, y_score),
        "mcc": compute_mcc(y_true, y_score),
        "macro_f1": compute_macro_f1(y_true, y_score),
        "ece": compute_ece(y_true, y_score),
    }

    if regime == "C":
        result["n_test_mirnas"] = int(len(split["test_mirnas"]))
        result["n_test_clusters"] = int(len(split["test_cluster_ids"]))

    return result


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "processed"

    A_full = np.load(
        data_dir / "hmdd_v4_strict_adjacency.npy"
    )

    clusters = load_precursor_clusters(data_dir)

    rng = np.random.default_rng(42)
    fold_seeds = rng.integers(
        0,
        10000,
        size=5,
    ).tolist()

    results = []

    print(f"Adjacency shape: {A_full.shape}")
    print(f"Positive edges: {int(np.sum(A_full > 0))}")
    print(f"Fold seeds: {fold_seeds}")

    for regime in ("A", "B", "C"):
        print(f"\n--- Regime {regime} ---")

        for fold_idx, seed in enumerate(fold_seeds, start=1):
            result = run_fold(
                A_full=A_full,
                clusters=clusters,
                regime=regime,
                fold=fold_idx,
                seed=int(seed),
            )

            results.append(result)

            print(
                f"Fold {fold_idx}: "
                f"AUPRC={result['auprc']:.4f}, "
                f"AUROC={result['auroc']:.4f}"
            )

    output_dir = PROJECT_ROOT / "results" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "lightgcn_abc_ratio1_raw.json"
    )

    payload = {
        "project": "ncLeakCheck-Disease",
        "dataset": "hmdd_v4_human_strict",
        "n_mirnas": int(A_full.shape[0]),
        "n_diseases": int(A_full.shape[1]),
        "n_positive_edges": int(np.sum(A_full > 0)),
        "folds": 5,
        "regimes": ["A", "B", "C"],
        "evaluation_negative_ratio": "1:1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved raw results to: {output_path}")


if __name__ == "__main__":
    main()