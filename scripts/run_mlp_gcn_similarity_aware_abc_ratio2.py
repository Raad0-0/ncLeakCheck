"""Five-fold similarity-aware MLP/GCN benchmark for Regimes A, B, and C.

A/B: uniform test negatives, 1:2.
C: held-out-miRNA test negatives, 1:2.
"""

from __future__ import annotations

import argparse
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
from engine.cold_start_negative_sampler import sample_cold_mirna_negatives
from engine.negative_sampler import sample_uniform_negatives
from engine.splitter import split_regime_a, split_regime_b, split_regime_c
from engine.test_negative_sampler import sample_uniform_test_negatives
from engine.training_loop import _make_test_loader, _make_train_loader, evaluate, train_epoch
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor

VERIFIED_FOLD_SEEDS = [892, 7739, 6545, 4388, 4330]


def load_precursor_clusters(data_dir: pathlib.Path) -> dict:
    with (data_dir / "hmdd_v4_strict_entity_index.json").open("r", encoding="utf-8") as f:
        entity_index = json.load(f)

    with (data_dir / "hmdd_v4_strict_precursor_family_clusters.json").open("r", encoding="utf-8") as f:
        raw_clusters = json.load(f)

    name_to_idx = entity_index["miRNA_index"]
    clusters = {}
    for cluster_id, names in raw_clusters["clusters"].items():
        indices = [name_to_idx[name] for name in names if name in name_to_idx]
        if indices:
            clusters[cluster_id] = indices

    if not clusters:
        raise ValueError("No valid precursor clusters were mapped.")
    return clusters


def create_model(model_name: str, n_mirnas: int, n_diseases: int) -> torch.nn.Module:
    if model_name == "mlp":
        return DecoupledMLP(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=32,
            hidden_dim=64,
        )
    if model_name == "gcn":
        return BipartiteGCNLinkPredictor(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=32,
            gcn_hidden=64,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_split(regime: str, A_full: np.ndarray, clusters: dict, seed: int) -> dict:
    if regime == "A":
        return split_regime_a(A_full=A_full, test_ratio=0.2, seed=seed)
    if regime == "B":
        return split_regime_b(A_full=A_full, test_ratio=0.2, seed=seed)
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
    evaluation_ratio: float,
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
            ratio=evaluation_ratio,
            seed=seed + 2,
        )
    else:
        test_neg = sample_uniform_test_negatives(
            train_pairs=split["train_pairs"],
            test_pairs=split["test_pairs"],
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            ratio=evaluation_ratio,
            seed=seed + 2,
        )

    return train_neg, test_neg


def run_fold(
    model_name: str,
    A_full: np.ndarray,
    clusters: dict,
    regime: str,
    fold: int,
    seed: int,
    evaluation_ratio: float,
    n_epochs: int,
    device: torch.device,
) -> dict:
    from engine.lightgcn_training import set_deterministic_seed

    set_deterministic_seed(seed)
    split = build_split(regime, A_full, clusters, seed)
    K_mirna = split.get("K_mirna")
    K_disease = split.get("K_disease")

    n_mirnas, n_diseases = A_full.shape
    train_neg, test_neg = make_negatives(
        regime,
        split,
        n_mirnas,
        n_diseases,
        seed,
        evaluation_ratio,
    )

    train_loader = _make_train_loader(
        split["train_pairs"], train_neg, batch_size=256, seed=seed
    )
    test_loader = _make_test_loader(
        split["test_pairs"], test_neg, batch_size=256
    )

    model = create_model(model_name, n_mirnas, n_diseases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    losses = []

    for _ in range(n_epochs):
        losses.append(
            train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                K_mirna=K_mirna,
                K_disease=K_disease,
            )
        )

    y_true, y_score = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        K_mirna=K_mirna,
        K_disease=K_disease,
    )

    result = {
        "model": model_name,
        "regime": regime,
        "fold": fold,
        "seed": int(seed),
        "evaluation_negative_ratio": "1:2",
        "train_negative_sampling_ratio": "1:2",
        "train_batch_ratio": "1:1",
        "n_epochs": int(n_epochs),
        "embed_dim": 32,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "similarity_aware": True,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-neg-ratio", type=float, default=2.0)
    parser.add_argument("--data", type=str, default="data/processed/hmdd_v4_strict_adjacency.npy")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    if args.folds != len(VERIFIED_FOLD_SEEDS):
        raise ValueError(
            f"This verified benchmark requires --folds {len(VERIFIED_FOLD_SEEDS)}."
        )

    A_full = np.load(args.data).astype(np.float32)
    if A_full.ndim != 2:
        raise ValueError("Adjacency matrix must be two-dimensional.")

    data_dir = PROJECT_ROOT / "data" / "processed"
    clusters = load_precursor_clusters(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"Adjacency shape: {A_full.shape}")
    print(f"Positive edges: {int(np.sum(A_full > 0))}")
    print(f"Using device: {device}")
    print(f"Fold seeds: {VERIFIED_FOLD_SEEDS}")

    for regime in ("A", "B", "C"):
        print(f"\n--- Regime {regime} ---")
        for fold_idx, fold_seed in enumerate(VERIFIED_FOLD_SEEDS, start=1):
            result = run_fold(
                model_name=args.model,
                A_full=A_full,
                clusters=clusters,
                regime=regime,
                fold=fold_idx,
                seed=fold_seed,
                evaluation_ratio=args.eval_neg_ratio,
                n_epochs=args.epochs,
                device=device,
            )
            results.append(result)
            print(
                f"Fold {fold_idx}: AUPRC={result['auprc']:.4f}, "
                f"AUROC={result['auroc']:.4f}, "
                f"test={result['n_test_pos']}:{result['n_test_neg']}"
            )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "ncLeakCheck-Disease",
        "dataset": "hmdd_v4_human_strict",
        "n_mirnas": int(A_full.shape[0]),
        "n_diseases": int(A_full.shape[1]),
        "n_positive_edges": int(np.sum(A_full > 0)),
        "folds": 5,
        "fold_seeds": VERIFIED_FOLD_SEEDS,
        "regimes": ["A", "B", "C"],
        "model": args.model,
        "similarity_aware": True,
        "evaluation_negative_ratio": "1:2",
        "train_negative_sampling_ratio": "1:2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved raw results to: {output_path}")


if __name__ == "__main__":
    main()