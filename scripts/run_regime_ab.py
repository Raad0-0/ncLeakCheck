"""
Mini-benchmark: 5-fold CV for Regime A vs B vs C with simple baselines.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from engine.training_loop import run_fold
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor


def run_5fold_cv(
    A_full: np.ndarray,
    model_name: str = "mlp",
    n_folds: int = 5,
    test_ratio: float = 0.2,
    neg_ratio: float = 2.0,
    n_epochs: int = 30,
    base_seed: int = 42,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_mirnas, n_diseases = A_full.shape
    rng = np.random.default_rng(base_seed)
    fold_seeds = rng.integers(0, 10000, size=n_folds).tolist()

    results = []

    for fold_idx, seed in enumerate(fold_seeds):
        for regime in ["A", "B"]:
            if model_name == "mlp":
                model = DecoupledMLP(n_mirnas=n_mirnas, n_diseases=n_diseases, embed_dim=32, hidden_dim=64)
            elif model_name == "gcn":
                model = BipartiteGCNLinkPredictor(n_mirnas=n_mirnas, n_diseases=n_diseases, embed_dim=32, gcn_hidden=64)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            metrics = run_fold(
                model, A_full, test_ratio=test_ratio, neg_ratio=neg_ratio,
                n_epochs=n_epochs, batch_size=256, lr=1e-3, seed=seed,
                regime=regime, device=device,
            )

            results.append({
                "fold": fold_idx + 1,
                "regime": regime,
                "model": model_name,
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "n_train_pos": metrics["n_train_pos"],
                "n_test_pos": metrics["n_test_pos"],
            })

    print(f"{'Fold':<5} {'Regime':<8} {'Model':<6} {'AUPRC':<8} {'AUROC':<8} {'N_train_pos':<12} {'N_test_pos':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['fold']:<5} {r['regime']:<8} {r['model']:<6} {r['auprc']:.4f}   {r['auroc']:.4f}   {r['n_train_pos']:<12} {r['n_test_pos']:<10}")

    print("\nSummary (mean ± std AUPRC):")
    for regime in ["A", "B"]:
        auprc_vals = [r["auprc"] for r in results if r["regime"] == regime]
        mean_auprc = np.mean(auprc_vals)
        std_auprc = np.std(auprc_vals)
        print(f"  Regime {regime}: {mean_auprc:.4f} ± {std_auprc:.4f}")


def run_5fold_cv_regime_c(
    A_full: np.ndarray,
    entity_index: dict,
    precursor_clusters: dict,
    model_name: str = "mlp",
    n_folds: int = 5,
    test_ratio: float = 0.2,
    neg_ratio: float = 2.0,
    n_epochs: int = 30,
    base_seed: int = 42,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_mirnas, n_diseases = A_full.shape

    # Build miRNA name -> index mapping
    mirna_name_to_idx = entity_index["miRNA_index"]

    # Convert precursor_clusters (name-based) to index-based
    precursor_clusters_idx = {}
    for cluster_id, mirna_names in precursor_clusters["clusters"].items():
        indices = [mirna_name_to_idx[name] for name in mirna_names if name in mirna_name_to_idx]
        if indices:
            precursor_clusters_idx[cluster_id] = indices

    if len(precursor_clusters_idx) == 0:
        raise ValueError("No valid miRNA clusters found after mapping to indices")

    rng = np.random.default_rng(base_seed)
    fold_seeds = rng.integers(0, 10000, size=n_folds).tolist()

    results = []

    for fold_idx, seed in enumerate(fold_seeds):
        if model_name == "mlp":
            model = DecoupledMLP(n_mirnas=n_mirnas, n_diseases=n_diseases, embed_dim=32, hidden_dim=64)
        elif model_name == "gcn":
            model = BipartiteGCNLinkPredictor(n_mirnas=n_mirnas, n_diseases=n_diseases, embed_dim=32, gcn_hidden=64)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        from engine.splitter import split_regime_c
        from engine.negative_sampler import sample_uniform_negatives
        from engine.training_loop import _make_train_loader, _make_test_loader, train_epoch, evaluate
        from diagnostics.metrics import compute_auprc, compute_auroc

        split = split_regime_c(A_full, precursor_clusters_idx, test_ratio=test_ratio, seed=seed)

        A_train = split["A_train"]
        train_pairs = split["train_pairs"]
        test_pairs = split["test_pairs"]

        train_neg = sample_uniform_negatives(
            train_pairs, test_pairs, n_mirnas, n_diseases, ratio=neg_ratio, seed=seed + 1
        )
        test_neg = sample_uniform_negatives(
            train_pairs, test_pairs, n_mirnas, n_diseases, ratio=neg_ratio, seed=seed + 2
        )

        train_loader = _make_train_loader(train_pairs, train_neg, batch_size=256, seed=seed)
        test_loader = _make_test_loader(test_pairs, test_neg, batch_size=256)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        for epoch in range(n_epochs):
            _ = train_epoch(model, train_loader, optimizer, criterion, device)

        y_true, y_score = evaluate(model, test_loader, device)

        results.append({
            "fold": fold_idx + 1,
            "regime": "C",
            "model": model_name,
            "auprc": compute_auprc(y_true, y_score),
            "auroc": compute_auroc(y_true, y_score),
            "n_train_pos": len(train_pairs),
            "n_test_pos": len(test_pairs),
            "n_test_mirnas": len(split["test_mirnas"]),
        })

    print(f"{'Fold':<5} {'Regime':<8} {'Model':<6} {'AUPRC':<8} {'AUROC':<8} {'N_train':<10} {'N_test':<8} {'N_test_miR':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['fold']:<5} {r['regime']:<8} {r['model']:<6} {r['auprc']:.4f}   {r['auroc']:.4f}   {r['n_train_pos']:<10} {r['n_test_pos']:<8} {r['n_test_mirnas']:<10}")

    auprc_vals = [r["auprc"] for r in results]
    mean_auprc = np.mean(auprc_vals)
    std_auprc = np.std(auprc_vals)
    print(f"\nRegime C (mean ± std AUPRC): {mean_auprc:.4f} ± {std_auprc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mini-benchmark: Regime A/B/C")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gcn"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="data/processed/hmdd_v4_strict_adjacency.npy")
    parser.add_argument("--entity-index", type=str, default="data/processed/hmdd_v4_strict_entity_index.json")
    parser.add_argument("--clusters", type=str, default="data/processed/hmdd_v4_strict_precursor_family_clusters.json")
    parser.add_argument("--regime", type=str, default="all", choices=["A", "B", "C", "AB", "all"])
    parser.add_argument("--subset", type=float, default=1.0)
    args = parser.parse_args()

    A_full = np.load(args.data)
    print(f"Loaded adjacency matrix: {A_full.shape}, {np.sum(A_full)} positive edges")

    if args.subset < 1.0:
        n_m, n_d = A_full.shape
        sub_m = max(1, int(n_m * args.subset))
        sub_d = max(1, int(n_d * args.subset))
        rng = np.random.default_rng(args.seed)
        mi_idx = rng.choice(n_m, size=sub_m, replace=False)
        di_idx = rng.choice(n_d, size=sub_d, replace=False)
        A_full = A_full[np.ix_(mi_idx, di_idx)]
        print(f"Subsampled to: {A_full.shape}, {np.sum(A_full)} positive edges")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    regimes_to_run = ["A", "B", "C"] if args.regime == "all" else list(args.regime)

    if "A" in regimes_to_run or "B" in regimes_to_run:
        print("\n" + "=" * 60)
        print("Running Regime A/B benchmarks...")
        print("=" * 60)

        if args.regime == "A":
            regimes_ab = ["A"]
        elif args.regime == "B":
            regimes_ab = ["B"]
        elif args.regime == "AB":
            regimes_ab = ["A", "B"]
        else:
            regimes_ab = ["A", "B"]

        from engine.training_loop import run_fold

        for regime in regimes_ab:
            print(f"\n--- Regime {regime} ---")
            rng = np.random.default_rng(args.seed)
            fold_seeds = rng.integers(0, 10000, size=args.folds).tolist()

            for fold_idx, seed in enumerate(fold_seeds):
                if args.model == "mlp":
                    model = DecoupledMLP(n_mirnas=A_full.shape[0], n_diseases=A_full.shape[1], embed_dim=32, hidden_dim=64)
                else:
                    model = BipartiteGCNLinkPredictor(n_mirnas=A_full.shape[0], n_diseases=A_full.shape[1], embed_dim=32, gcn_hidden=64)

                metrics = run_fold(
                    model, A_full, test_ratio=0.2, neg_ratio=2.0, n_epochs=args.epochs,
                    lr=1e-3, seed=seed, regime=regime, device=device
                )
                print(f"Fold {fold_idx+1}: AUPRC={metrics['auprc']:.4f}, AUROC={metrics['auroc']:.4f}")

    if "C" in regimes_to_run:
        print("\n" + "=" * 60)
        print("Running Regime C benchmarks...")
        print("=" * 60)

        with open(args.entity_index, "r", encoding="utf-8") as f:
            entity_index = json.load(f)
        with open(args.clusters, "r", encoding="utf-8") as f:
            precursor_clusters = json.load(f)

        run_5fold_cv_regime_c(
            A_full, entity_index, precursor_clusters,
            model_name=args.model, n_folds=args.folds, n_epochs=args.epochs,
            base_seed=args.seed, device=device
        )


if __name__ == "__main__":
    main()