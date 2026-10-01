"""Degree-stratified audit for similarity-aware MLP/GCN Regime B.

Retrains each model for every verified fold, saves predictions, and computes
AUPRC/AUROC by miRNA training-degree group.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.metrics import compute_auprc, compute_auroc
from engine.negative_sampler import sample_uniform_negatives
from engine.splitter import split_regime_b
from engine.test_negative_sampler import sample_uniform_test_negatives
from engine.training_loop import _make_test_loader, _make_train_loader, evaluate, train_epoch
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor

FOLD_SEEDS = [892, 7739, 6545, 4388, 4330]
GROUPS = {
    "low": lambda degree: degree <= 3,
    "medium": lambda degree: 4 <= degree <= 19,
    "high": lambda degree: degree >= 20,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(name: str, n_mirnas: int, n_diseases: int) -> torch.nn.Module:
    if name == "mlp":
        return DecoupledMLP(n_mirnas, n_diseases, embed_dim=32, hidden_dim=64)
    if name == "gcn":
        return BipartiteGCNLinkPredictor(n_mirnas, n_diseases, embed_dim=32, gcn_hidden=64)
    raise ValueError(name)


def evaluate_degree_groups(y_true, y_score, mirna_indices, degrees):
    output = {}
    mirna_indices = np.asarray(mirna_indices)
    degrees = np.asarray(degrees)

    for group_name, predicate in GROUPS.items():
        mask = np.asarray([predicate(int(degrees[mi])) for mi in mirna_indices])
        if not np.any(mask):
            output[group_name] = {
                "n_samples": 0,
                "n_positive": 0,
                "n_negative": 0,
                "auprc": None,
                "auroc": None,
            }
            continue

        group_true = y_true[mask]
        group_score = y_score[mask]
        output[group_name] = {
            "n_samples": int(mask.sum()),
            "n_positive": int(group_true.sum()),
            "n_negative": int((1 - group_true).sum()),
            "auprc": float(compute_auprc(group_true, group_score))
            if len(np.unique(group_true)) == 2
            else None,
            "auroc": float(compute_auroc(group_true, group_score))
            if len(np.unique(group_true)) == 2
            else None,
        }

    low = output["low"]["auprc"]
    high = output["high"]["auprc"]
    output["hub_to_low_auprc_ratio"] = (
        float(high / low) if low is not None and low > 0 and high is not None else None
    )
    return output


def run_fold(model_name, A_full, seed, epochs, device):
    set_seed(seed)
    split = split_regime_b(A_full, test_ratio=0.2, seed=seed)
    train_pairs = split["train_pairs"]
    test_pairs = split["test_pairs"]
    n_mirnas, n_diseases = A_full.shape

    train_neg = sample_uniform_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=2.0, seed=seed + 1
    )
    test_neg = sample_uniform_test_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=2.0, seed=seed + 2
    )

    train_loader = _make_train_loader(train_pairs, train_neg, batch_size=256, seed=seed)
    test_loader = _make_test_loader(test_pairs, test_neg, batch_size=256)
    model = create_model(model_name, n_mirnas, n_diseases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    for _ in range(epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            K_mirna=split["K_mirna"],
            K_disease=split["K_disease"],
        )

    y_true, y_score = evaluate(
        model,
        test_loader,
        device,
        K_mirna=split["K_mirna"],
        K_disease=split["K_disease"],
    )

    mirna_indices = np.concatenate([test_pairs[:, 0], test_neg[:, 0]])
    degrees = split["A_train"].sum(axis=1).astype(np.int64)

    return {
        "seed": int(seed),
        "degree_groups": evaluate_degree_groups(y_true, y_score, mirna_indices, degrees),
        "predictions": {
            "y_true": y_true.astype(float).tolist(),
            "y_score": y_score.astype(float).tolist(),
            "mirna_indices": mirna_indices.astype(int).tolist(),
            "disease_indices": np.concatenate([test_pairs[:, 1], test_neg[:, 1]]).astype(int).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data", default="data/processed/hmdd_v4_strict_adjacency.npy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    A_full = np.load(args.data).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold, seed in enumerate(FOLD_SEEDS, start=1):
        result = run_fold(args.model, A_full, seed, args.epochs, device)
        result["fold"] = fold
        results.append(result)
        print(f"Fold {fold}: {result['degree_groups']}")

    payload = {
        "model": args.model,
        "regime": "B",
        "epochs": args.epochs,
        "fold_seeds": FOLD_SEEDS,
        "degree_thresholds": {"low": "<=3", "medium": "4-19", "high": ">=20"},
        "results": results,
    }
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()