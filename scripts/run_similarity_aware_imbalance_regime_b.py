"""Five-fold Regime B negative-sampling sensitivity benchmark."""

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

from diagnostics.metrics import (
    compute_auprc,
    compute_auroc,
    compute_ece,
    compute_macro_f1,
    compute_mcc,
)
from engine.negative_sampler import sample_uniform_negatives
from engine.splitter import split_regime_b
from engine.test_negative_sampler import sample_uniform_test_negatives
from engine.training_loop import _make_test_loader, _make_train_loader, evaluate, train_epoch
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor

FOLD_SEEDS = [892, 7739, 6545, 4388, 4330]
RATIOS = [1.0, 2.0, 5.0, 10.0]


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


def run_one(model_name, A_full, ratio, seed, epochs, device):
    set_seed(seed)
    split = split_regime_b(A_full, test_ratio=0.2, seed=seed)
    train_pairs = split["train_pairs"]
    test_pairs = split["test_pairs"]
    n_mirnas, n_diseases = A_full.shape

    train_neg = sample_uniform_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=ratio, seed=seed + 1
    )
    test_neg = sample_uniform_test_negatives(
        train_pairs, test_pairs, n_mirnas, n_diseases, ratio=ratio, seed=seed + 2
    )

    train_loader = _make_train_loader(train_pairs, train_neg, batch_size=256, seed=seed)
    test_loader = _make_test_loader(test_pairs, test_neg, batch_size=256)

    model = create_model(model_name, n_mirnas, n_diseases).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    losses = []
    for _ in range(epochs):
        losses.append(
            train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                K_mirna=split["K_mirna"],
                K_disease=split["K_disease"],
            )
        )

    y_true, y_score = evaluate(
        model,
        test_loader,
        device,
        K_mirna=split["K_mirna"],
        K_disease=split["K_disease"],
    )

    return {
        "model": model_name,
        "regime": "B",
        "seed": int(seed),
        "negative_ratio": f"1:{int(ratio)}",
        "negative_ratio_numeric": ratio,
        "epochs": int(epochs),
        "batch_size": 256,
        "learning_rate": 1e-3,
        "similarity_aware": True,
        "n_train_pos": int(len(train_pairs)),
        "n_train_neg": int(len(train_neg)),
        "n_test_pos": int(len(test_pairs)),
        "n_test_neg": int(len(test_neg)),
        "final_loss": float(losses[-1]),
        "auprc": compute_auprc(y_true, y_score),
        "auroc": compute_auroc(y_true, y_score),
        "mcc": compute_mcc(y_true, y_score),
        "macro_f1": compute_macro_f1(y_true, y_score),
        "ece": compute_ece(y_true, y_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--data", default="data/processed/hmdd_v4_strict_adjacency.npy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    A_full = np.load(args.data).astype(np.float32)
    if A_full.ndim != 2:
        raise ValueError("Adjacency matrix must be two-dimensional.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"Adjacency shape: {A_full.shape}")
    print(f"Positive edges: {int(np.sum(A_full > 0))}")
    print(f"Using device: {device}")
    print(f"Fold seeds: {FOLD_SEEDS}")

    for ratio in RATIOS:
        print(f"\n--- Regime B ratio 1:{int(ratio)} ---")
        for fold, seed in enumerate(FOLD_SEEDS, start=1):
            result = run_one(args.model, A_full, ratio, seed, args.epochs, device)
            result["fold"] = fold
            results.append(result)
            print(
                f"Fold {fold}: AUPRC={result['auprc']:.4f}, "
                f"AUROC={result['auroc']:.4f}, "
                f"train={result['n_train_pos']}:{result['n_train_neg']}, "
                f"test={result['n_test_pos']}:{result['n_test_neg']}"
            )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project": "ncLeakCheck-Disease",
        "dataset": "hmdd_v4_human_strict",
        "regime": "B",
        "model": args.model,
        "similarity_aware": True,
        "fold_seeds": FOLD_SEEDS,
        "negative_ratios": ["1:1", "1:2", "1:5", "1:10"],
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main() 