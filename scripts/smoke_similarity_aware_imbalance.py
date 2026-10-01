from __future__ import annotations

import numpy as np
import torch
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.negative_sampler import sample_uniform_negatives
from engine.splitter import split_regime_b
from engine.test_negative_sampler import sample_uniform_test_negatives
from engine.training_loop import (
    _make_test_loader,
    _make_train_loader,
    evaluate,
    train_epoch,
)
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor


A_full = np.load(
    "data/processed/hmdd_v4_strict_adjacency.npy"
).astype(np.float32)

device = torch.device("cpu")
seed = 892
split = split_regime_b(A_full, test_ratio=0.2, seed=seed)

n_mirnas, n_diseases = A_full.shape
train_pairs = split["train_pairs"]
test_pairs = split["test_pairs"]

for model_name in ["mlp", "gcn"]:
    for ratio in [1.0, 2.0, 5.0, 10.0]:
        if model_name == "mlp":
            model = DecoupledMLP(
                n_mirnas=n_mirnas,
                n_diseases=n_diseases,
                embed_dim=32,
                hidden_dim=64,
            )
        else:
            model = BipartiteGCNLinkPredictor(
                n_mirnas=n_mirnas,
                n_diseases=n_diseases,
                embed_dim=32,
                gcn_hidden=64,
            )

        train_neg = sample_uniform_negatives(
            train_pairs,
            test_pairs,
            n_mirnas,
            n_diseases,
            ratio=ratio,
            seed=seed + 1,
        )

        test_neg = sample_uniform_test_negatives(
            train_pairs,
            test_pairs,
            n_mirnas,
            n_diseases,
            ratio=ratio,
            seed=seed + 2,
        )

        train_loader = _make_train_loader(
            train_pairs,
            train_neg,
            batch_size=256,
            seed=seed,
        )
        test_loader = _make_test_loader(
            test_pairs,
            test_neg,
            batch_size=256,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        for _ in range(2):
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

        print(
            model_name,
            f"1:{int(ratio)}",
            f"test={len(test_pairs)}:{len(test_neg)}",
            f"finite={np.isfinite(y_score).all()}",
            f"score_min={y_score.min():.6f}",
            f"score_max={y_score.max():.6f}",
        )