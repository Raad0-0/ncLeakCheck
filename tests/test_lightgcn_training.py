from __future__ import annotations

import numpy as np
import torch

from engine.lightgcn_training import train_lightgcn_fold
from engine.splitter import split_regime_b
from models.lightgcn_bipartite_clean import BipartiteLightGCN


def test_lightgcn_fold_training_smoke() -> None:
    rng = np.random.default_rng(7)

    A_full = (rng.random((12, 10)) < 0.15).astype(np.float32)

    if A_full.sum() < 8:
        A_full[0, 0] = 1.0
        A_full[1, 1] = 1.0
        A_full[2, 2] = 1.0
        A_full[3, 3] = 1.0
        A_full[4, 4] = 1.0
        A_full[5, 5] = 1.0
        A_full[6, 6] = 1.0
        A_full[7, 7] = 1.0

    split = split_regime_b(
        A_full,
        test_ratio=0.2,
        seed=42,
    )

    model = BipartiteLightGCN(
        n_mirnas=A_full.shape[0],
        n_diseases=A_full.shape[1],
        embed_dim=8,
        n_layers=1,
    )

    metrics = train_lightgcn_fold(
        model=model,
        A_train=split["A_train"],
        train_pairs=split["train_pairs"],
        test_pairs=split["test_pairs"],
        n_mirnas=A_full.shape[0],
        n_diseases=A_full.shape[1],
        neg_ratio=1.0,
        n_epochs=2,
        batch_size=16,
        lr=1e-3,
        seed=42,
        device=torch.device("cpu"),
    )

    assert np.isfinite(metrics["auprc"])
    assert np.isfinite(metrics["auroc"])
    assert 0.0 <= metrics["auprc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0

    assert np.isfinite(metrics["final_loss"])
    assert metrics["final_loss"] >= 0.0
    assert metrics["n_train_pos"] == len(split["train_pairs"])
    assert metrics["n_test_pos"] == len(split["test_pairs"])