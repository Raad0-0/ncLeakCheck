"""
Smoke test for minimal training loop.
"""

from __future__ import annotations

import numpy as np
import torch

from engine.training_loop import run_fold
from models.decoupled_mlp import DecoupledMLP
from engine.training_loop import _make_test_loader

def test_test_loader_preserves_1_to_2_ratio() -> None:
    test_pairs = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    neg_pairs = np.array(
        [
            [0, 2],
            [1, 2],
            [0, 3],
            [1, 3],
        ],
        dtype=np.int32,
    )

    loader = _make_test_loader(
        test_pairs=test_pairs,
        neg_pairs=neg_pairs,
        batch_size=16,
    )

    mirna_idx, disease_idx, labels = next(iter(loader))

    assert len(mirna_idx) == 6
    assert len(disease_idx) == 6
    assert len(labels) == 6
    assert int((labels == 1).sum()) == 2
    assert int((labels == 0).sum()) == 4


def _make_small_graph(n_m: int = 50, n_d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n_m, n_d)) < 0.1).astype(np.int32)
    if np.sum(A) == 0:
        A[0, 0] = 1
    return A

 
def test_run_fold_decoupled_mlp_regime_b():
    A_full = _make_small_graph(seed=42)
    n_m, n_d = A_full.shape

    model = DecoupledMLP(n_mirnas=n_m, n_diseases=n_d, embed_dim=16, hidden_dim=32)

    metrics = run_fold(
        model,
        A_full,
        test_ratio=0.2,
        neg_ratio=2.0,
        n_epochs=2,  # very short smoke test
        batch_size=64,
        lr=1e-3,
        seed=7,
        regime="B",
        device=torch.device("cpu"),
    )

    assert 0.0 <= metrics["auprc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["n_train_pos"] > 0
    assert metrics["n_test_pos"] > 0