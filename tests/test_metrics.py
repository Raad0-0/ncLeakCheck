"""
Tests for diagnostic metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.metrics import (
    compute_auprc,
    compute_auroc,
    compute_mcc,
    compute_macro_f1,
    compute_ece,
    compute_recall_at_k,
)


def _make_binary_classification_data(n_samples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n_samples, dtype=np.int32)
    # Simulate scores correlated with labels
    y_score = 0.3 * y_true + rng.random(n_samples).astype(np.float32) * 0.7
    y_score = np.clip(y_score, 0.0, 1.0)
    return y_true, y_score


class TestAUPRC:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=42)
        auc = compute_auprc(y_true, y_score)
        assert 0.0 <= auc <= 1.0


class TestAUROC:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=43)
        auroc = compute_auroc(y_true, y_score)
        assert 0.0 <= auroc <= 1.0


class TestMCC:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=44)
        mcc = compute_mcc(y_true, y_score, threshold=0.5)
        assert -1.0 <= mcc <= 1.0


class TestMacroF1:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=45)
        f1 = compute_macro_f1(y_true, y_score, threshold=0.5)
        assert 0.0 <= f1 <= 1.0


class TestECE:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=46)
        ece = compute_ece(y_true, y_score, n_bins=10)
        assert 0.0 <= ece <= 1.0


class TestRecallAtK:
    def test_basic(self):
        y_true, y_score = _make_binary_classification_data(seed=47)
        recall_10 = compute_recall_at_k(y_true, y_score, k=10)
        assert 0.0 <= recall_10 <= 1.0

    def test_k_larger_than_samples(self):
        y_true, y_score = _make_binary_classification_data(n_samples=50, seed=48)
        recall_100 = compute_recall_at_k(y_true, y_score, k=100)
        assert 0.0 <= recall_100 <= 1.0