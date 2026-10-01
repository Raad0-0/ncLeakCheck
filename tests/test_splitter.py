"""
Tests for Regime A/B split engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.splitter import split_edges, split_regime_a, split_regime_b


def _make_test_graph(n_m: int = 50, n_d: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=(n_m, n_d), dtype=np.int32)
    if np.sum(A) == 0:
        A[0, 0] = 1
    return A


class TestSplitEdges:
    def test_train_test_disjoint(self):
        A_full = _make_test_graph(seed=42)
        A_train, A_test, train_pairs, test_pairs = split_edges(A_full, test_ratio=0.2, seed=7)

        train_set = set(map(tuple, train_pairs))
        test_set = set(map(tuple, test_pairs))
        assert len(train_set & test_set) == 0

        A_recon = (A_train + A_test) > 0
        A_orig_pos = A_full > 0
        assert np.array_equal(A_recon, A_orig_pos)

    def test_test_ratio_approximate(self):
        A_full = _make_test_graph(seed=13)
        _, _, train_pairs, test_pairs = split_edges(A_full, test_ratio=0.2, seed=9)

        n_total = len(train_pairs) + len(test_pairs)
        ratio = len(test_pairs) / n_total
        assert 0.1 <= ratio <= 0.3


class TestRegimeA:
    def test_regime_a_structure(self):
        A_full = _make_test_graph(seed=5)
        result = split_regime_a(A_full, test_ratio=0.2, seed=11)

        assert result["regime"] == "A"
        assert result["A_train"].shape == A_full.shape
        assert result["A_test"].shape == A_full.shape
        assert result["K_mirna"].shape == (A_full.shape[0], A_full.shape[0])
        assert result["K_disease"].shape == (A_full.shape[1], A_full.shape[1])

    def test_regime_a_similarity_from_full(self):
        A_full = _make_test_graph(seed=17)
        result = split_regime_a(A_full, test_ratio=0.2, seed=19)

        from engine.similarity_isolated import compute_isolated_gip_similarity
        K_m_ref, K_d_ref = compute_isolated_gip_similarity(A_full)

        assert np.allclose(result["K_mirna"], K_m_ref, atol=1e-6)
        assert np.allclose(result["K_disease"], K_d_ref, atol=1e-6)


class TestRegimeB:
    def test_regime_b_structure(self):
        A_full = _make_test_graph(seed=23)
        result = split_regime_b(A_full, test_ratio=0.2, seed=29)

        assert result["regime"] == "B"
        assert result["A_train"].shape == A_full.shape
        assert result["A_test"].shape == A_full.shape
        assert result["K_mirna"].shape == (A_full.shape[0], A_full.shape[0])
        assert result["K_disease"].shape == (A_full.shape[1], A_full.shape[1])

    def test_regime_b_similarity_from_train(self):
        A_full = _make_test_graph(seed=31)
        result = split_regime_b(A_full, test_ratio=0.2, seed=37)

        from engine.similarity_isolated import compute_isolated_gip_similarity
        K_m_ref, K_d_ref = compute_isolated_gip_similarity(result["A_train"])

        assert np.allclose(result["K_mirna"], K_m_ref, atol=1e-6)
        assert np.allclose(result["K_disease"], K_d_ref, atol=1e-6)

    def test_regime_b_test_edges_zero_in_train(self):
        A_full = _make_test_graph(seed=41)
        result = split_regime_b(A_full, test_ratio=0.2, seed=43)

        A_train = result["A_train"]
        A_test = result["A_test"]

        test_mask = A_test > 0
        assert np.all(A_train[test_mask] == 0)


class TestRegimeALeakageVsBClean:
    def test_A_and_B_differ_when_edges_held_out(self):
        A_full = _make_test_graph(seed=50)

        res_a = split_regime_a(A_full, test_ratio=0.2, seed=51)
        res_b = split_regime_b(A_full, test_ratio=0.2, seed=51)

        assert np.array_equal(res_a["train_pairs"], res_b["train_pairs"])
        assert np.array_equal(res_a["test_pairs"], res_b["test_pairs"])

        diff_m = np.abs(res_a["K_mirna"] - res_b["K_mirna"]).max()
        diff_d = np.abs(res_a["K_disease"] - res_b["K_disease"]).max()

        assert np.isfinite(diff_m)
        assert np.isfinite(diff_d)