"""
Formal leakage unit tests for splitter (Day 13).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.splitter import split_regime_a, split_regime_b, split_regime_c, split_edges


def _make_test_graph(n_m: int = 50, n_d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n_m, n_d)) < 0.1).astype(np.int32)
    if np.sum(A) == 0:
        A[0, 0] = 1
    return A


def _make_clusters(n_m: int = 50, n_clusters: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    mirna_indices = rng.permutation(n_m)
    cluster_size = n_m // n_clusters
    clusters = {}
    for i in range(n_clusters):
        start = i * cluster_size
        end = (i + 1) * cluster_size if i < n_clusters - 1 else n_m
        clusters[f"cluster_{i}"] = mirna_indices[start:end].tolist()
    return clusters


class TestLeakageEdgeDisjoint:
    def test_train_test_edges_disjoint_regime_a(self):
        A_full = _make_test_graph(seed=42)
        result = split_regime_a(A_full, test_ratio=0.2, seed=7)

        train_set = set(map(tuple, result["train_pairs"]))
        test_set = set(map(tuple, result["test_pairs"]))
        assert len(train_set & test_set) == 0

    def test_train_test_edges_disjoint_regime_b(self):
        A_full = _make_test_graph(seed=43)
        result = split_regime_b(A_full, test_ratio=0.2, seed=9)

        train_set = set(map(tuple, result["train_pairs"]))
        test_set = set(map(tuple, result["test_pairs"]))
        assert len(train_set & test_set) == 0

    def test_train_test_edges_disjoint_regime_c(self):
        A_full = _make_test_graph(seed=44)
        clusters = _make_clusters(n_m=A_full.shape[0], seed=45)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=11)

        train_set = set(map(tuple, result["train_pairs"]))
        test_set = set(map(tuple, result["test_pairs"]))
        assert len(train_set & test_set) == 0


class TestLeakageSimilaritySource:
    def test_regime_a_uses_full_graph(self):
        A_full = _make_test_graph(seed=46)
        result_a = split_regime_a(A_full, test_ratio=0.2, seed=13)

        from engine.similarity_isolated import compute_isolated_gip_similarity
        K_m_full, K_d_full = compute_isolated_gip_similarity(A_full)

        assert np.allclose(result_a["K_mirna"], K_m_full, atol=1e-6)
        assert np.allclose(result_a["K_disease"], K_d_full, atol=1e-6)

    def test_regime_b_uses_train_graph(self):
        A_full = _make_test_graph(seed=47)
        result_b = split_regime_b(A_full, test_ratio=0.2, seed=15)

        from engine.similarity_isolated import compute_isolated_gip_similarity
        K_m_train, K_d_train = compute_isolated_gip_similarity(result_b["A_train"])

        assert np.allclose(result_b["K_mirna"], K_m_train, atol=1e-6)
        assert np.allclose(result_b["K_disease"], K_d_train, atol=1e-6)


class TestRegimeCColdStart:
    def test_test_mirnas_zero_degree_in_train(self):
        A_full = _make_test_graph(seed=48)
        clusters = _make_clusters(n_m=A_full.shape[0], seed=49)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=17)

        A_train = result["A_train"]
        test_mirnas = result["test_mirnas"]

        for mi in test_mirnas:
            assert np.sum(A_train[mi, :]) == 0, f"Test miRNA {mi} has non-zero degree"

    def test_train_test_mirnas_disjoint(self):
        A_full = _make_test_graph(seed=50)
        clusters = _make_clusters(n_m=A_full.shape[0], seed=51)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=19)

        train_set = set(result["train_mirnas"])
        test_set = set(result["test_mirnas"])
        assert len(train_set & test_set) == 0


class TestLeakageDifference:
    def test_regime_a_b_similarity_differ(self):
        A_full = _make_test_graph(seed=52)
        result_a = split_regime_a(A_full, test_ratio=0.2, seed=21)
        result_b = split_regime_b(A_full, test_ratio=0.2, seed=21)

        # Same split seed => same edges, different similarity source
        assert np.array_equal(result_a["train_pairs"], result_b["train_pairs"])
        assert np.array_equal(result_a["test_pairs"], result_b["test_pairs"])

        # Similarity matrices should differ (at least slightly)
        diff_m = np.abs(result_a["K_mirna"] - result_b["K_mirna"]).max()
        diff_d = np.abs(result_a["K_disease"] - result_b["K_disease"]).max()

        assert np.isfinite(diff_m)
        assert np.isfinite(diff_d)
        # In practice diff > 0 for non-trivial graphs
        assert diff_m > 1e-6 or diff_d > 1e-6


class TestReconstruction:
    def test_a_train_plus_a_test_equals_a_full(self):
        A_full = _make_test_graph(seed=53)

        result_a = split_regime_a(A_full, test_ratio=0.2, seed=23)
        result_b = split_regime_b(A_full, test_ratio=0.2, seed=23)
        result_c = split_regime_c(A_full, _make_clusters(A_full.shape[0], seed=24), test_ratio=0.2, seed=25)

        for name, result in [("A", result_a), ("B", result_b), ("C", result_c)]:
            A_recon = (result["A_train"] + result["A_test"]) > 0
            assert np.array_equal(A_recon, A_full > 0), f"{name}: reconstruction failed"