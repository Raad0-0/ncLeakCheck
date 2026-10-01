"""
Tests for Regime C (cold-miRNA) split engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.splitter import split_regime_c


def _make_test_graph_with_clusters(n_m: int = 50, n_d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n_m, n_d)) < 0.1).astype(np.int32)
    if np.sum(A) == 0:
        A[0, 0] = 1

    # Create dummy precursor clusters (e.g., 10 clusters)
    n_clusters = 10
    clusters = {}
    rng = np.random.default_rng(seed + 1)
    mirna_indices = rng.permutation(n_m)
    cluster_size = n_m // n_clusters

    for i in range(n_clusters):
        start = i * cluster_size
        end = (i + 1) * cluster_size if i < n_clusters - 1 else n_m
        clusters[f"cluster_{i}"] = mirna_indices[start:end].tolist()

    return A, clusters


class TestRegimeC:
    def test_structure(self):
        A_full, clusters = _make_test_graph_with_clusters(seed=42)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=7)

        assert result["regime"] == "C"
        assert result["A_train"].shape == A_full.shape
        assert result["A_test"].shape == A_full.shape
        assert len(result["train_mirnas"]) > 0
        assert len(result["test_mirnas"]) > 0

    def test_test_mirnas_zero_degree_in_train(self):
        A_full, clusters = _make_test_graph_with_clusters(seed=43)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=9)

        A_train = result["A_train"]
        test_mirnas = result["test_mirnas"]

        # Test miRNAs should have zero degree in A_train
        for mi in test_mirnas:
            assert np.sum(A_train[mi, :]) == 0, f"Test miRNA {mi} has non-zero degree in A_train"

    def test_train_test_mirnas_disjoint(self):
        A_full, clusters = _make_test_graph_with_clusters(seed=44)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=11)

        train_set = set(result["train_mirnas"])
        test_set = set(result["test_mirnas"])

        assert len(train_set & test_set) == 0

    def test_all_edges_accounted_for(self):
        A_full, clusters = _make_test_graph_with_clusters(seed=45)
        result = split_regime_c(A_full, clusters, test_ratio=0.2, seed=13)

        A_train = result["A_train"]
        A_test = result["A_test"]

        # A_train + A_test should reconstruct A_full
        A_recon = (A_train + A_test) > 0
        A_orig = A_full > 0
        assert np.array_equal(A_recon, A_orig)