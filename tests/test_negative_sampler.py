"""
Tests for negative sampling engines.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.negative_sampler import (
    sample_uniform_negatives,
    sample_degree_matched_negatives,
    sample_all_unobserved_negatives,
)


def _make_test_graph_and_pairs(
    n_m: int = 50,
    n_d: int = 60,
    density: float = 0.05,
    seed: int = 0,
):
    """
    Create a sparse binary graph and split into train/test.

    density: fraction of positive edges (e.g., 0.05 => ~5% positives).
    """
    rng = np.random.default_rng(seed)

    # Sparse random graph
    A = rng.random((n_m, n_d)) < density
    A = A.astype(np.int32)

    # Ensure at least some positives
    if np.sum(A) == 0:
        A[0, 0] = 1

    pos_m, pos_d = np.where(A > 0)
    all_pairs = np.stack([pos_m, pos_d], axis=1)

    # Split into train/test deterministically
    n_test = max(1, len(all_pairs) // 5)
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    A_train = A.astype(np.float32, copy=True)
    for mi, di in test_pairs:
        A_train[mi, di] = 0.0

    return A, A_train, train_pairs, test_pairs


class TestUniformNegatives:
    def test_shape_and_ratio(self):
        A_full, A_train, train_pairs, test_pairs = _make_test_graph_and_pairs(
            density=0.05, seed=42
        )
        n_m, n_d = A_full.shape
        ratio = 2.0

        neg_pairs = sample_uniform_negatives(
            train_pairs, test_pairs, n_m, n_d, ratio=ratio, seed=7
        )

        n_pos = len(train_pairs)
        n_neg_expected = max(1, int(np.round(n_pos * ratio)))
        assert neg_pairs.shape == (n_neg_expected, 2)

    def test_no_overlap_with_observed(self):
        A_full, A_train, train_pairs, test_pairs = _make_test_graph_and_pairs(
            density=0.05, seed=13
        )
        n_m, n_d = A_full.shape

        neg_pairs = sample_uniform_negatives(
            train_pairs, test_pairs, n_m, n_d, ratio=2.0, seed=9
        )

        train_set = set(map(tuple, train_pairs))
        test_set = set(map(tuple, test_pairs))
        observed = train_set | test_set
        neg_set = set(map(tuple, neg_pairs))

        assert len(observed & neg_set) == 0


class TestDegreeMatchedNegatives:
    def test_shape_basic(self):
        A_full, A_train, train_pairs, test_pairs = _make_test_graph_and_pairs(
            density=0.05, seed=17
        )
        n_m, n_d = A_train.shape
        ratio = 2.0

        neg_pairs = sample_degree_matched_negatives(
            train_pairs, test_pairs, A_train, ratio=ratio, seed=19
        )

        n_pos = len(train_pairs)
        n_neg_target = max(1, int(np.round(n_pos * ratio)))

        # May be less if not enough candidates, but should be > 0
        assert len(neg_pairs) > 0
        assert neg_pairs.shape[1] == 2

    def test_no_overlap_with_observed(self):
        A_full, A_train, train_pairs, test_pairs = _make_test_graph_and_pairs(
            density=0.05, seed=23
        )
        n_m, n_d = A_train.shape

        neg_pairs = sample_degree_matched_negatives(
            train_pairs, test_pairs, A_train, ratio=1.0, seed=29
        )

        train_set = set(map(tuple, train_pairs))
        test_set = set(map(tuple, test_pairs))
        observed = train_set | test_set
        neg_set = set(map(tuple, neg_pairs))

        assert len(observed & neg_set) == 0


class TestAllUnobserved:
    def test_exhaustive_and_disjoint(self):
        A_full, A_train, train_pairs, test_pairs = _make_test_graph_and_pairs(
            density=0.05, seed=31
        )
        n_m, n_d = A_full.shape

        all_neg = sample_all_unobserved_negatives(train_pairs, test_pairs, n_m, n_d)

        train_set = set(map(tuple, train_pairs))
        test_set = set(map(tuple, test_pairs))
        observed = train_set | test_set
        all_set = set(map(tuple, all_neg))

        # No overlap
        assert len(observed & all_set) == 0

        # Total pairs = observed + unobserved
        total_pairs = n_m * n_d
        assert len(observed) + len(all_set) == total_pairs