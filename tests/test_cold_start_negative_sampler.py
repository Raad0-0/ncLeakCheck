from __future__ import annotations

import numpy as np

from engine.cold_start_negative_sampler import (
    sample_cold_mirna_negatives,
)


def test_negatives_use_only_test_mirnas() -> None:
    train_pairs = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    test_pairs = np.array(
        [
            [2, 0],
            [2, 1],
            [3, 2],
        ],
        dtype=np.int32,
    )

    test_mirnas = np.array([2, 3], dtype=np.int32)

    negatives = sample_cold_mirna_negatives(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        test_mirnas=test_mirnas,
        n_diseases=5,
        ratio=2.0,
        seed=42,
    )

    assert negatives.shape == (6, 2)
    assert set(negatives[:, 0]).issubset({2, 3})


def test_negatives_do_not_overlap_observed_pairs() -> None:
    train_pairs = np.array(
        [
            [0, 0],
            [1, 1],
        ],
        dtype=np.int32,
    )

    test_pairs = np.array(
        [
            [2, 0],
            [2, 1],
        ],
        dtype=np.int32,
    )

    test_mirnas = np.array([2], dtype=np.int32)

    negatives = sample_cold_mirna_negatives(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        test_mirnas=test_mirnas,
        n_diseases=6,
        ratio=2.0,
        seed=7,
    )

    observed = set(map(tuple, train_pairs)) | set(map(tuple, test_pairs))
    sampled = set(map(tuple, negatives))

    assert observed.isdisjoint(sampled)


def test_invalid_empty_test_mirnas() -> None:
    train_pairs = np.array([[0, 0]], dtype=np.int32)
    test_pairs = np.array([[1, 1]], dtype=np.int32)

    try:
        sample_cold_mirna_negatives(
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            test_mirnas=np.array([], dtype=np.int32),
            n_diseases=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty test_mirnas.")