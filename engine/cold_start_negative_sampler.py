"""
Negative sampling restricted to held-out cold-start nodes.
"""

from __future__ import annotations

import numpy as np


def sample_cold_mirna_negatives(
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    test_mirnas: np.ndarray,
    n_diseases: int,
    ratio: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample unobserved negatives using only held-out miRNAs.

    This keeps positive and negative test pairs on the same cold-miRNA
    node population.
    """
    rng = np.random.default_rng(seed)

    train_set = set(map(tuple, train_pairs))
    test_set = set(map(tuple, test_pairs))
    observed = train_set | test_set

    test_mirnas = np.asarray(test_mirnas, dtype=np.int32)

    if test_mirnas.ndim != 1 or len(test_mirnas) == 0:
        raise ValueError("test_mirnas must be a non-empty 1D array.")

    n_target = max(1, int(np.round(len(test_pairs) * ratio)))

    negatives = []
    attempts = 0
    max_attempts = n_target * 100

    while len(negatives) < n_target and attempts < max_attempts:
        mirna_idx = int(rng.choice(test_mirnas))
        disease_idx = int(rng.integers(0, n_diseases))
        pair = (mirna_idx, disease_idx)

        attempts += 1

        if pair not in observed:
            negatives.append(pair)
            observed.add(pair)

    if len(negatives) < n_target:
        raise RuntimeError(
            f"Only generated {len(negatives)}/{n_target} cold-miRNA negatives."
        )

    return np.asarray(negatives, dtype=np.int32)