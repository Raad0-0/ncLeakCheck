from __future__ import annotations

import numpy as np


def sample_uniform_test_negatives(
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    n_mirnas: int,
    n_diseases: int,
    ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample test negatives based on the number of test positives.
    """
    rng = np.random.default_rng(seed)

    observed = set(map(tuple, train_pairs)) | set(map(tuple, test_pairs))

    n_target = max(
        1,
        int(np.round(len(test_pairs) * ratio)),
    )

    negatives = []
    attempts = 0
    max_attempts = n_target * 100

    while len(negatives) < n_target and attempts < max_attempts:
        mi = int(rng.integers(0, n_mirnas))
        di = int(rng.integers(0, n_diseases))
        pair = (mi, di)

        attempts += 1

        if pair not in observed:
            negatives.append(pair)
            observed.add(pair)

    if len(negatives) < n_target:
        raise RuntimeError(
            f"Only generated {len(negatives)}/{n_target} test negatives."
        )

    return np.asarray(negatives, dtype=np.int32)