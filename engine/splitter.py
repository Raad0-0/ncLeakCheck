"""
Regime A/B split engine for ncLeakCheck-Disease.
"""

from __future__ import annotations

import numpy as np

from .similarity_isolated import compute_isolated_gip_similarity


def split_edges(
    A_full: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split positive edges into train/test. Return A_train, A_test, train_pairs, test_pairs."""
    if A_full.ndim != 2:
        raise ValueError("A_full must be a 2D array")

    rng = np.random.default_rng(seed)

    pos_m, pos_d = np.where(A_full > 0)
    n_pos = len(pos_m)
    if n_pos == 0:
        raise ValueError("A_full has no positive edges to split")

    idx = rng.permutation(n_pos)
    n_test = max(1, int(np.round(n_pos * test_ratio)))

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    test_pairs = np.stack([pos_m[test_idx], pos_d[test_idx]], axis=1)
    train_pairs = np.stack([pos_m[train_idx], pos_d[train_idx]], axis=1)

    A_train = A_full.astype(np.float32, copy=True)
    A_test = np.zeros_like(A_train)

    for mi, di in test_pairs:
        A_train[mi, di] = 0.0
        A_test[mi, di] = 1.0

    return A_train, A_test, train_pairs, test_pairs


def split_regime_a(
    A_full: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Regime A: leaky transductive control.

    Compute GIP from A_full, then split edges.
    """
    K_mirna_full, K_disease_full = compute_isolated_gip_similarity(A_full)
    A_train, A_test, train_pairs, test_pairs = split_edges(
        A_full, test_ratio=test_ratio, seed=seed
    )

    return {
        "regime": "A",
        "A_train": A_train,
        "A_test": A_test,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "K_mirna": K_mirna_full,
        "K_disease": K_disease_full,
    }


def split_regime_b(
    A_full: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Regime B: clean transductive evaluation.

    Split edges first, then compute GIP from A_train only.
    """
    A_train, A_test, train_pairs, test_pairs = split_edges(
        A_full, test_ratio=test_ratio, seed=seed
    )
    K_mirna_train, K_disease_train = compute_isolated_gip_similarity(A_train)

    return {
        "regime": "B",
        "A_train": A_train,
        "A_test": A_test,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "K_mirna": K_mirna_train,
        "K_disease": K_disease_train,
    }

def split_regime_c(
    A_full: np.ndarray,
    precursor_clusters: dict,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Regime C: cold-miRNA (precursor-family holdout).

    Holds out entire miRNA precursor-family clusters.
    Test miRNAs have zero degree in A_train.

    Parameters
    ----------
    A_full : np.ndarray
        Full adjacency (n_mirnas, n_diseases).
    precursor_clusters : dict
        Mapping: cluster_id -> list of miRNA indices.
    test_ratio : float
        Fraction of clusters to hold out (not fraction of edges).
    seed : int
        Random seed.

    Returns
    -------
    dict with:
        A_train, A_test, train_pairs, test_pairs,
        train_mirnas, test_mirnas,
        regime: "C"
    """
    rng = np.random.default_rng(seed)
    n_mirnas, n_diseases = A_full.shape

    # Get list of clusters
    cluster_ids = list(precursor_clusters.keys())
    if len(cluster_ids) == 0:
        raise ValueError("precursor_clusters is empty")

    # Shuffle and split clusters
    rng.shuffle(cluster_ids)
    n_test_clusters = max(1, int(np.round(len(cluster_ids) * test_ratio)))
    test_cluster_ids = cluster_ids[:n_test_clusters]
    train_cluster_ids = cluster_ids[n_test_clusters:]

    # Build miRNA sets
    test_mirna_set = set()
    for cid in test_cluster_ids:
        test_mirna_set.update(precursor_clusters[cid])

    train_mirna_set = set(range(n_mirnas)) - test_mirna_set

    # Build A_train (mask all edges for test miRNAs)
    A_train = A_full.astype(np.float32, copy=True)
    A_test = np.zeros_like(A_train)

    for mi in test_mirna_set:
        A_test[mi, :] = A_train[mi, :]  # move test miRNA edges to A_test
        A_train[mi, :] = 0.0  # zero out in train

    # Extract train/test pairs
    train_m, train_d = np.where(A_train > 0)
    test_m, test_d = np.where(A_test > 0)

    train_pairs = np.stack([train_m, train_d], axis=1)
    test_pairs = np.stack([test_m, test_d], axis=1)
    K_mirna_train, K_disease_train = compute_isolated_gip_similarity(A_train)

        # Compute similarities strictly from A_train.
    # Held-out miRNA edges must not influence Regime C features.
    K_mirna_train, K_disease_train = compute_isolated_gip_similarity(
        A_train
    )

    return {
        "regime": "C",
        "A_train": A_train,
        "A_test": A_test,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_mirnas": np.array(
            sorted(train_mirna_set),
            dtype=np.int32,
        ),
        "test_mirnas": np.array(
            sorted(test_mirna_set),
            dtype=np.int32,
        ),
        "test_cluster_ids": test_cluster_ids,
        "train_cluster_ids": train_cluster_ids,
        "K_mirna": K_mirna_train,
        "K_disease": K_disease_train,
    }