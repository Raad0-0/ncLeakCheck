"""
Fold-isolated Gaussian Interaction Profile (GIP) similarity engine.

This module computes miRNA-side and disease-side GIP similarity matrices
strictly from a binary training adjacency matrix A_train, as required for
Regime B (clean transductive) evaluation in the ncLeakCheck-Disease benchmark.

Key properties:
- Input: A_train is a binary ndarray with shape (n_mirnas, n_diseases).
- Output: Two float32 similarity matrices:
    - K_mirna: (n_mirnas, n_mirnas)
    - K_disease: (n_diseases, n_diseases)
- Vectorized squared Euclidean distance:
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u^T v
- Normalized gamma:
    gamma = 1.0 / mean(||profile||^2)
- All-zero profiles are handled safely (no division by zero, finite output).
- Diagonal self-similarity is enforced to exactly 1.0.
- No mutation of A_train.
"""

from __future__ import annotations

import numpy as np


def _gip_similarity_from_profiles(
    profiles: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute GIP similarity matrix from a set of interaction profiles.

    Parameters
    ----------
    profiles : np.ndarray
        Interaction profiles with shape (n_entities, n_targets). Each row is
        the binary interaction profile of one entity (miRNA or disease).
    eps : float
        Small constant to avoid division by zero when computing gamma.

    Returns
    -------
    K : np.ndarray
        Symmetric GIP similarity matrix with shape (n_entities, n_entities),
        dtype float32, diagonal exactly 1.0, and all entries finite.
    """
    if profiles.ndim != 2:
        raise ValueError("profiles must be a 2D array")

    n_entities = profiles.shape[0]

    # Compute squared norms for each profile: ||u||^2
    sq_norms = np.sum(profiles**2, axis=1, keepdims=True)  # (n_entities, 1)

    # gamma = 1 / mean(||profile||^2), with safe handling for all-zero profiles
    mean_sq_norm = float(np.mean(sq_norms))
    if mean_sq_norm < eps:
        # All profiles are zero or near-zero; set gamma to a safe default
        gamma = 1.0
    else:
        gamma = 1.0 / mean_sq_norm

    # Vectorized squared Euclidean distances:
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u^T v
    sq_dists = sq_norms + sq_norms.T - 2.0 * (profiles @ profiles.T)

    # Guard against tiny negative values from floating-point error
    sq_dists = np.maximum(sq_dists, 0.0)

    # GIP kernel: K(i, j) = exp(-gamma * ||u - v||^2)
    K = np.exp(-gamma * sq_dists)

    # Enforce diagonal = 1.0 exactly
    np.fill_diagonal(K, 1.0)

    # Ensure float32 output
    K = K.astype(np.float32, copy=False)

    # Sanity check: all finite
    if not np.all(np.isfinite(K)):
        K = np.nan_to_num(K, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return K


def compute_isolated_gip_similarity(
    A_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fold-isolated GIP similarity matrices for miRNAs and diseases.

    This function implements the vectorized GIP kernel strictly from the
    training adjacency matrix A_train, as required for Regime B (clean
    transductive) evaluation.

    Parameters
    ----------
    A_train : np.ndarray
        Binary training adjacency matrix with shape (n_mirnas, n_diseases).
        Must be a 2D array of 0s and 1s. This array is not modified.

    Returns
    -------
    K_mirna : np.ndarray
        miRNA-side GIP similarity matrix with shape (n_mirnas, n_mirnas),
        dtype float32, symmetric, diagonal exactly 1.0, all entries finite.
    K_disease : np.ndarray
        Disease-side GIP similarity matrix with shape (n_diseases, n_diseases),
        dtype float32, symmetric, diagonal exactly 1.0, all entries finite.
    """
    if A_train.ndim != 2:
        raise ValueError("A_train must be a 2D array")

    n_mirnas, n_diseases = A_train.shape

    if n_mirnas == 0 or n_diseases == 0:
        raise ValueError("A_train must have at least one miRNA and one disease")

    # Ensure float32 and avoid mutation by copying if needed
    A_train = np.asarray(A_train, dtype=np.float32)

    # miRNA-side profiles: rows of A_train
    mirna_profiles = A_train  # (n_mirnas, n_diseases)

    # Disease-side profiles: columns of A_train (transpose)
    disease_profiles = A_train.T  # (n_diseases, n_mirnas)

    K_mirna = _gip_similarity_from_profiles(mirna_profiles)
    K_disease = _gip_similarity_from_profiles(disease_profiles)

    # Final shape and dtype checks (defensive)
    assert K_mirna.shape == (n_mirnas, n_mirnas), "K_mirna shape mismatch"
    assert K_disease.shape == (n_diseases, n_diseases), "K_disease shape mismatch"
    assert K_mirna.dtype == np.float32, "K_mirna dtype mismatch"
    assert K_disease.dtype == np.float32, "K_disease dtype mismatch"

    return K_mirna, K_disease


def compute_isolated_functional_similarity(
    A_train: np.ndarray,
    S_disease: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute fold-isolated functional similarity between miRNAs.

    This function computes a cosine-like functional similarity between miRNAs
    based on their disease association profiles in A_train, optionally weighted
    by a disease-disease functional similarity matrix S_disease.

    When S_disease is None, it is treated as an identity matrix, so the
    similarity reduces to cosine similarity over shared diseases.

    All computations use only A_train (no test edges), satisfying Regime B.

    Parameters
    ----------
    A_train : np.ndarray
        Binary training adjacency matrix with shape (n_mirnas, n_diseases).
    S_disease : np.ndarray | None
        Optional disease-disease functional similarity matrix with shape
        (n_diseases, n_diseases). If None, treated as identity.
    eps : float
        Small constant to avoid division by zero in normalization.

    Returns
    -------
    K_func_mirna : np.ndarray
        miRNA-miRNA functional similarity matrix with shape
        (n_mirnas, n_mirnas), dtype float32, symmetric, diagonal ~1.0,
        all entries finite.
    """
    if A_train.ndim != 2:
        raise ValueError("A_train must be a 2D array")

    n_mirnas, n_diseases = A_train.shape
    if n_mirnas == 0 or n_diseases == 0:
        raise ValueError("A_train must have at least one miRNA and one disease")

    A_train = np.asarray(A_train, dtype=np.float32)

    if S_disease is None:
        # Treat as identity: similarity based on raw dot products of profiles
        # K = P P^T, then cosine-normalized
        P = A_train  # (n_mirnas, n_diseases)
        Gram = P @ P.T  # (n_mirnas, n_mirnas)
    else:
        S_disease = np.asarray(S_disease, dtype=np.float32)
        if S_disease.shape != (n_diseases, n_diseases):
            raise ValueError(
                f"S_disease shape {S_disease.shape} does not match "
                f"expected ({n_diseases}, {n_diseases})"
            )
        # K = P S_disease P^T
        P = A_train  # (n_mirnas, n_diseases)
        PS = P @ S_disease  # (n_mirnas, n_diseases)
        Gram = PS @ P.T  # (n_mirnas, n_mirnas)

    # Cosine-like normalization:
    # K_func(i,j) = Gram(i,j) / sqrt(Gram(i,i) * Gram(j,j) + eps)
    diag = np.diag(Gram).astype(np.float32)  # (n_mirnas,)
    denom = np.sqrt(np.outer(diag, diag) + eps)
    K_func = Gram / denom

    # Ensure diagonal is exactly 1.0 (should be already, up to numerical error)
    np.fill_diagonal(K_func, 1.0)

    K_func = K_func.astype(np.float32, copy=False)

    if not np.all(np.isfinite(K_func)):
        K_func = np.nan_to_num(K_func, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return K_func


if __name__ == "__main__":
    print("Running quick self-check for GIP similarity engine...")

    # Tiny toy adjacency: 3 miRNAs x 4 diseases
    A = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 1],
        ],
        dtype=np.float32,
    )

    K_mirna, K_disease = compute_isolated_gip_similarity(A)

    print("A_train shape:", A.shape)
    print("K_mirna shape:", K_mirna.shape, "dtype:", K_mirna.dtype)
    print("K_disease shape:", K_disease.shape, "dtype:", K_disease.dtype)
    print("Diagonal K_mirna:", np.diag(K_mirna))
    print("Diagonal K_disease:", np.diag(K_disease))

    # Quick functional similarity self-check (S_disease = None)
    K_func = compute_isolated_functional_similarity(A, S_disease=None)
    print("K_func_mirna shape:", K_func.shape, "dtype:", K_func.dtype)
    print("Diagonal K_func_mirna:", np.diag(K_func))

    print("Quick self-check completed successfully.")