"""
Unit tests for the fold-isolated GIP and functional similarity engines.

These tests verify:
- Matrix shapes are correct.
- Similarity matrices are symmetric.
- Diagonal elements are exactly 1.0.
- Zero-degree profiles produce finite values (no NaN/Inf).
- A_train is not mutated.
- A held-out edge cannot affect Regime-B similarity when absent from A_train.
- Functional similarity behaves correctly with and without S_disease.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.similarity_isolated import (
    compute_isolated_gip_similarity,
    compute_isolated_functional_similarity,
)


def _make_binary_matrix(n_rows: int, n_cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_rows, n_cols), dtype=np.int32)


class TestGIPShapeAndBasicProperties:
    def test_matrix_shapes_small(self):
        A = _make_binary_matrix(10, 15, seed=42)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)
        assert K_mirna.shape == (10, 10), "miRNA similarity shape mismatch"
        assert K_disease.shape == (15, 15), "disease similarity shape mismatch"

    def test_matrix_shapes_larger(self):
        A = _make_binary_matrix(100, 200, seed=1337)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)
        assert K_mirna.shape == (100, 100)
        assert K_disease.shape == (200, 200)

    def test_symmetry(self):
        A = _make_binary_matrix(50, 70, seed=2026)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)

        # Symmetry: K == K.T
        assert np.allclose(K_mirna, K_mirna.T, atol=1e-6), "K_mirna not symmetric"
        assert np.allclose(K_disease, K_disease.T, atol=1e-6), "K_disease not symmetric"

    def test_diagonal_is_one(self):
        A = _make_binary_matrix(30, 40, seed=7)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)

        diag_mirna = np.diag(K_mirna)
        diag_disease = np.diag(K_disease)

        assert np.all(diag_mirna == 1.0), "K_mirna diagonal not exactly 1.0"
        assert np.all(diag_disease == 1.0), "K_disease diagonal not exactly 1.0"

    def test_dtype_is_float32(self):
        A = _make_binary_matrix(20, 25, seed=99)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)
        assert K_mirna.dtype == np.float32
        assert K_disease.dtype == np.float32


class TestZeroDegreeProfiles:
    def test_zero_row_and_zero_column_finite(self):
        # Construct a matrix with at least one all-zero row and one all-zero column
        A = np.zeros((10, 12), dtype=np.float32)
        # Add some edges, but leave row 3 and column 5 all-zero
        A[0, 1] = 1.0
        A[1, 2] = 1.0
        A[2, 0] = 1.0
        # row 3 remains all zeros
        # column 5 remains all zeros

        K_mirna, K_disease = compute_isolated_gip_similarity(A)

        assert np.all(np.isfinite(K_mirna)), "K_mirna has non-finite values"
        assert np.all(np.isfinite(K_disease)), "K_disease has non-finite values"

        # Diagonal must still be 1.0 even for zero-degree entities
        assert K_mirna[3, 3] == 1.0
        assert K_disease[5, 5] == 1.0

    def test_all_zero_matrix(self):
        A = np.zeros((8, 9), dtype=np.float32)
        K_mirna, K_disease = compute_isolated_gip_similarity(A)

        assert K_mirna.shape == (8, 8)
        assert K_disease.shape == (9, 9)
        assert np.all(np.isfinite(K_mirna))
        assert np.all(np.isfinite(K_disease))
        assert np.allclose(np.diag(K_mirna), 1.0)
        assert np.allclose(np.diag(K_disease), 1.0)


class TestNoMutation:
    def test_A_train_not_mutated(self):
        A_original = _make_binary_matrix(25, 30, seed=123)
        A_copy = A_original.copy()

        _ = compute_isolated_gip_similarity(A_original)

        assert np.array_equal(A_original, A_copy), "A_train was mutated"


class TestRegimeBLeakageIsolation:
    """
    Verify that a held-out edge cannot affect Regime-B similarity matrices
    when that edge is absent from A_train.
    """

    def test_held_out_edge_has_no_effect_on_similarity(self):
        # Start from a base adjacency
        rng = np.random.default_rng(42)
        A_full = rng.integers(0, 2, size=(50, 60), dtype=np.int32)

        # Ensure there is at least one positive edge to hold out
        pos_indices = np.argwhere(A_full == 1)
        if len(pos_indices) == 0:
            A_full[0, 0] = 1
            pos_indices = np.argwhere(A_full == 1)

        i, j = pos_indices[0]

        # Regime B: split first, then compute similarity from A_train only
        A_train = A_full.copy()
        A_train[i, j] = 0  # hold out this edge

        K_mirna_train, K_disease_train = compute_isolated_gip_similarity(A_train)

        # Check that K_train is well-formed:
        assert np.all(np.isfinite(K_mirna_train))
        assert np.all(np.isfinite(K_disease_train))
        assert np.allclose(np.diag(K_mirna_train), 1.0)
        assert np.allclose(np.diag(K_disease_train), 1.0)

        # Determinism: recomputing from the same A_train yields identical K
        K_mirna_train_2, K_disease_train_2 = compute_isolated_gip_similarity(A_train)
        assert np.array_equal(K_mirna_train, K_mirna_train_2)
        assert np.array_equal(K_disease_train, K_disease_train_2)

        # Stronger Regime-B check:
        # Create a second full matrix that differs only in the held-out edge,
        # but with the SAME A_train after masking. Similarities must be identical.
        A_full_alt = A_full.copy()
        # Flip the held-out edge in the alternative full matrix
        A_full_alt[i, j] = 1 - A_full_alt[i, j]

        A_train_alt = A_full_alt.copy()
        A_train_alt[i, j] = 0  # mask the same edge

        # A_train and A_train_alt should be identical now
        assert np.array_equal(A_train, A_train_alt), "A_train matrices should match"

        K_mirna_train_alt, K_disease_train_alt = compute_isolated_gip_similarity(
            A_train_alt
        )

        # Since A_train == A_train_alt, similarities must be identical
        assert np.array_equal(K_mirna_train, K_mirna_train_alt), (
            "Regime-B miRNA similarity should not depend on held-out edge"
        )
        assert np.array_equal(K_disease_train, K_disease_train_alt), (
            "Regime-B disease similarity should not depend on held-out edge"
        )


class TestFunctionalSimilarityBasic:
    def test_shape_and_dtype_no_S_disease(self):
        A = _make_binary_matrix(12, 18, seed=5)
        K_func = compute_isolated_functional_similarity(A, S_disease=None)
        assert K_func.shape == (12, 12)
        assert K_func.dtype == np.float32

    def test_symmetry_and_diagonal_no_S_disease(self):
        A = _make_binary_matrix(20, 25, seed=9)
        K_func = compute_isolated_functional_similarity(A, S_disease=None)
        assert np.allclose(K_func, K_func.T, atol=1e-6), "K_func not symmetric"
        assert np.allclose(np.diag(K_func), 1.0, atol=1e-6), "K_func diagonal not 1.0"

    def test_finite_for_zero_profiles(self):
        A = np.zeros((8, 10), dtype=np.float32)
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        # rows 1,3..7 are zero
        K_func = compute_isolated_functional_similarity(A, S_disease=None)
        assert np.all(np.isfinite(K_func))
        assert np.allclose(np.diag(K_func), 1.0, atol=1e-6)


class TestFunctionalSimilarityWithS:
    def test_shape_and_dtype_with_S_disease(self):
        n_m, n_d = 10, 15
        A = _make_binary_matrix(n_m, n_d, seed=3)
        rng = np.random.default_rng(7)
        S = rng.random((n_d, n_d)).astype(np.float32)
        S = (S + S.T) / 2.0  # symmetrize

        K_func = compute_isolated_functional_similarity(A, S_disease=S)
        assert K_func.shape == (n_m, n_m)
        assert K_func.dtype == np.float32

    def test_symmetry_and_diagonal_with_S_disease(self):
        n_m, n_d = 14, 20
        A = _make_binary_matrix(n_m, n_d, seed=11)
        rng = np.random.default_rng(13)
        S = rng.random((n_d, n_d)).astype(np.float32)
        S = (S + S.T) / 2.0

        K_func = compute_isolated_functional_similarity(A, S_disease=S)
        assert np.allclose(K_func, K_func.T, atol=1e-6), "K_func not symmetric"
        assert np.allclose(np.diag(K_func), 1.0, atol=1e-6), "K_func diagonal not 1.0"


class TestFunctionalNoMutation:
    def test_A_train_not_mutated_functional(self):
        A_original = _make_binary_matrix(15, 20, seed=17)
        A_copy = A_original.copy()

        _ = compute_isolated_functional_similarity(A_original, S_disease=None)

        assert np.array_equal(A_original, A_copy), "A_train was mutated in functional sim"