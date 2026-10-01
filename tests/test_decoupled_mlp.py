"""
Tests for DecoupledMLP baseline.
"""

from __future__ import annotations

import torch
import pytest

from models.decoupled_mlp import DecoupledMLP


class TestDecoupledMLP:
    def test_forward_shape(self):
        model = DecoupledMLP(n_mirnas=100, n_diseases=150, embed_dim=32, hidden_dim=64)
        batch_size = 32
        mirna_idx = torch.randint(0, 100, (batch_size,))
        disease_idx = torch.randint(0, 150, (batch_size,))

        scores = model(mirna_idx, disease_idx)

        assert scores.shape == (batch_size,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_parameter_count_positive(self):
        model = DecoupledMLP(n_mirnas=50, n_diseases=60, embed_dim=16, hidden_dim=32)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_embedding_indices_in_range(self):
        model = DecoupledMLP(n_mirnas=40, n_diseases=50, embed_dim=8, hidden_dim=16)
        mirna_idx = torch.tensor([0, 10, 39])
        disease_idx = torch.tensor([0, 25, 49])

        scores = model(mirna_idx, disease_idx)
        assert scores.shape == (3,)