"""
Tests for BipartiteGCNLinkPredictor.
"""

from __future__ import annotations

import torch
import pytest

from models.gcn_link_predictor import BipartiteGCNLinkPredictor


class TestBipartiteGCNLinkPredictor:
    def test_forward_shape(self):
        model = BipartiteGCNLinkPredictor(n_mirnas=100, n_diseases=150, embed_dim=32, gcn_hidden=64)
        batch_size = 32
        mirna_idx = torch.randint(0, 100, (batch_size,))
        disease_idx = torch.randint(0, 150, (batch_size,))

        scores = model(mirna_idx, disease_idx)

        assert scores.shape == (batch_size,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_with_similarity_matrices(self):
        model = BipartiteGCNLinkPredictor(n_mirnas=50, n_diseases=60, embed_dim=16, gcn_hidden=32)
        batch_size = 16
        mirna_idx = torch.randint(0, 50, (batch_size,))
        disease_idx = torch.randint(0, 60, (batch_size,))

        K_mirna = torch.eye(50)
        K_disease = torch.eye(60)

        scores = model(mirna_idx, disease_idx, K_mirna=K_mirna, K_disease=K_disease)
        assert scores.shape == (batch_size,)

    def test_parameter_count_positive(self):
        model = BipartiteGCNLinkPredictor(n_mirnas=40, n_diseases=50, embed_dim=8, gcn_hidden=16)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0