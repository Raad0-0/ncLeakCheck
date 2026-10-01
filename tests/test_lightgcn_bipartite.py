"""
Tests for the LightGCN bipartite baseline.
"""

from __future__ import annotations

import torch

from models.lightgcn_bipartite import LightGCNBipartite


def make_edge_index() -> torch.Tensor:
    # miRNA nodes: 0, 1, 2
    # disease nodes: 0, 1, 2
    return torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
        ],
        dtype=torch.long,
    )


def test_constructor_configuration() -> None:
    model = LightGCNBipartite(
        n_mirnas=10,
        n_diseases=15,
        embed_dim=64,
        n_layers=3,
    )

    assert model.embed_dim == 64
    assert model.n_layers == 3
    assert model.mirna_embedding.weight.shape == (10, 64)
    assert model.disease_embedding.weight.shape == (15, 64)


def test_forward_shape_and_probability_range() -> None:
    model = LightGCNBipartite(
        n_mirnas=10,
        n_diseases=15,
        embed_dim=64,
        n_layers=3,
    )

    mirna_idx = torch.tensor([0, 1, 2, 3])
    disease_idx = torch.tensor([0, 1, 2, 3])

    scores = model(
        mirna_idx,
        disease_idx,
        edge_index=make_edge_index(),
    )

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_propagation_runs_without_nan() -> None:
    model = LightGCNBipartite(
        n_mirnas=4,
        n_diseases=5,
        embed_dim=64,
        n_layers=3,
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=torch.long,
    )

    mirna_idx = torch.tensor([0, 1, 2, 3])
    disease_idx = torch.tensor([0, 1, 2, 3])

    scores = model(mirna_idx, disease_idx, edge_index=edge_index)

    assert torch.isfinite(scores).all()


def test_isolated_nodes_are_supported() -> None:
    model = LightGCNBipartite(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=64,
        n_layers=3,
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=torch.long,
    )

    mirna_idx = torch.tensor([2, 3, 4])
    disease_idx = torch.tensor([2, 3, 4])

    scores = model(mirna_idx, disease_idx, edge_index=edge_index)

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_parameter_count_positive() -> None:
    model = LightGCNBipartite(
        n_mirnas=10,
        n_diseases=15,
        embed_dim=64,
        n_layers=3,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) > 0