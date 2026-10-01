from __future__ import annotations

import torch

from models.lightgcn_bipartite_clean import BipartiteLightGCN


def make_edge_index() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
        ],
        dtype=torch.long,
    )


def test_constructor_shapes() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
    )

    assert model.mirna_embedding.weight.shape == (5, 16)
    assert model.disease_embedding.weight.shape == (6, 16)
    assert model.n_layers == 2


def test_propagation_shapes_and_finiteness() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
    )

    mirna_emb, disease_emb = model.propagate(make_edge_index())

    assert mirna_emb.shape == (5, 16)
    assert disease_emb.shape == (6, 16)
    assert torch.isfinite(mirna_emb).all()
    assert torch.isfinite(disease_emb).all()


def test_pair_scores_shape_and_range() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
    )

    mirna_idx = torch.tensor([0, 1, 2])
    disease_idx = torch.tensor([0, 1, 2])

    scores = model(
        mirna_idx,
        disease_idx,
        edge_index=make_edge_index(),
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_isolated_nodes_are_supported() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
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

    scores = model(
        mirna_idx,
        disease_idx,
        edge_index=edge_index,
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_score_from_precomputed_embeddings() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
    )

    mirna_embeddings, disease_embeddings = model.propagate(
        make_edge_index()
    )

    mirna_idx = torch.tensor([0, 1, 2])
    disease_idx = torch.tensor([0, 1, 2])

    scores = model.score_from_embeddings(
        mirna_idx=mirna_idx,
        disease_idx=disease_idx,
        mirna_embeddings=mirna_embeddings,
        disease_embeddings=disease_embeddings,
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)


def test_gradients_reach_embeddings() -> None:
    model = BipartiteLightGCN(
        n_mirnas=5,
        n_diseases=6,
        embed_dim=16,
        n_layers=2,
    )

    mirna_idx = torch.tensor([0, 1, 2])
    disease_idx = torch.tensor([0, 1, 2])

    scores = model(
        mirna_idx,
        disease_idx,
        edge_index=make_edge_index(),
    )

    loss = scores.mean()
    loss.backward()

    assert model.mirna_embedding.weight.grad is not None
    assert model.disease_embedding.weight.grad is not None