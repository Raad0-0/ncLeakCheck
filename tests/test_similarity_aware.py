import pytest
import torch

from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor


@pytest.fixture
def small_graph():
    return {
        "n_mirnas": 4,
        "n_diseases": 3,
        "mirna_idx": torch.tensor([0, 1], dtype=torch.long),
        "disease_idx": torch.tensor([0, 1], dtype=torch.long),
        "K_mirna": torch.eye(4),
        "K_disease": torch.eye(3),
    }


def test_mlp_similarity_changes_output(small_graph):
    torch.manual_seed(0)
    model = DecoupledMLP(
        n_mirnas=small_graph["n_mirnas"],
        n_diseases=small_graph["n_diseases"],
        embed_dim=8,
        hidden_dim=16,
        dropout=0.0,
    ).eval()

    base = model(
        small_graph["mirna_idx"],
        small_graph["disease_idx"],
    )
    changed = model(
        small_graph["mirna_idx"],
        small_graph["disease_idx"],
        K_mirna=small_graph["K_mirna"] + 0.25,
        K_disease=small_graph["K_disease"] + 0.25,
    )

    assert base.shape == (2,)
    assert changed.shape == (2,)
    assert not torch.allclose(base, changed)


def test_gcn_similarity_changes_output(small_graph):
    torch.manual_seed(0)
    model = BipartiteGCNLinkPredictor(
        n_mirnas=small_graph["n_mirnas"],
        n_diseases=small_graph["n_diseases"],
        embed_dim=8,
        gcn_hidden=8,
        dropout=0.0,
    ).eval()

    base = model(
        small_graph["mirna_idx"],
        small_graph["disease_idx"],
        K_mirna=small_graph["K_mirna"],
        K_disease=small_graph["K_disease"],
    )
    changed = model(
        small_graph["mirna_idx"],
        small_graph["disease_idx"],
        K_mirna=torch.ones(4, 4),
        K_disease=torch.ones(3, 3),
    )

    assert base.shape == (2,)
    assert changed.shape == (2,)
    assert not torch.allclose(base, changed)


def test_gcn_similarity_output_is_finite(small_graph):
    torch.manual_seed(0)
    model = BipartiteGCNLinkPredictor(
        n_mirnas=small_graph["n_mirnas"],
        n_diseases=small_graph["n_diseases"],
        embed_dim=8,
        gcn_hidden=8,
        dropout=0.0,
    ).eval()

    output = model(
        small_graph["mirna_idx"],
        small_graph["disease_idx"],
        K_mirna=small_graph["K_mirna"],
        K_disease=small_graph["K_disease"],
    )

    assert output.shape == (2,)
    assert torch.isfinite(output).all()