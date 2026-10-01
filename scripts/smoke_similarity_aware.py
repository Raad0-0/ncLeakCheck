import numpy as np
import torch
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.training_loop import run_fold
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor


A_full = np.load(
    "data/processed/hmdd_v4_strict_adjacency.npy"
).astype(np.float32)

n_mirnas, n_diseases = A_full.shape

for model_name, model_factory in [
    (
        "mlp",
        lambda: DecoupledMLP(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=64,
            hidden_dim=128,
            dropout=0.3,
        ),
    ),
    (
        "gcn",
        lambda: BipartiteGCNLinkPredictor(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=64,
            gcn_hidden=64,
            dropout=0.3,
        ),
    ),
]:
    for regime in ["A", "B"]:
        metrics = run_fold(
            model=model_factory(),
            A_full=A_full,
            test_ratio=0.2,
            neg_ratio=2.0,
            n_epochs=2,
            batch_size=256,
            lr=1e-3,
            seed=892,
            regime=regime,
            device=torch.device("cpu"),
        )

        print(model_name, regime, metrics)