"""
Run a small benchmark with MLP, GCN, and LightGCN
under Regimes A, B, and C on HMDD v4 strict.
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import json
import numpy as np
import torch

from engine.training_loop import run_fold
from models.decoupled_mlp import DecoupledMLP
from models.gcn_link_predictor import BipartiteGCNLinkPredictor
from models.lightgcn_bipartite import LightGCNBipartite


def main() -> None:
    # Paths
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    A_full = np.load(data_dir / "hmdd_v4_strict_adjacency.npy")

    n_mirnas, n_diseases = A_full.shape

    # Model configurations
    models = {
        "mlp": DecoupledMLP(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=64,
            hidden_dim=128,
        ),
        "gcn": BipartiteGCNLinkPredictor(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=64,
            gcn_hidden=128,
        ),
        "lightgcn": LightGCNBipartite(
            n_mirnas=n_mirnas,
            n_diseases=n_diseases,
            embed_dim=64,
            n_layers=3,
        ),
    }

    regimes = ["A", "B"]
    seeds = [42, 1337, 2026]
    folds = 1

    results = []

    device = torch.device("cpu")  # or "cuda" if available

    for model_name, model in models.items():
        for regime in regimes:
            for fold in range(folds):
                for seed in seeds:
                    print(
                        f"Running {model_name} | Regime {regime} | Fold {fold} | Seed {seed}"
                    )

                    # For now, we ignore fold index in run_fold; you can extend later
                    metrics = run_fold(
                        model=model,
                        A_full=A_full,
                        test_ratio=0.2,
                        neg_ratio=2.0,
                        n_epochs=30,
                        batch_size=256,
                        lr=1e-3,
                        seed=seed,
                        regime=regime,
                        device=device,
                    )

                    metrics["model"] = model_name
                    metrics["fold"] = fold

                    results.append(metrics)

    # Save raw results
    out_dir = project_root / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "mlp_gcn_lightgcn_ab_raw.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved raw results to: {out_path}")


if __name__ == "__main__":
    main()