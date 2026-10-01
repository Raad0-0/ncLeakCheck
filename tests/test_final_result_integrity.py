import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

FOLD_SEEDS = [892, 7739, 6545, 4388, 4330]

ABC_FILES = [
    ROOT / "results/benchmarks/mlp_similarity_aware_abc_ratio2_raw.json",
    ROOT / "results/benchmarks/gcn_similarity_aware_abc_ratio2_raw.json",
]

IMBALANCE_FILES = [
    ROOT / "results/raw/mlp_similarity_aware_imbalance_regime_b.json",
    ROOT / "results/raw/gcn_similarity_aware_imbalance_regime_b.json",
]


def load_json(path):
    assert path.exists(), f"Missing result file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_abc_result_files_are_complete():
    for path in ABC_FILES:
        payload = load_json(path)
        rows = payload["results"]

        assert len(rows) == 15
        assert payload["fold_seeds"] == FOLD_SEEDS
        assert payload["regimes"] == ["A", "B", "C"]
        assert payload["similarity_aware"] is True

        keys = {
            (row["regime"], int(row["seed"]))
            for row in rows
        }

        expected = {
            (regime, seed)
            for regime in ["A", "B", "C"]
            for seed in FOLD_SEEDS
        }

        assert keys == expected


def test_imbalance_result_files_are_complete():
    expected_ratios = ["1:1", "1:2", "1:5", "1:10"]

    for path in IMBALANCE_FILES:
        payload = load_json(path)
        rows = payload["results"]

        assert len(rows) == 20
        assert payload["fold_seeds"] == FOLD_SEEDS
        assert payload["regime"] == "B"
        assert payload["similarity_aware"] is True

        keys = {
            (row["negative_ratio"], int(row["seed"]))
            for row in rows
        }

        expected = {
            (ratio, seed)
            for ratio in expected_ratios
            for seed in FOLD_SEEDS
        }

        assert keys == expected


def test_imbalance_counts_are_correct():
    expected_negative_multipliers = {
        "1:1": 1,
        "1:2": 2,
        "1:5": 5,
        "1:10": 10,
    }

    for path in IMBALANCE_FILES:
        payload = load_json(path)

        for row in payload["results"]:
            ratio = row["negative_ratio"]
            multiplier = expected_negative_multipliers[ratio]

            assert row["n_train_neg"] == row["n_train_pos"] * multiplier
            assert row["n_test_neg"] == row["n_test_pos"] * multiplier


def test_all_final_metrics_are_finite():
    metric_names = [
        "auprc",
        "auroc",
        "mcc",
        "macro_f1",
        "ece",
    ]

    for path in ABC_FILES + IMBALANCE_FILES:
        payload = load_json(path)

        for row in payload["results"]:
            for metric in metric_names:
                assert np.isfinite(row[metric])