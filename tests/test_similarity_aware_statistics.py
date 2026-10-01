import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
MLP_PATH = ROOT / "results" / "benchmarks" / "mlp_similarity_aware_abc_ratio2_raw.json"
GCN_PATH = ROOT / "results" / "benchmarks" / "gcn_similarity_aware_abc_ratio2_raw.json"
EXPECTED_SEEDS = [892, 7739, 6545, 4388, 4330]


def _load_rows(path):
    if not path.exists():
        pytest.skip(f"Benchmark file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"]
    assert len(rows) == 15
    return {(row["regime"], int(row["seed"])): row for row in rows}


def _paired_values(rows, metric, left, right):
    left_values = np.array(
        [rows[(left, seed)][metric] for seed in EXPECTED_SEEDS],
        dtype=float,
    )
    right_values = np.array(
        [rows[(right, seed)][metric] for seed in EXPECTED_SEEDS],
        dtype=float,
    )
    return left_values, right_values


def _effect(rows, metric, left, right):
    left_values, right_values = _paired_values(rows, metric, left, right)
    differences = left_values - right_values
    return {
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "mean_difference": float(differences.mean()),
        "difference_std": float(differences.std(ddof=1)),
    }


def _wilcoxon(rows, metric, left, right):
    left_values, right_values = _paired_values(rows, metric, left, right)
    differences = left_values - right_values

    if np.allclose(differences, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}

    result = wilcoxon(
        left_values,
        right_values,
        alternative="two-sided",
        method="auto",
    )
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


@pytest.mark.parametrize("path", [MLP_PATH, GCN_PATH])
def test_benchmark_has_complete_paired_records(path):
    rows = _load_rows(path)
    assert set(regime for regime, _ in rows) == {"A", "B", "C"}
    assert set(seed for _, seed in rows) == set(EXPECTED_SEEDS)

    for metric in ["auprc", "auroc", "mcc", "macro_f1", "ece"]:
        for row in rows.values():
            assert np.isfinite(row[metric])


@pytest.mark.parametrize("path", [MLP_PATH, GCN_PATH])
def test_regime_a_outperforms_regime_b_on_mean_auprc(path):
    rows = _load_rows(path)
    effect = _effect(rows, "auprc", "A", "B")
    assert effect["mean_difference"] > 0.0


@pytest.mark.parametrize("path", [MLP_PATH, GCN_PATH])
def test_regime_b_outperforms_regime_c_on_mean_auprc(path):
    rows = _load_rows(path)
    effect = _effect(rows, "auprc", "B", "C")
    assert effect["mean_difference"] > 0.0


def test_wilcoxon_returns_valid_results():
    for path in [MLP_PATH, GCN_PATH]:
        rows = _load_rows(path)
        for metric in ["auprc", "auroc"]:
            result = _wilcoxon(rows, metric, "A", "B")
            assert np.isfinite(result["statistic"])
            assert 0.0 <= result["p_value"] <= 1.0


def test_printable_significance_summary(capsys):
    for model_name, path in [("MLP", MLP_PATH), ("GCN", GCN_PATH)]:
        rows = _load_rows(path)
        effect = _effect(rows, "auprc", "A", "B")
        test = _wilcoxon(rows, "auprc", "A", "B")
        print(
            f"{model_name} AUPRC A-B: "
            f"difference={effect['mean_difference']:.6f}, "
            f"p={test['p_value']:.6f}"
        )

    captured = capsys.readouterr().out
    assert "MLP AUPRC A-B" in captured
    assert "GCN AUPRC A-B" in captured