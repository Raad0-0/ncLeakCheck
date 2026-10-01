"""Summarize final similarity-aware MLP/GCN A/B/C benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = [892, 7739, 6545, 4388, 4330]
EXPECTED_REGIMES = ["A", "B", "C"]
METRICS = ["auprc", "auroc", "mcc", "macro_f1", "ece"]


def load_results(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    if len(results) != 15:
        raise ValueError(f"{path}: expected 15 records, found {len(results)}")

    seeds = sorted({int(row["seed"]) for row in results})
    regimes = [regime for regime in EXPECTED_REGIMES if any(row["regime"] == regime for row in results)]

    if seeds != sorted(EXPECTED_SEEDS):
        raise ValueError(f"{path}: unexpected fold seeds: {seeds}")
    if regimes != EXPECTED_REGIMES:
        raise ValueError(f"{path}: unexpected regimes: {regimes}")

    for row in results:
        if row.get("evaluation_negative_ratio") != "1:2":
            raise ValueError(f"{path}: invalid evaluation ratio in {row}")
        if row.get("similarity_aware") is not True:
            raise ValueError(f"{path}: similarity_aware flag missing or false")

    return payload


def grouped_rows(payload: dict) -> dict[tuple[str, int], dict]:
    return {(row["regime"], int(row["seed"])): row for row in payload["results"]}


def mean_std(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def paired_effects(rows: dict[tuple[str, int], dict], metric: str, left: str, right: str) -> tuple[float, float, float]:
    left_values = np.asarray([rows[(left, seed)][metric] for seed in EXPECTED_SEEDS], dtype=float)
    right_values = np.asarray([rows[(right, seed)][metric] for seed in EXPECTED_SEEDS], dtype=float)
    difference = left_values - right_values
    absolute = float(difference.mean())
    relative_to_right = float(absolute / right_values.mean() * 100.0)
    return absolute, relative_to_right, float(difference.std(ddof=1))


def write_summary(payloads: dict[str, dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"models": {}, "comparisons": {}}

    for model_name, payload in payloads.items():
        rows = grouped_rows(payload)
        summary["models"][model_name] = {}
        for regime in EXPECTED_REGIMES:
            regime_rows = [rows[(regime, seed)] for seed in EXPECTED_SEEDS]
            summary["models"][model_name][regime] = {
                metric: {
                    "mean": mean_std(regime_rows, metric)[0],
                    "std": mean_std(regime_rows, metric)[1],
                }
                for metric in METRICS
            }

        summary["comparisons"][model_name] = {
            "A_vs_B": {
                metric: {
                    "absolute_difference": paired_effects(rows, metric, "A", "B")[0],
                    "relative_to_B_percent": paired_effects(rows, metric, "A", "B")[1],
                    "paired_difference_std": paired_effects(rows, metric, "A", "B")[2],
                }
                for metric in METRICS
            },
            "B_vs_C": {
                metric: {
                    "absolute_difference": paired_effects(rows, metric, "B", "C")[0],
                    "relative_to_C_percent": paired_effects(rows, metric, "B", "C")[1],
                    "paired_difference_std": paired_effects(rows, metric, "B", "C")[2],
                }
                for metric in METRICS
            },
        }

    json_path = output_dir / "similarity_aware_abc_ratio2_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tsv_path = output_dir / "similarity_aware_abc_ratio2_summary.tsv"
    lines = ["model\tregime\tmetric\tmean\tstd"]
    for model_name, regimes in summary["models"].items():
        for regime, metrics in regimes.items():
            for metric, values in metrics.items():
                lines.append(
                    f"{model_name}\t{regime}\t{metric}\t"
                    f"{values['mean']:.8f}\t{values['std']:.8f}"
                )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {tsv_path}")

    for model_name, regimes in summary["models"].items():
        print(f"\n{model_name.upper()}")
        for regime in EXPECTED_REGIMES:
            auprc = regimes[regime]["auprc"]
            auroc = regimes[regime]["auroc"]
            print(
                f"Regime {regime}: "
                f"AUPRC={auprc['mean']:.4f} +/- {auprc['std']:.4f}, "
                f"AUROC={auroc['mean']:.4f} +/- {auroc['std']:.4f}"
            )

        for comparison in ["A_vs_B", "B_vs_C"]:
            auprc = summary["comparisons"][model_name][comparison]["auprc"]
            print(
                f"{comparison}: AUPRC absolute="
                f"{auprc['absolute_difference']:.4f}, "
                f"relative={auprc.get('relative_to_B_percent', auprc.get('relative_to_C_percent')):.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlp",
        default="results/benchmarks/mlp_similarity_aware_abc_ratio2_raw.json",
    )
    parser.add_argument(
        "--gcn",
        default="results/benchmarks/gcn_similarity_aware_abc_ratio2_raw.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
    )
    args = parser.parse_args()

    payloads = {
        "mlp": load_results(Path(args.mlp)),
        "gcn": load_results(Path(args.gcn)),
    }
    write_summary(payloads, Path(args.output_dir))


if __name__ == "__main__":
    main()