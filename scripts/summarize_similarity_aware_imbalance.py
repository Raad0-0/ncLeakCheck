"""Aggregate similarity-aware Regime B imbalance sweep results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RATIOS = ["1:1", "1:2", "1:5", "1:10"]
SEEDS = [892, 7739, 6545, 4388, 4330]
METRICS = ["auprc", "auroc", "mcc", "macro_f1", "ece"]


def load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    expected = len(RATIOS) * len(SEEDS)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")

    keys = {(row["negative_ratio"], int(row["seed"])) for row in rows}
    expected_keys = {(ratio, seed) for ratio in RATIOS for seed in SEEDS}
    if keys != expected_keys:
        raise ValueError(f"{path}: ratio/seed coverage is incomplete")

    return payload


def summarize(payload: dict) -> dict:
    rows = payload["results"]
    output = {
        "model": payload["model"],
        "regime": payload["regime"],
        "similarity_aware": payload["similarity_aware"],
        "ratios": {},
    }

    for ratio in RATIOS:
        ratio_rows = [
            row for row in rows
            if row["negative_ratio"] == ratio
        ]

        if len(ratio_rows) != len(SEEDS):
            raise ValueError(
                f"Expected {len(SEEDS)} rows for ratio {ratio}, "
                f"found {len(ratio_rows)}."
            )

        ratio_summary = {
            "n_train_pos": ratio_rows[0]["n_train_pos"],
            "n_train_neg": ratio_rows[0]["n_train_neg"],
            "n_test_pos": ratio_rows[0]["n_test_pos"],
            "n_test_neg": ratio_rows[0]["n_test_neg"],
        }

        for metric in METRICS:
            values = np.asarray(
                [row[metric] for row in ratio_rows],
                dtype=float,
            )

            ratio_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }

        output["ratios"][ratio] = ratio_summary

    return output

def write_tsv(summaries: dict[str, dict], path: Path) -> None:
    lines = [
        "model\tratio\tn_train_pos\tn_train_neg\tn_test_pos\tn_test_neg\tmetric\tmean\tstd"
    ]
    for model, summary in summaries.items():
        for ratio in RATIOS:
            values = summary["ratios"][ratio]
            for metric in METRICS:
                lines.append(
                    f"{model}\t{ratio}\t{values['n_train_pos']}\t"
                    f"{values['n_train_neg']}\t{values['n_test_pos']}\t"
                    f"{values['n_test_neg']}\t{metric}\t"
                    f"{values[metric]['mean']:.8f}\t{values[metric]['std']:.8f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlp",
        default="results/raw/mlp_similarity_aware_imbalance_regime_b.json",
    )
    parser.add_argument(
        "--gcn",
        default="results/raw/gcn_similarity_aware_imbalance_regime_b.json",
    )
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    summaries = {
        "mlp": summarize(load_payload(Path(args.mlp))),
        "gcn": summarize(load_payload(Path(args.gcn))),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "similarity_aware_imbalance_regime_b_summary.json"
    tsv_path = output_dir / "similarity_aware_imbalance_regime_b_summary.tsv"

    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_tsv(summaries, tsv_path)

    for model, summary in summaries.items():
        print(f"\n{model.upper()}")
        for ratio in RATIOS:
            values = summary["ratios"][ratio]
            print(
                f"{ratio}: "
                f"AUPRC={values['auprc']['mean']:.4f} +/- {values['auprc']['std']:.4f}, "
                f"AUROC={values['auroc']['mean']:.4f} +/- {values['auroc']['std']:.4f}, "
                f"MCC={values['mcc']['mean']:.4f} +/- {values['mcc']['std']:.4f}, "
                f"F1={values['macro_f1']['mean']:.4f} +/- {values['macro_f1']['std']:.4f}, "
                f"ECE={values['ece']['mean']:.4f} +/- {values['ece']['std']:.4f}"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()