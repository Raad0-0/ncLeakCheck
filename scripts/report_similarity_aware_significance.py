import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [892, 7739, 6545, 4388, 4330]

FILES = {
    "MLP": ROOT / "results/benchmarks/mlp_similarity_aware_abc_ratio2_raw.json",
    "GCN": ROOT / "results/benchmarks/gcn_similarity_aware_abc_ratio2_raw.json",
}


def load_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["regime"], int(row["seed"])): row
        for row in payload["results"]
    }


def compare(rows, metric, left, right):
    left_values = np.array(
        [rows[(left, seed)][metric] for seed in SEEDS],
        dtype=float,
    )
    right_values = np.array(
        [rows[(right, seed)][metric] for seed in SEEDS],
        dtype=float,
    )

    differences = left_values - right_values
    test = wilcoxon(
        left_values,
        right_values,
        alternative="two-sided",
        method="auto",
    )

    return {
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "absolute_difference": float(differences.mean()),
        "difference_std": float(differences.std(ddof=1)),
        "wilcoxon_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def main():
    output = {}

    for model_name, path in FILES.items():
        rows = load_rows(path)
        output[model_name] = {
            "A_vs_B": {
                metric: compare(rows, metric, "A", "B")
                for metric in ["auprc", "auroc"]
            },
            "B_vs_C": {
                metric: compare(rows, metric, "B", "C")
                for metric in ["auprc", "auroc"]
            },
        }

    output_path = ROOT / "results/tables/similarity_aware_significance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for model_name, comparisons in output.items():
        print(f"\n{model_name}")
        for comparison_name, metrics in comparisons.items():
            for metric, result in metrics.items():
                print(
                    f"{comparison_name} {metric}: "
                    f"mean_difference={result['absolute_difference']:.6f}, "
                    f"p={result['p_value']:.6f}"
                )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()