"""Plot imbalance sensitivity for similarity-aware MLP and GCN."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_PATH = PROJECT_ROOT / "results/tables/similarity_aware_imbalance_regime_b_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "results/figures/figure_imbalance_sensitivity.png"
RATIOS = ["1:1", "1:2", "1:5", "1:10"]
X = list(range(len(RATIOS)))
COLORS = {"mlp": "#1f77b4", "gcn": "#d62728"}
LABELS = {"mlp": "Similarity-aware MLP", "gcn": "Similarity-aware GCN"}


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("auprc", "AUPRC", "higher is better"),
        ("auroc", "AUROC", "higher is better"),
        ("ece", "ECE", "lower is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for ax, (metric, ylabel, note) in zip(axes, panels):
        for model in ["mlp", "gcn"]:
            values = [summary[model]["ratios"][ratio][metric] for ratio in RATIOS]
            means = [entry["mean"] for entry in values]
            stds = [entry["std"] for entry in values]

            ax.errorbar(
                X,
                means,
                yerr=stds,
                marker="o",
                linewidth=2,
                capsize=4,
                color=COLORS[model],
                label=LABELS[model],
            )

        ax.set_xticks(X, RATIOS)
        ax.set_xlabel("Negative:positive ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({note})")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle("Performance and calibration under matched training and evaluation negative-sampling ratios.", fontsize=14)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()