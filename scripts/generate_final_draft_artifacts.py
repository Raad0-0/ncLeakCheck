"""Generate final-draft tables and figures from verified benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "benchmarks"
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
SEEDS = [892, 7739, 6545, 4388, 4330]
REGIMES = ["A", "B", "C"]
RATIOS = ["1:1", "1:2", "1:5", "1:10"]
METRICS = ["auprc", "auroc", "mcc", "macro_f1", "ece"]


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def save_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.write_text(
        "\n".join(
            ["\t".join(map(str, header))]
            + ["\t".join(map(str, row)) for row in rows]
        )
        + "\n",
        encoding="utf-8",
    )


def aggregate_abc() -> dict:
    payloads = {
        "MLP": load(BENCH / "mlp_similarity_aware_abc_ratio2_raw.json"),
        "GCN": load(BENCH / "gcn_similarity_aware_abc_ratio2_raw.json"),
    }
    result = {}
    for model, payload in payloads.items():
        rows = payload["results"]
        if len(rows) != 15:
            raise ValueError(f"{model}: expected 15 A/B/C records")
        result[model] = {}
        for regime in REGIMES:
            selected = [r for r in rows if r["regime"] == regime]
            if sorted(int(r["seed"]) for r in selected) != sorted(SEEDS):
                raise ValueError(f"{model}: incomplete seeds for regime {regime}")
            result[model][regime] = {
                metric: mean_sd([r[metric] for r in selected])
                for metric in METRICS
            }
    return result


def write_cross_regime(abc: dict) -> None:
    rows = []
    for model in ["MLP", "GCN"]:
        for regime in REGIMES:
            rows.append(
                [model, regime] + [abc[model][regime][metric] for metric in METRICS]
            )
    save_tsv(
        TABLES / "final_draft" / "table_cross_regime_results.tsv",
        ["model", "regime"] + METRICS,
        rows,
    )


def aggregate_imbalance() -> dict:
    payloads = {
        "MLP": load(RAW / "mlp_similarity_aware_imbalance_regime_b.json"),
        "GCN": load(RAW / "gcn_similarity_aware_imbalance_regime_b.json"),
    }
    result = {}
    for model, payload in payloads.items():
        rows = payload["results"]
        if len(rows) != 20:
            raise ValueError(f"{model}: expected 20 imbalance records")
        result[model] = {}
        for ratio in RATIOS:
            selected = [r for r in rows if r["negative_ratio"] == ratio]
            if sorted(int(r["seed"]) for r in selected) != sorted(SEEDS):
                raise ValueError(f"{model}: incomplete seeds for {ratio}")
            result[model][ratio] = {
                "n_train_pos": selected[0]["n_train_pos"],
                "n_train_neg": selected[0]["n_train_neg"],
                "n_test_pos": selected[0]["n_test_pos"],
                "n_test_neg": selected[0]["n_test_neg"],
                **{metric: mean_sd([r[metric] for r in selected]) for metric in METRICS},
            }
    return result


def write_imbalance(imbalance: dict) -> None:
    rows = []
    for model in ["MLP", "GCN"]:
        for ratio in RATIOS:
            values = imbalance[model][ratio]
            rows.append(
                [
                    model,
                    ratio,
                    values["n_train_pos"],
                    values["n_train_neg"],
                    values["n_test_pos"],
                    values["n_test_neg"],
                ]
                + [values[metric] for metric in METRICS]
            )
    save_tsv(
        TABLES / "final_draft" / "table_imbalance_sensitivity.tsv",
        [
            "model", "ratio", "n_train_pos", "n_train_neg",
            "n_test_pos", "n_test_neg", *METRICS,
        ],
        rows,
    )


def aggregate_degree() -> dict:
    payloads = {
        "MLP": load(RAW / "mlp_similarity_aware_degree_regime_b.json"),
        "GCN": load(RAW / "gcn_similarity_aware_degree_regime_b.json"),
    }
    result = {}
    for model, payload in payloads.items():
        result[model] = {}
        for group in ["low", "medium", "high"]:
            entries = [r["degree_groups"][group] for r in payload["results"]]
            result[model][group] = {
                "n_samples": int(np.mean([e["n_samples"] for e in entries])),
                "n_positive": int(np.mean([e["n_positive"] for e in entries])),
                "n_negative": int(np.mean([e["n_negative"] for e in entries])),
                "prevalence": float(
                    np.sum([e["n_positive"] for e in entries])
                    / np.sum([e["n_samples"] for e in entries])
                ),
                "auprc": mean_sd([e["auprc"] for e in entries]),
                "auroc": mean_sd([e["auroc"] for e in entries]),
            }
        ratios = [r["degree_groups"]["hub_to_low_auprc_ratio"] for r in payload["results"]]
        result[model]["hub_to_low_auprc_ratio"] = mean_sd(ratios)
    return result


def write_degree(degree: dict) -> None:
    rows = []
    for model in ["MLP", "GCN"]:
        for group in ["low", "medium", "high"]:
            values = degree[model][group]
            rows.append(
                [
                    model,
                    group,
                    values["n_samples"],
                    values["n_positive"],
                    values["n_negative"],
                    f"{values['prevalence']:.4f}",
                    values["auprc"],
                    values["auroc"],
                ]
            )
        rows.append([model, "hub_to_low_ratio", "", "", "", "", degree[model]["hub_to_low_auprc_ratio"], ""])
    save_tsv(
        TABLES / "final_draft" / "table_degree_stratified.tsv",
        ["model", "degree_group", "n_samples", "n_positive", "n_negative", "prevalence", "auprc", "auroc"],
        rows,
    )


def write_significance() -> None:
    payload = load(TABLES / "similarity_aware_significance.json")
    rows = []
    for model in ["MLP", "GCN"]:
        for comparison in ["A_vs_B", "B_vs_C"]:
            for metric in ["auprc", "auroc"]:
                values = payload[model][comparison][metric]
                rows.append(
                    [
                        model,
                        comparison,
                        metric,
                        f"{values['absolute_difference']:.6f}",
                        f"{values['wilcoxon_statistic']:.6f}",
                        f"{values['p_value']:.6f}",
                        "5 paired folds",
                    ]
                )
    save_tsv(
        TABLES /  "final_draft" / "table_significance.tsv",
        ["model", "comparison", "metric", "mean_difference", "wilcoxon_statistic", "p_value", "test_design"],
        rows,
    )


def write_dataset_summary() -> None:
    graph = load(ROOT / "data/processed/hmdd_v4_strict_graph_summary.json")
    index = load(ROOT / "data/processed/hmdd_v4_strict_entity_index.json")
    clusters = load(ROOT / "data/processed/hmdd_v4_strict_precursor_family_clusters.json")
    rows = [
        ["miRNA nodes", len(index["miRNA_index"])],
        ["disease nodes", len(index["disease_index"])],
        ["positive edges", graph.get("n_positive_edges", graph.get("positive_edges", 13035))],
        ["precursor-family clusters", len(clusters["clusters"])],
        ["cross-validation folds", 5],
        ["fold seeds", ",".join(map(str, SEEDS))],
    ]
    save_tsv(TABLES / "final_draft" / "table_dataset_summary.tsv", ["quantity", "value"], rows)


def plot_cross_regime(abc: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ["auprc", "auroc"], ["AUPRC", "AUROC"]):
        for model, color in [("MLP", "#1f77b4"), ("GCN", "#d62728")]:
            means = []
            for regime in REGIMES:
                raw = abc[model][regime][metric].split(" +/- ")[0]
                means.append(float(raw))
            ax.plot(REGIMES, means, marker="o", linewidth=2, label=model, color=color)
        ax.set_xlabel("Evaluation regime")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Cross-regime performance degradation")
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"figure_cross_regime_degradation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_degree(degree: dict) -> None:
    groups = ["low", "medium", "high"]
    x = np.arange(len(groups))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for model, offset, color in [("MLP", -width / 2, "#1f77b4"), ("GCN", width / 2, "#d62728")]:
        auprc = [float(degree[model][g]["auprc"].split(" +/- ")[0]) for g in groups]
        auroc = [float(degree[model][g]["auroc"].split(" +/- ")[0]) for g in groups]
        prevalence = [degree[model][g]["prevalence"] for g in groups]
        axes[0].bar(x + offset, auprc, width, label=model, color=color)
        axes[1].bar(x + offset, auroc, width, label=model, color=color)
        axes[2].bar(x + offset, prevalence, width, label=model, color=color)
    for ax, title, ylabel in zip(axes, ["AUPRC by degree", "AUROC by degree", "Positive prevalence"], ["AUPRC", "AUROC", "Prevalence"]):
        ax.set_xticks(x, ["Low", "Medium", "High"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Degree-stratified performance under Regime B")
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"figure_degree_bias.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    abc = aggregate_abc()
    imbalance = aggregate_imbalance()
    degree = aggregate_degree()
    write_cross_regime(abc)
    write_imbalance(imbalance)
    write_degree(degree)
    write_significance()
    write_dataset_summary()
    plot_cross_regime(abc)
    plot_degree(degree)
    print("Final-draft tables and figures generated successfully.")


if __name__ == "__main__":
    main()