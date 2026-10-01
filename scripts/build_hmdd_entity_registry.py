from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUTS = {
    "strict": Path("data/processed/hmdd_v4_human_strict.tsv"),
    "broad": Path("data/processed/hmdd_v4_human_broad.tsv"),
}

OUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if len(values) else 0.0


def build_registry(name: str, rows: list[dict[str, str]]) -> dict:
    mirnas = sorted({row["normalized_mirna_id"] for row in rows})
    diseases = sorted({row["normalized_disease_label"] for row in rows})

    mirna_to_idx = {mirna: idx for idx, mirna in enumerate(mirnas)}
    disease_to_idx = {disease: idx for idx, disease in enumerate(diseases)}

    mirna_degree = Counter(row["normalized_mirna_id"] for row in rows)
    disease_degree = Counter(row["normalized_disease_label"] for row in rows)

    n_mirna = len(mirnas)
    n_disease = len(diseases)
    n_edges = len(rows)
    possible_edges = n_mirna * n_disease
    sparsity = 1.0 - (n_edges / possible_edges) if possible_edges else 0.0

    mirna_degrees = np.array([mirna_degree[node] for node in mirnas], dtype=int)
    disease_degrees = np.array([disease_degree[node] for node in diseases], dtype=int)

    matrix = np.zeros((n_mirna, n_disease), dtype=np.uint8)
    for row in rows:
        i = mirna_to_idx[row["normalized_mirna_id"]]
        j = disease_to_idx[row["normalized_disease_label"]]
        matrix[i, j] = 1

    assert int(matrix.sum()) == n_edges
    assert matrix.shape == (n_mirna, n_disease)

    entity_index = {
        "dataset": f"hmdd_v4_human_{name}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "miRNA_index": mirna_to_idx,
        "disease_index": disease_to_idx,
        "matrix_shape": [n_mirna, n_disease],
        "edge_count": n_edges,
    }

    summary = {
        "dataset": f"hmdd_v4_human_{name}",
        "miRNAs": n_mirna,
        "diseases": n_disease,
        "edges": n_edges,
        "possible_edges": possible_edges,
        "density": n_edges / possible_edges,
        "sparsity": sparsity,
        "miRNA_degree": {
            "min": int(mirna_degrees.min()),
            "median": percentile(mirna_degrees, 50),
            "mean": float(mirna_degrees.mean()),
            "p90": percentile(mirna_degrees, 90),
            "max": int(mirna_degrees.max()),
        },
        "disease_degree": {
            "min": int(disease_degrees.min()),
            "median": percentile(disease_degrees, 50),
            "mean": float(disease_degrees.mean()),
            "p90": percentile(disease_degrees, 90),
            "max": int(disease_degrees.max()),
        },
        "top_miRNA_hubs": mirna_degree.most_common(20),
        "top_disease_hubs": disease_degree.most_common(20),
    }

    np.save(OUT_DIR / f"hmdd_v4_{name}_adjacency.npy", matrix)
    (OUT_DIR / f"hmdd_v4_{name}_entity_index.json").write_text(
        json.dumps(entity_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / f"hmdd_v4_{name}_graph_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing curated dataset: {path}")
        summaries[name] = build_registry(name, read_pairs(path))

    report_lines = [
        "# HMDD v4 entity registry and graph summary",
        "",
    ]

    for name, summary in summaries.items():
        report_lines.extend(
            [
                f"## {name}",
                "",
                f"- miRNAs: {summary['miRNAs']}",
                f"- diseases: {summary['diseases']}",
                f"- edges: {summary['edges']}",
                f"- matrix shape: {summary['miRNAs']} x {summary['diseases']}",
                f"- density: {summary['density']:.8f}",
                f"- sparsity: {summary['sparsity']:.8%}",
                f"- miRNA degree — median: {summary['miRNA_degree']['median']:.2f}, p90: {summary['miRNA_degree']['p90']:.2f}, max: {summary['miRNA_degree']['max']}",
                f"- disease degree — median: {summary['disease_degree']['median']:.2f}, p90: {summary['disease_degree']['p90']:.2f}, max: {summary['disease_degree']['max']}",
                "",
                "### Top five miRNA hubs",
                "",
            ]
        )
        report_lines.extend(
            f"- `{node}`: {degree}" for node, degree in summary["top_miRNA_hubs"][:5]
        )
        report_lines.extend(["", "### Top five disease hubs", ""])
        report_lines.extend(
            f"- `{node}`: {degree}" for node, degree in summary["top_disease_hubs"][:5]
        )
        report_lines.append("")

    report_path = LOG_DIR / "hmdd_v4_graph_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    for name, summary in summaries.items():
        print(f"[{name}]")
        print(
            f"miRNAs={summary['miRNAs']} | diseases={summary['diseases']} | "
            f"edges={summary['edges']}"
        )
        print(
            f"shape={summary['miRNAs']}x{summary['diseases']} | "
            f"density={summary['density']:.8f} | sparsity={summary['sparsity']:.8%}"
        )
        print(
            f"miRNA degree: median={summary['miRNA_degree']['median']:.2f}, "
            f"p90={summary['miRNA_degree']['p90']:.2f}, "
            f"max={summary['miRNA_degree']['max']}"
        )
        print(
            f"disease degree: median={summary['disease_degree']['median']:.2f}, "
            f"p90={summary['disease_degree']['p90']:.2f}, "
            f"max={summary['disease_degree']['max']}"
        )

    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()