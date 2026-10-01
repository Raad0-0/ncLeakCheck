from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RAW_FILE = Path("data/raw/hmdd/hmdd_v4_2023_07_original.txt")
MANIFEST_FILE = Path("data/raw/hmdd/hmdd_download_manifest.json")

PAIR_FILE = Path("data/processed/hmdd_v4_human_strict.tsv")
MATRIX_FILE = Path("data/processed/hmdd_v4_strict_adjacency.npy")
INDEX_FILE = Path("data/processed/hmdd_v4_strict_entity_index.json")
SUMMARY_FILE = Path("data/processed/hmdd_v4_strict_graph_summary.json")

REPORT_FILE = Path("logs/hmdd_v4_primary_dataset_verification.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_pairs() -> list[dict[str, str]]:
    with PAIR_FILE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def is_contiguous(index: dict[str, int]) -> bool:
    return sorted(index.values()) == list(range(len(index)))


def main() -> None:
    required_paths = [
        RAW_FILE,
        MANIFEST_FILE,
        PAIR_FILE,
        MATRIX_FILE,
        INDEX_FILE,
        SUMMARY_FILE,
    ]
    for path in required_paths:
        require(path.exists(), f"Missing required file: {path}")

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    recorded_hash = manifest["raw_files"][0]["sha256"]
    observed_hash = sha256(RAW_FILE)
    require(
        observed_hash == recorded_hash,
        "Raw HMDD SHA-256 does not match the download manifest.",
    )

    rows = read_pairs()
    matrix = np.load(MATRIX_FILE)
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))

    mirna_index = index["miRNA_index"]
    disease_index = index["disease_index"]

    require(is_contiguous(mirna_index), "miRNA indices are not contiguous.")
    require(is_contiguous(disease_index), "Disease indices are not contiguous.")
    require(
        matrix.shape == (len(mirna_index), len(disease_index)),
        "Matrix shape does not match entity-index sizes.",
    )
    require(
        int(matrix.sum()) == len(rows),
        "Matrix edge sum does not equal curated TSV pair count.",
    )
    require(
        int(matrix.sum()) == summary["edges"],
        "Matrix edge sum does not equal graph summary edge count.",
    )

    pairs = set()
    for row in rows:
        mirna = row["normalized_mirna_id"]
        disease = row["normalized_disease_label"]
        pair = (mirna, disease)

        require(mirna in mirna_index, f"Unindexed miRNA: {mirna}")
        require(disease in disease_index, f"Unindexed disease: {disease}")
        require(pair not in pairs, f"Duplicate pair in curated TSV: {pair}")
        pairs.add(pair)

        i = mirna_index[mirna]
        j = disease_index[disease]
        require(matrix[i, j] == 1, f"Missing matrix edge for pair: {pair}")

    mirna_degrees = matrix.sum(axis=1)
    disease_degrees = matrix.sum(axis=0)
    require(np.all(mirna_degrees > 0), "Zero-degree miRNA exists in primary registry.")
    require(np.all(disease_degrees > 0), "Zero-degree disease exists in primary registry.")

    report = [
        "# HMDD v4 strict primary-dataset verification",
        "",
        f"- Raw SHA-256 verified: `{observed_hash}`",
        f"- Curated unique pairs: {len(rows)}",
        f"- Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}",
        f"- Matrix edge sum: {int(matrix.sum())}",
        f"- miRNA nodes: {len(mirna_index)}",
        f"- Disease nodes: {len(disease_index)}",
        f"- Minimum miRNA degree: {int(mirna_degrees.min())}",
        f"- Minimum disease degree: {int(disease_degrees.min())}",
        "",
        "All primary-data integrity assertions passed.",
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Raw SHA-256: PASS")
    print("Entity index contiguity: PASS")
    print("TSV-to-matrix edge consistency: PASS")
    print("Pair uniqueness: PASS")
    print("No zero-degree primary entities: PASS")
    print("Primary dataset verification: PASS")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()