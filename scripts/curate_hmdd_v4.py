from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

RAW_FILE = Path("data/raw/hmdd/hmdd_v4_2023_07_original.txt")
OUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")

STRICT_CODES = {
    "genetics_knock down_promote",
    "genetics_knock down_suppress",
    "genetics_overexpression_promote",
    "genetics_overexpression_suppress",
    "genetics_GWAS",
    "circulation_biomarker_diagnosis_up",
    "circulation_biomarker_diagnosis_down",
    "circulation_biomarker_diagnosis_ns",
    "circulation_biomarker_prognosis_up",
    "circulation_biomarker_prognosis_down",
    "circulation_biomarker_prognosis_ns",
    "tissue_expression_up",
    "tissue_expression_down",
    "tissue_expression_ns",
    "epigenetics",
    "exosome",
    "therapeutic target",
}

BROAD_ONLY_CODES = {"other"}
EXCLUDED_CODES = {
    "virus_miRNA",
    "target gene",
    "transcription factor target",
    "lncRNA target",
    "circRNA target",
}
EXPECTED_FIELDS = ["code", "PMID", "miRNA", "disease", "description"]


def clean_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def normalize_mirna(value: str) -> str:
    return clean_text(value).lower()


def normalize_disease(value: str) -> str:
    return clean_text(value).casefold()


def serialize(values: set[str]) -> str:
    return "|".join(sorted(value for value in values if value))


def aggregate_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}

    for row in rows:
        mirna_id = normalize_mirna(row["miRNA"])
        disease_id = normalize_disease(row["disease"])
        key = (mirna_id, disease_id)

        if key not in grouped:
            grouped[key] = {
                "normalized_mirna_id": mirna_id,
                "normalized_disease_label": disease_id,
                "miRNA": clean_text(row["miRNA"]),
                "disease": clean_text(row["disease"]),
                "codes": set(),
                "PMIDs": set(),
                "descriptions": set(),
                "source_row_count": 0,
            }

        entry = grouped[key]
        entry["codes"].add(clean_text(row["code"]))  # type: ignore[index]
        entry["PMIDs"].add(clean_text(row["PMID"]))  # type: ignore[index]
        entry["descriptions"].add(clean_text(row["description"]))  # type: ignore[index]
        entry["source_row_count"] += 1  # type: ignore[operator]

    output: list[dict[str, str]] = []
    for entry in grouped.values():
        output.append(
            {
                "normalized_mirna_id": str(entry["normalized_mirna_id"]),
                "normalized_disease_label": str(entry["normalized_disease_label"]),
                "miRNA": str(entry["miRNA"]),
                "disease": str(entry["disease"]),
                "codes": serialize(entry["codes"]),  # type: ignore[arg-type]
                "PMIDs": serialize(entry["PMIDs"]),  # type: ignore[arg-type]
                "descriptions": serialize(entry["descriptions"]),  # type: ignore[arg-type]
                "source_row_count": str(entry["source_row_count"]),
            }
        )

    return sorted(output, key=lambda x: (x["normalized_mirna_id"], x["normalized_disease_label"]))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "normalized_mirna_id",
        "normalized_disease_label",
        "miRNA",
        "disease",
        "codes",
        "PMIDs",
        "descriptions",
        "source_row_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw source file: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    strict_rows: list[dict[str, str]] = []
    broad_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    audit: defaultdict[str, int] = defaultdict(int)

    with RAW_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected header: {reader.fieldnames}")

        for raw in reader:
            row = {field: clean_text(raw.get(field)) for field in EXPECTED_FIELDS}
            audit["raw_rows"] += 1

            if not row["miRNA"] or not row["disease"]:
                audit["excluded_missing_mirna_or_disease"] += 1
                excluded_rows.append({**row, "exclusion_reason": "missing_mirna_or_disease"})
                continue

            if not normalize_mirna(row["miRNA"]).startswith("hsa-"):
                audit["excluded_nonhuman_or_unresolved_mirna"] += 1
                excluded_rows.append({**row, "exclusion_reason": "nonhuman_or_unresolved_mirna"})
                continue

            code = row["code"]
            if code in STRICT_CODES:
                audit["strict_retained_rows"] += 1
                strict_rows.append(row)
                broad_rows.append(row)
            elif code in BROAD_ONLY_CODES:
                audit["broad_only_retained_rows"] += 1
                broad_rows.append(row)
            elif code in EXCLUDED_CODES:
                audit[f"excluded_code::{code}"] += 1
                excluded_rows.append({**row, "exclusion_reason": f"excluded_code::{code}"})
            else:
                audit[f"excluded_unknown_code::{code or '<blank>'}"] += 1
                excluded_rows.append({**row, "exclusion_reason": f"unknown_code::{code or '<blank>'}"})

    strict_pairs = aggregate_pairs(strict_rows)
    broad_pairs = aggregate_pairs(broad_rows)

    strict_path = OUT_DIR / "hmdd_v4_human_strict.tsv"
    broad_path = OUT_DIR / "hmdd_v4_human_broad.tsv"
    excluded_path = OUT_DIR / "hmdd_v4_excluded_records.tsv"
    audit_path = OUT_DIR / "hmdd_v4_curation_audit.json"
    report_path = LOG_DIR / "hmdd_v4_curation_report.md"

    write_tsv(strict_path, strict_pairs)
    write_tsv(broad_path, broad_pairs)

    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [*EXPECTED_FIELDS, "exclusion_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded_rows)

    audit_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": RAW_FILE.as_posix(),
        "policy": {
            "species_rule": "retain records whose normalized miRNA label begins with hsa-",
            "strict_include_codes": sorted(STRICT_CODES),
            "broad_only_include_codes": sorted(BROAD_ONLY_CODES),
            "excluded_codes": sorted(EXCLUDED_CODES),
            "pair_deduplication_key": ["normalized_mirna_id", "normalized_disease_label"],
            "provenance_aggregation": ["codes", "PMIDs", "descriptions", "source_row_count"],
        },
        "counts": dict(sorted(audit.items())),
        "outputs": {
            "strict_pair_count": len(strict_pairs),
            "broad_pair_count": len(broad_pairs),
            "excluded_record_count": len(excluded_rows),
            "strict_output": strict_path.as_posix(),
            "broad_output": broad_path.as_posix(),
            "excluded_output": excluded_path.as_posix(),
        },
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = [
        "# HMDD v4.0 curation report",
        "",
        "## Policy",
        "",
        "- Retain only records whose miRNA label begins with `hsa-` after normalization.",
        "- Strict graph: direct human phenotype-related categories defined in `STRICT_CODES`.",
        "- Broad sensitivity graph: strict graph plus the `other` category.",
        "- Exclude viral, non-human/unresolved miRNAs and mechanistic target categories.",
        "- Deduplicate final graph edges by normalized `(miRNA, disease)` pair, while retaining aggregated provenance.",
        "",
        "## Counts",
        "",
        f"- Raw records: {audit['raw_rows']}",
        f"- Strict retained records before pair deduplication: {audit['strict_retained_rows']}",
        f"- Broad-only retained records before pair deduplication: {audit['broad_only_retained_rows']}",
        f"- Strict unique miRNA-disease pairs: {len(strict_pairs)}",
        f"- Broad unique miRNA-disease pairs: {len(broad_pairs)}",
        f"- Excluded records: {len(excluded_rows)}",
        "",
        "## Exclusion breakdown",
        "",
    ]
    for key, value in sorted(audit.items()):
        if key.startswith("excluded_"):
            report.append(f"- `{key}`: {value}")
    report.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Strict dataset: `{strict_path.as_posix()}`",
            f"- Broad dataset: `{broad_path.as_posix()}`",
            f"- Excluded-record audit: `{excluded_path.as_posix()}`",
            f"- Machine-readable audit: `{audit_path.as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Raw records: {audit['raw_rows']}")
    print(f"Strict retained records before deduplication: {audit['strict_retained_rows']}")
    print(f"Broad-only retained records before deduplication: {audit['broad_only_retained_rows']}")
    print(f"Strict unique pairs: {len(strict_pairs)}")
    print(f"Broad unique pairs: {len(broad_pairs)}")
    print(f"Excluded records: {len(excluded_rows)}")
    print(f"Strict output: {strict_path}")
    print(f"Broad output: {broad_path}")
    print(f"Audit: {audit_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()