from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

INPUT_FILE = Path("data/processed/hmdd_v4_human_strict.tsv")
OUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")

REGISTRY_TSV = OUT_DIR / "hmdd_v4_strict_disease_registry.tsv"
REGISTRY_JSON = OUT_DIR / "hmdd_v4_strict_disease_registry.json"
VARIANTS_TSV = OUT_DIR / "hmdd_v4_strict_disease_label_variants.tsv"
REPORT_FILE = LOG_DIR / "hmdd_v4_strict_disease_registry_report.md"

PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_display_label(value: str | None) -> str:
    return SPACE_RE.sub(" ", (value or "").strip())


def canonical_text_key(value: str | None) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().strip()
    text = PUNCTUATION_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing curated pair file: {INPUT_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw_display_by_normalized: defaultdict[str, Counter[str]] = defaultdict(Counter)
    edge_degree: Counter[str] = Counter()
    source_row_count: Counter[str] = Counter()
    total_pairs = 0

    with INPUT_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "normalized_disease_label",
            "disease",
            "source_row_count",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")

        for row in reader:
            total_pairs += 1
            normalized_label = normalize_display_label(row["normalized_disease_label"])
            raw_label = normalize_display_label(row["disease"])

            if not normalized_label or not raw_label:
                raise ValueError("Encountered a blank disease label in curated strict dataset.")

            raw_display_by_normalized[normalized_label][raw_label] += 1
            edge_degree[normalized_label] += 1
            source_row_count[normalized_label] += int(row["source_row_count"])

    disease_labels = sorted(raw_display_by_normalized)
    index = {label: position for position, label in enumerate(disease_labels)}
    canonical_groups: defaultdict[str, list[str]] = defaultdict(list)
    for label in disease_labels:
        canonical_groups[canonical_text_key(label)].append(label)

    registry_rows: list[dict[str, str]] = []
    variant_rows: list[dict[str, str]] = []
    collision_count = 0

    for label in disease_labels:
        raw_variants = raw_display_by_normalized[label]
        preferred_display = sorted(
            raw_variants.items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        canonical_key = canonical_text_key(label)
        collision_group = sorted(canonical_groups[canonical_key])
        has_collision = len(collision_group) > 1
        if has_collision:
            collision_count += 1

        registry_rows.append(
            {
                "disease_index": str(index[label]),
                "normalized_disease_label": label,
                "preferred_display_label": preferred_display,
                "canonical_text_key": canonical_key,
                "edge_degree": str(edge_degree[label]),
                "supporting_raw_record_count": str(source_row_count[label]),
                "raw_display_variant_count": str(len(raw_variants)),
                "canonical_key_collision": str(has_collision).lower(),
                "provisional_ontology_id": "",
                "ontology_mapping_status": "unmapped_not_attempted",
                "ontology_mapping_note": "No automatic DOID/MONDO/MeSH mapping performed in Day 06.",
            }
        )

        for raw_label, count in sorted(raw_variants.items(), key=lambda item: (-item[1], item[0])):
            variant_rows.append(
                {
                    "normalized_disease_label": label,
                    "raw_display_label": raw_label,
                    "pair_count": str(count),
                    "canonical_text_key": canonical_key,
                }
            )

    registry_fields = [
        "disease_index",
        "normalized_disease_label",
        "preferred_display_label",
        "canonical_text_key",
        "edge_degree",
        "supporting_raw_record_count",
        "raw_display_variant_count",
        "canonical_key_collision",
        "provisional_ontology_id",
        "ontology_mapping_status",
        "ontology_mapping_note",
    ]
    with REGISTRY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=registry_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(registry_rows)

    with VARIANTS_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "normalized_disease_label",
                "raw_display_label",
                "pair_count",
                "canonical_text_key",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(variant_rows)

    payload = {
        "dataset": "hmdd_v4_human_strict",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pair_file": INPUT_FILE.as_posix(),
        "source_pair_file_sha256": sha256(INPUT_FILE),
        "normalization_policy": {
            "normalized_label_source": "Existing curated normalized_disease_label retained unchanged except whitespace normalization.",
            "canonical_text_key": "Unicode NFKD normalization, diacritic removal, case folding, punctuation-to-space, whitespace collapse.",
            "ontology_mapping": "Not attempted; registry intentionally preserves unmapped labels for later curated mapping.",
        },
        "pair_count": total_pairs,
        "disease_count": len(disease_labels),
        "canonical_key_collision_labels": collision_count,
        "disease_index": index,
    }
    REGISTRY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    degree_values = [edge_degree[label] for label in disease_labels]
    top_hubs = sorted(edge_degree.items(), key=lambda item: (-item[1], item[0]))[:20]
    collision_groups = {
        key: sorted(labels)
        for key, labels in canonical_groups.items()
        if len(labels) > 1
    }

    report_lines = [
        "# HMDD v4 strict disease registry report",
        "",
        "## Summary",
        "",
        f"- Curated pair rows read: {total_pairs}",
        f"- Unique disease labels: {len(disease_labels)}",
        f"- Canonical text-key collision labels: {collision_count}",
        f"- Pair-file SHA-256: `{sha256(INPUT_FILE)}`",
        "",
        "## Ontology policy",
        "",
        "- No automatic Disease Ontology, MONDO, or MeSH mapping was performed in this step.",
        "- The registry is a deterministic text-level reference layer, not an ontology assertion.",
        "- Ontology IDs may be added later only with an auditable mapping source and confidence review.",
        "",
        "## Top 20 disease hubs by strict-graph degree",
        "",
    ]
    report_lines.extend(f"- `{label}`: {degree}" for label, degree in top_hubs)
    report_lines.extend(["", "## Canonical text-key collisions", ""])
    if collision_groups:
        for key, labels in sorted(collision_groups.items()):
            report_lines.append(f"- `{key}`: {' | '.join(labels)}")
    else:
        report_lines.append("- None")

    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Curated pair rows: {total_pairs}")
    print(f"Unique disease labels: {len(disease_labels)}")
    print(f"Canonical-key collision labels: {collision_count}")
    print(f"Top disease hub: {top_hubs[0][0]} ({top_hubs[0][1]} edges)" if top_hubs else "Top disease hub: unavailable")
    print(f"Registry TSV: {REGISTRY_TSV}")
    print(f"Registry JSON: {REGISTRY_JSON}")
    print(f"Variants TSV: {VARIANTS_TSV}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()