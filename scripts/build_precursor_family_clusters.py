from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FASTA_FILE = Path("data/raw/mirbase/hairpin.fa")
HMDD_FILE = Path("data/processed/hmdd_v4_human_strict.tsv")
OUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")

MAPPING_FILE = OUT_DIR / "hmdd_v4_strict_precursor_sequence_mapping.tsv"
CLUSTER_FILE = OUT_DIR / "hmdd_v4_strict_precursor_family_clusters.json"
UNMAPPED_FILE = OUT_DIR / "hmdd_v4_strict_precursor_unmapped.tsv"
REPORT_FILE = LOG_DIR / "hmdd_v4_strict_precursor_family_report.md"

VALID_RNA = set("ACGUN")
LOCUS_SUFFIX_RE = re.compile(r"^(.*)-(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def normalize_mirna(value: str | None) -> str:
    return normalize_text(value).lower()


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        records.append((header, "".join(chunks)))

    return records


def parse_human_hairpins(
    records: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    by_name: dict[str, dict[str, str]] = {}
    invalid_records: list[dict[str, str]] = []

    for header, raw_sequence in records:
        parts = header.split()
        if not parts:
            continue

        precursor_name = normalize_mirna(parts[0])
        if not precursor_name.startswith("hsa-"):
            continue

        mi_id = next((token for token in parts if token.upper().startswith("MI")), "")
        sequence = raw_sequence.upper().replace("T", "U").replace(" ", "")
        invalid_chars = sorted(set(sequence) - VALID_RNA)

        if not sequence:
            invalid_records.append(
                {
                    "precursor_name": precursor_name,
                    "mi_id": mi_id,
                    "sequence": sequence,
                    "reason": "empty_sequence",
                }
            )
            continue

        if invalid_chars:
            invalid_records.append(
                {
                    "precursor_name": precursor_name,
                    "mi_id": mi_id,
                    "sequence": sequence,
                    "reason": f"invalid_rna_characters:{''.join(invalid_chars)}",
                }
            )
            continue

        if precursor_name in by_name:
            invalid_records.append(
                {
                    "precursor_name": precursor_name,
                    "mi_id": mi_id,
                    "sequence": sequence,
                    "reason": "duplicate_precursor_name",
                }
            )
            continue

        by_name[precursor_name] = {
            "mirbase_precursor_name": precursor_name,
            "mi_id": mi_id,
            "hairpin_sequence": sequence,
            "hairpin_length": str(len(sequence)),
        }

    return by_name, invalid_records


def family_key(name: str, known_names: set[str]) -> str:
    """Group only an explicit terminal locus suffix when its base precursor exists.

    Examples:
      hsa-mir-1-1 + existing hsa-mir-1 -> hsa-mir-1
      hsa-let-7a-2 + existing hsa-let-7a -> hsa-let-7a
      hsa-mir-103a-2 + existing hsa-mir-103a -> hsa-mir-103a
      hsa-mir-21 -> hsa-mir-21

    This does not merge similarly named but biologically distinct families.
    """
    match = LOCUS_SUFFIX_RE.match(name)
    if match:
        base = match.group(1)
        if base in known_names:
            return base
    return name


def read_hmdd_mirnas(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return sorted(
            {
                normalize_mirna(row["normalized_mirna_id"])
                for row in reader
                if normalize_mirna(row["normalized_mirna_id"])
            }
        )


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for path in (FASTA_FILE, HMDD_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fasta_records = parse_fasta(FASTA_FILE)
    hairpins, invalid_records = parse_human_hairpins(fasta_records)
    hmdd_mirnas = read_hmdd_mirnas(HMDD_FILE)

    mapped_rows: list[dict[str, str]] = []
    unmapped_rows: list[dict[str, str]] = []
    clusters: dict[str, list[str]] = defaultdict(list)

    all_hmdd_names = set(hmdd_mirnas)
    exact_mapping_count = 0

    for hmdd_mirna in hmdd_mirnas:
        record = hairpins.get(hmdd_mirna)
        if record is None:
            unmapped_rows.append(
                {
                    "hmdd_mirna": hmdd_mirna,
                    "reason": "no_exact_mirbase_hairpin_name_match",
                }
            )
            continue

        exact_mapping_count += 1
        cluster_base = family_key(hmdd_mirna, all_hmdd_names)
        cluster_id = f"PRECURSOR_FAMILY_{cluster_base}"
        clusters[cluster_id].append(hmdd_mirna)

        mapped_rows.append(
            {
                "hmdd_mirna": hmdd_mirna,
                "mirbase_precursor_name": record["mirbase_precursor_name"],
                "mirbase_mi_id": record["mi_id"],
                "hairpin_sequence": record["hairpin_sequence"],
                "hairpin_length": record["hairpin_length"],
                "precursor_family_base": cluster_base,
                "precursor_family_cluster_id": cluster_id,
            }
        )

    mapped_rows.sort(key=lambda row: row["hmdd_mirna"])
    unmapped_rows.sort(key=lambda row: row["hmdd_mirna"])
    cluster_sizes = sorted((len(nodes) for nodes in clusters.values()), reverse=True)
    lengths = [int(row["hairpin_length"]) for row in mapped_rows]
    coverage = len(mapped_rows) / len(hmdd_mirnas) if hmdd_mirnas else 0.0

    payload = {
        "dataset": "hmdd_v4_human_strict",
        "cluster_type": "precursor_family_locus_aware",
        "cluster_definition": (
            "Each mapped HMDD precursor node remains distinct in the graph. "
            "For Regime C only, a terminal numeric locus suffix is collapsed to its "
            "base name only if that base node also exists in HMDD; all such nodes are "
            "held out together. No 5p/3p mature-arm assignment is inferred."
        ),
        "source_hairpin_fasta": FASTA_FILE.as_posix(),
        "source_hairpin_fasta_sha256": sha256(FASTA_FILE),
        "hmdd_mirna_total": len(hmdd_mirnas),
        "exact_hairpin_mappings": exact_mapping_count,
        "unmapped_hmdd_mirnas": len(unmapped_rows),
        "mapping_coverage": coverage,
        "cluster_count": len(clusters),
        "clusters": {
            cluster_id: sorted(nodes)
            for cluster_id, nodes in sorted(clusters.items())
        },
    }

    write_tsv(
        MAPPING_FILE,
        mapped_rows,
        [
            "hmdd_mirna",
            "mirbase_precursor_name",
            "mirbase_mi_id",
            "hairpin_sequence",
            "hairpin_length",
            "precursor_family_base",
            "precursor_family_cluster_id",
        ],
    )
    write_tsv(UNMAPPED_FILE, unmapped_rows, ["hmdd_mirna", "reason"])
    CLUSTER_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    family_size_counts = Counter(len(nodes) for nodes in clusters.values())
    length_text = f"{min(lengths)}–{max(lengths)} nt" if lengths else "unavailable"
    report_lines = [
        "# HMDD v4 strict precursor-family clustering report",
        "",
        "## Inputs",
        "",
        f"- miRBase hairpin FASTA: `{FASTA_FILE.as_posix()}`",
        f"- Hairpin FASTA SHA-256: `{sha256(FASTA_FILE)}`",
        f"- All FASTA records: {len(fasta_records)}",
        f"- Valid human precursor hairpins: {len(hairpins)}",
        f"- Invalid/duplicate human hairpin records: {len(invalid_records)}",
        "",
        "## HMDD mapping",
        "",
        f"- Strict-HMDD miRNA nodes: {len(hmdd_mirnas)}",
        f"- Exact precursor hairpin mappings: {exact_mapping_count}",
        f"- Unmapped HMDD nodes: {len(unmapped_rows)}",
        f"- Mapping coverage: {coverage:.2%}",
        f"- Hairpin sequence-length range: {length_text}",
        "",
        "## Regime C clustering",
        "",
        f"- Precursor-family clusters: {len(clusters)}",
        f"- Largest cluster size: {cluster_sizes[0] if cluster_sizes else 0}",
        f"- Singleton clusters: {family_size_counts.get(1, 0)}",
        f"- Multi-node clusters: {sum(count for size, count in family_size_counts.items() if size > 1)}",
        "",
        "## Interpretation",
        "",
        "Nodes are not merged in the association graph.",
        "Clusters are used only to assign whole related precursor/locus groups to the same Regime C fold.",
        "No mature 5p/3p arm, mature sequence, or mature seed is inferred from an arm-agnostic HMDD label.",
        "Unmapped nodes remain valid for transductive Regimes A/B but are excluded from precursor-family Regime C unless a later versioned alias mapping is added.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"All hairpin FASTA records: {len(fasta_records)}")
    print(f"Valid human precursor hairpins: {len(hairpins)}")
    print(f"Invalid/duplicate human hairpins: {len(invalid_records)}")
    print(f"Strict-HMDD miRNA nodes: {len(hmdd_mirnas)}")
    print(f"Exact precursor mappings: {exact_mapping_count}")
    print(f"Unmapped HMDD nodes: {len(unmapped_rows)}")
    print(f"Mapping coverage: {coverage:.2%}")
    print(f"Precursor-family clusters: {len(clusters)}")
    print(f"Largest cluster size: {cluster_sizes[0] if cluster_sizes else 0}")
    print(f"Mapping file: {MAPPING_FILE}")
    print(f"Cluster file: {CLUSTER_FILE}")
    print(f"Unmapped file: {UNMAPPED_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()