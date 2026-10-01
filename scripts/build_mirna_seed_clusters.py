from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


FASTA_FILE = Path("data/raw/mirbase/mature.fa")
HMDD_FILE = Path("data/processed/hmdd_v4_human_strict.tsv")

OUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")

MAPPING_FILE = OUT_DIR / "hmdd_v4_strict_mirna_sequence_mapping.tsv"
CLUSTER_FILE = OUT_DIR / "hmdd_v4_strict_mirna_seed_clusters.json"
UNMAPPED_FILE = OUT_DIR / "hmdd_v4_strict_mirna_unmapped.tsv"
AMBIGUOUS_FILE = OUT_DIR / "hmdd_v4_strict_mirna_ambiguous.tsv"
REPORT_FILE = LOG_DIR / "hmdd_v4_strict_mirna_seed_report.md"

VALID_RNA = set("ACGUN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def normalize_mirna(value: str) -> str:
    return normalize_text(value).lower()


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_chunks: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_chunks)))

                header = line[1:]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if header is not None:
        records.append((header, "".join(sequence_chunks)))

    return records


def mature_to_precursor_name(mature_name: str) -> str:
    """
    Convert a mature-arm name to an arm-agnostic precursor-style name.

    Example:
    hsa-let-7a-5p -> hsa-let-7a
    hsa-miR-21-3p -> hsa-mir-21
    """
    name = normalize_mirna(mature_name)

    if name.endswith("-5p") or name.endswith("-3p"):
        return name[:-3]

    return name


def parse_human_mature_records(
    records: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    by_mature_name: dict[str, dict[str, str]] = {}
    by_precursor_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    invalid_records: list[dict[str, str]] = []

    for header, raw_sequence in records:
        parts = header.split()

        if not parts:
            continue

        mature_name = normalize_mirna(parts[0])

        if not mature_name.startswith("hsa-"):
            continue

        mimat = next(
            (token for token in parts if token.upper().startswith("MIMAT")),
            "",
        )

        sequence = raw_sequence.upper().replace("T", "U").replace(" ", "")
        invalid_chars = sorted(set(sequence) - VALID_RNA)

        if len(sequence) < 8:
            invalid_records.append(
                {
                    "miRNA": mature_name,
                    "MIMAT": mimat,
                    "sequence": sequence,
                    "reason": "sequence_shorter_than_8_nt",
                }
            )
            continue

        if invalid_chars:
            invalid_records.append(
                {
                    "miRNA": mature_name,
                    "MIMAT": mimat,
                    "sequence": sequence,
                    "reason": f"invalid_rna_characters:{''.join(invalid_chars)}",
                }
            )
            continue

        record = {
            "miRNA": mature_name,
            "MIMAT": mimat,
            "sequence": sequence,
            "sequence_length": str(len(sequence)),
            "seed_2_8": sequence[1:8],
            "precursor_style_name": mature_to_precursor_name(mature_name),
        }

        if mature_name in by_mature_name:
            invalid_records.append(
                {
                    "miRNA": mature_name,
                    "MIMAT": mimat,
                    "sequence": sequence,
                    "reason": "duplicate_hsa_mature_name",
                }
            )
            continue

        by_mature_name[mature_name] = record
        by_precursor_name[record["precursor_style_name"]].append(record)

    return by_mature_name, by_precursor_name, invalid_records


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
    by_mature, by_precursor, invalid_records = parse_human_mature_records(fasta_records)
    hmdd_mirnas = read_hmdd_mirnas(HMDD_FILE)

    mapped_rows: list[dict[str, str]] = []
    unmapped_rows: list[dict[str, str]] = []
    ambiguous_rows: list[dict[str, str]] = []
    cluster_to_mirnas: dict[str, list[str]] = defaultdict(list)

    exact_count = 0
    unique_precursor_count = 0

    for hmdd_mirna in hmdd_mirnas:
        exact = by_mature.get(hmdd_mirna)

        if exact is not None:
            selected = exact
            match_type = "exact_mature_match"
            exact_count += 1
        else:
            candidates = by_precursor.get(hmdd_mirna, [])

            if len(candidates) == 1:
                selected = candidates[0]
                match_type = "unique_precursor_to_mature_match"
                unique_precursor_count += 1
            elif len(candidates) > 1:
                ambiguous_rows.append(
                    {
                        "hmdd_mirna": hmdd_mirna,
                        "reason": "multiple_mature_arms_or_sequences",
                        "candidate_mature_names": "|".join(
                            sorted(candidate["miRNA"] for candidate in candidates)
                        ),
                        "candidate_mimat_ids": "|".join(
                            sorted(candidate["MIMAT"] for candidate in candidates)
                        ),
                        "candidate_seeds_2_8": "|".join(
                            sorted(candidate["seed_2_8"] for candidate in candidates)
                        ),
                    }
                )
                continue
            else:
                unmapped_rows.append(
                    {
                        "hmdd_mirna": hmdd_mirna,
                        "reason": "no_mature_or_precursor_style_mirbase_match",
                    }
                )
                continue

        seed_cluster_id = f"SEED_{selected['seed_2_8']}"
        mapped_rows.append(
            {
                "hmdd_mirna": hmdd_mirna,
                "mirbase_mature_name": selected["miRNA"],
                "MIMAT": selected["MIMAT"],
                "match_type": match_type,
                "sequence": selected["sequence"],
                "sequence_length": selected["sequence_length"],
                "seed_2_8": selected["seed_2_8"],
                "seed_cluster_id": seed_cluster_id,
            }
        )
        cluster_to_mirnas[seed_cluster_id].append(hmdd_mirna)

    mapped_rows.sort(key=lambda row: row["hmdd_mirna"])
    unmapped_rows.sort(key=lambda row: row["hmdd_mirna"])
    ambiguous_rows.sort(key=lambda row: row["hmdd_mirna"])

    cluster_payload = {
        "dataset": "hmdd_v4_human_strict",
        "seed_definition": "mature miRNA nucleotides 2 through 8 inclusive",
        "python_slice": "sequence[1:8]",
        "source_fasta": FASTA_FILE.as_posix(),
        "source_fasta_sha256": sha256(FASTA_FILE),
        "matching_policy": {
            "exact_match": "HMDD mature name exactly equals a miRBase mature name",
            "precursor_style_match": "HMDD arm-agnostic name maps only when exactly one miRBase mature sequence exists",
            "ambiguous_match": "HMDD arm-agnostic name with multiple mature candidates is excluded from seed-family split unless a separate documented resolution rule is adopted",
        },
        "hmdd_mirna_total": len(hmdd_mirnas),
        "exact_mature_matches": exact_count,
        "unique_precursor_to_mature_matches": unique_precursor_count,
        "ambiguous_mirnas": len(ambiguous_rows),
        "unmapped_mirnas": len(unmapped_rows),
        "mapped_hmdd_mirna_total": len(mapped_rows),
        "cluster_count": len(cluster_to_mirnas),
        "clusters": {
            cluster_id: sorted(mirnas)
            for cluster_id, mirnas in sorted(cluster_to_mirnas.items())
        },
    }

    write_tsv(
        MAPPING_FILE,
        mapped_rows,
        [
            "hmdd_mirna",
            "mirbase_mature_name",
            "MIMAT",
            "match_type",
            "sequence",
            "sequence_length",
            "seed_2_8",
            "seed_cluster_id",
        ],
    )
    write_tsv(
        UNMAPPED_FILE,
        unmapped_rows,
        ["hmdd_mirna", "reason"],
    )
    write_tsv(
        AMBIGUOUS_FILE,
        ambiguous_rows,
        [
            "hmdd_mirna",
            "reason",
            "candidate_mature_names",
            "candidate_mimat_ids",
            "candidate_seeds_2_8",
        ],
    )

    CLUSTER_FILE.write_text(
        json.dumps(cluster_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cluster_sizes = sorted(
        (len(mirnas) for mirnas in cluster_to_mirnas.values()),
        reverse=True,
    )
    sequence_lengths = [int(row["sequence_length"]) for row in mapped_rows]
    coverage = len(mapped_rows) / len(hmdd_mirnas) if hmdd_mirnas else 0.0

    range_text = (
        f"{min(sequence_lengths)}–{max(sequence_lengths)} nt"
        if sequence_lengths
        else "unavailable because no miRNAs were mapped"
    )

    report = [
        "# HMDD strict miRNA seed-family mapping report",
        "",
        f"- miRBase FASTA records: {len(fasta_records)}",
        f"- Valid human mature miRNA records: {len(by_mature)}",
        f"- Invalid or duplicate human records: {len(invalid_records)}",
        f"- Strict-HMDD miRNAs: {len(hmdd_mirnas)}",
        f"- Exact mature matches: {exact_count}",
        f"- Unique precursor-to-mature matches: {unique_precursor_count}",
        f"- Ambiguous arm-level matches: {len(ambiguous_rows)}",
        f"- Unmapped HMDD miRNAs: {len(unmapped_rows)}",
        f"- Mapped HMDD miRNAs: {len(mapped_rows)}",
        f"- Mapping coverage: {coverage:.2%}",
        f"- Seed clusters: {len(cluster_to_mirnas)}",
        f"- Largest seed-cluster size: {cluster_sizes[0] if cluster_sizes else 0}",
        f"- Mapped sequence-length range: {range_text}",
        "",
        "## Regime C rule",
        "",
        "Only miRNAs with one unambiguous mature sequence mapping receive a 7-nt seed cluster.",
        "Ambiguous and unmapped nodes must not be silently assigned to arbitrary mature arms.",
    ]
    REPORT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"All FASTA records: {len(fasta_records)}")
    print(f"Valid human mature miRNA records: {len(by_mature)}")
    print(f"Invalid/duplicate human records: {len(invalid_records)}")
    print(f"Strict-HMDD miRNAs: {len(hmdd_mirnas)}")
    print(f"Exact mature matches: {exact_count}")
    print(f"Unique precursor-to-mature matches: {unique_precursor_count}")
    print(f"Ambiguous matches: {len(ambiguous_rows)}")
    print(f"Unmapped HMDD miRNAs: {len(unmapped_rows)}")
    print(f"Mapped HMDD miRNAs: {len(mapped_rows)}")
    print(f"Seed clusters: {len(cluster_to_mirnas)}")
    print(f"Mapping file: {MAPPING_FILE}")
    print(f"Cluster file: {CLUSTER_FILE}")
    print(f"Unmapped file: {UNMAPPED_FILE}")
    print(f"Ambiguous file: {AMBIGUOUS_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()