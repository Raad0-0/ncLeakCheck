import csv
import json
import re
from collections import Counter
from pathlib import Path

RAW_FILE = Path("data/raw/hmdd/hmdd_v4_2023_07_original.txt")
OUT_DIR = Path("data/processed")
REPORT_FILE = Path("logs/hmdd_v4_schema_profile.md")
PROFILE_FILE = OUT_DIR / "hmdd_v4_raw_profile.json"
SAMPLE_FILE = OUT_DIR / "hmdd_v4_raw_samples.tsv"

HUMAN_MIRNA_RE = re.compile(r"^hsa-[A-Za-z0-9-]+$", flags=re.IGNORECASE)
VIRAL_PREFIXES = (
    "ebv-",
    "hcmv-",
    "hiv",
    "hsv",
    "kshv-",
    "mdv",
    "mcmv-",
    "rhv",
)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def classify_mirna(value: str) -> str:
    label = normalize_text(value)
    lower = label.lower()
    if HUMAN_MIRNA_RE.match(label):
        return "human_hsa"
    if lower.startswith(VIRAL_PREFIXES):
        return "likely_viral"
    if lower.startswith("mmu-") or lower.startswith("rno-") or lower.startswith("dre-"):
        return "nonhuman_model_organism"
    if not label:
        return "missing"
    return "other_or_unresolved"


def top(counter: Counter, n: int = 20) -> list[dict[str, object]]:
    return [{"value": key, "count": value} for key, value in counter.most_common(n)]


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    code_counter: Counter[str] = Counter()
    mirna_class_counter: Counter[str] = Counter()
    mirna_counter: Counter[str] = Counter()
    disease_counter: Counter[str] = Counter()
    pmid_counter: Counter[str] = Counter()
    description_length_counter: Counter[str] = Counter()

    duplicate_triplets: Counter[tuple[str, str, str]] = Counter()
    sample_rows: list[dict[str, str]] = []
    total_rows = 0
    missing_by_column: Counter[str] = Counter()

    with RAW_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = ["code", "PMID", "miRNA", "disease", "description"]
        if reader.fieldnames != required:
            raise ValueError(f"Unexpected header: {reader.fieldnames}")

        for row in reader:
            total_rows += 1
            clean = {key: normalize_text(value or "") for key, value in row.items()}
            for key, value in clean.items():
                if not value:
                    missing_by_column[key] += 1

            code_counter[clean["code"]] += 1
            mirna_class_counter[classify_mirna(clean["miRNA"])] += 1
            mirna_counter[clean["miRNA"]] += 1
            disease_counter[clean["disease"]] += 1
            pmid_counter[clean["PMID"]] += 1
            duplicate_triplets[(clean["miRNA"], clean["disease"], clean["PMID"])] += 1

            length = len(clean["description"])
            if length == 0:
                description_length_counter["0"] += 1
            elif length <= 100:
                description_length_counter["1-100"] += 1
            elif length <= 500:
                description_length_counter["101-500"] += 1
            elif length <= 2000:
                description_length_counter["501-2000"] += 1
            else:
                description_length_counter[">2000"] += 1

            if len(sample_rows) < 50:
                sample_rows.append(clean)

    repeated_triplets = sum(1 for count in duplicate_triplets.values() if count > 1)
    extra_duplicate_rows = sum(count - 1 for count in duplicate_triplets.values() if count > 1)

    profile = {
        "source_file": RAW_FILE.as_posix(),
        "schema": ["code", "PMID", "miRNA", "disease", "description"],
        "row_count": total_rows,
        "missing_values_by_column": dict(missing_by_column),
        "code_distribution": dict(code_counter),
        "mirna_class_distribution": dict(mirna_class_counter),
        "unique_mirna_labels": len(mirna_counter),
        "unique_disease_labels": len(disease_counter),
        "unique_pmids": len(pmid_counter),
        "unique_mirna_disease_pmid_triplets": len(duplicate_triplets),
        "repeated_triplet_keys": repeated_triplets,
        "extra_rows_from_repeated_triplets": extra_duplicate_rows,
        "description_length_distribution": dict(description_length_counter),
        "top_codes": top(code_counter),
        "top_mirnas": top(mirna_counter),
        "top_diseases": top(disease_counter),
    }

    PROFILE_FILE.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with SAMPLE_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "PMID", "miRNA", "disease", "description"], delimiter="\t")
        writer.writeheader()
        writer.writerows(sample_rows)

    report = [
        "# HMDD v4.0 schema and content profile",
        "",
        f"- Raw rows: {total_rows}",
        f"- Unique raw miRNA labels: {len(mirna_counter)}",
        f"- Unique raw disease labels: {len(disease_counter)}",
        f"- Unique PMIDs: {len(pmid_counter)}",
        f"- Unique (miRNA, disease, PMID) triplets: {len(duplicate_triplets)}",
        f"- Repeated triplet keys: {repeated_triplets}",
        f"- Extra rows from repeated triplets: {extra_duplicate_rows}",
        "",
        "## Missing values",
        "",
    ]
    report.extend(f"- `{key}`: {missing_by_column.get(key, 0)}" for key in ["code", "PMID", "miRNA", "disease", "description"])
    report.extend(["", "## Code distribution", ""])
    report.extend(f"- `{key}`: {value}" for key, value in code_counter.most_common())
    report.extend(["", "## miRNA label classes", ""])
    report.extend(f"- `{key}`: {value}" for key, value in mirna_class_counter.most_common())
    report.extend(["", "## Description lengths", ""])
    report.extend(f"- `{key}`: {value}" for key, value in sorted(description_length_counter.items()))
    report.extend(["", "## Top 20 raw miRNA labels", ""])
    report.extend(f"- `{key}`: {value}" for key, value in mirna_counter.most_common(20))
    report.extend(["", "## Top 20 raw disease labels", ""])
    report.extend(f"- `{key}`: {value}" for key, value in disease_counter.most_common(20))
    REPORT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Rows: {total_rows}")
    print(f"Code distribution: {dict(code_counter)}")
    print(f"miRNA label classes: {dict(mirna_class_counter)}")
    print(f"Unique miRNA labels: {len(mirna_counter)}")
    print(f"Unique disease labels: {len(disease_counter)}")
    print(f"Unique PMIDs: {len(pmid_counter)}")
    print(f"Unique (miRNA, disease, PMID) triplets: {len(duplicate_triplets)}")
    print(f"Repeated triplet keys: {repeated_triplets}")
    print(f"Extra duplicate rows: {extra_duplicate_rows}")
    print(f"Profile: {PROFILE_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"Samples: {SAMPLE_FILE}")


if __name__ == "__main__":
    main()