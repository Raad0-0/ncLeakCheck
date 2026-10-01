from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

HMDD_FILE = Path("data/processed/hmdd_v4_human_strict.tsv")
OUT_FILE = Path("data/processed/hmdd_v4_strict_mirna_name_examples.tsv")


def main() -> None:
    if not HMDD_FILE.exists():
        raise FileNotFoundError(f"Missing input: {HMDD_FILE}")

    with HMDD_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        mirnas = sorted({row["normalized_mirna_id"] for row in reader})

    categories = Counter()
    rows: list[dict[str, str]] = []

    for name in mirnas:
        if name.endswith("-5p") or name.endswith("-3p"):
            category = "arm_specific"
        elif "*" in name:
            category = "star_notation"
        elif "/" in name:
            category = "compound_or_slash_name"
        elif name.startswith("hsa-mir-") or name.startswith("hsa-let-"):
            category = "arm_agnostic_or_legacy"
        else:
            category = "other"

        categories[category] += 1
        rows.append({"hmdd_mirna": name, "name_category": category})

    with OUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hmdd_mirna", "name_category"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total strict-HMDD miRNAs: {len(mirnas)}")
    print(f"Name-category distribution: {dict(categories)}")
    print("First 80 normalized names:")
    for name in mirnas[:80]:
        print(name)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()