from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

UNMAPPED_FILE = Path("data/processed/hmdd_v4_strict_precursor_unmapped.tsv")
OUT_FILE = Path("data/processed/hmdd_v4_strict_precursor_unmapped_profile.tsv")


def classify(name: str) -> str:
    if "/" in name:
        return "compound_or_slash_label"
    if "*" in name:
        return "star_notation"
    if re.search(r"[a-z]\d+[a-z]$", name):
        return "letter_suffix_variant"
    if re.search(r"-\d+$", name):
        return "terminal_numeric_suffix"
    if name.startswith("hsa-let-"):
        return "let_family_or_legacy_alias"
    if name.startswith("hsa-mir-"):
        return "mirbase_alias_or_deprecated_name"
    return "other"


def main() -> None:
    if not UNMAPPED_FILE.exists():
        raise FileNotFoundError(f"Missing input: {UNMAPPED_FILE}")

    with UNMAPPED_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    output = []
    counter = Counter()
    for row in rows:
        name = row["hmdd_mirna"]
        category = classify(name)
        counter[category] += 1
        output.append({"hmdd_mirna": name, "reason": row["reason"], "name_category": category})

    output.sort(key=lambda row: (row["name_category"], row["hmdd_mirna"]))
    with OUT_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hmdd_mirna", "reason", "name_category"], delimiter="\t")
        writer.writeheader()
        writer.writerows(output)

    print(f"Total unmapped HMDD nodes: {len(rows)}")
    print(f"Unmapped name categories: {dict(counter)}")
    print("First 80 unmapped names:")
    for row in sorted(output, key=lambda x: x["hmdd_mirna"])[:80]:
        print(row["hmdd_mirna"])
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":   
    main()