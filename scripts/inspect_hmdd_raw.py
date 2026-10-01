import csv
import hashlib
import json
from datetime import date
from pathlib import Path

RAW_DIR = Path("data/raw/hmdd")
TXT_FILE = RAW_DIR / "hmdd_v4_2023_07_original.txt"
MANIFEST_FILE = RAW_DIR / "hmdd_download_manifest.json"
REPORT_FILE = Path("logs/hmdd_v4_raw_inspection.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_preview(path: Path, encoding: str) -> list[str]:
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        return [handle.readline().rstrip("\r\n") for _ in range(12)]


def detect_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                handle.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode raw file with configured encodings.")


def detect_delimiter(lines: list[str]) -> str:
    sample = "\n".join(line for line in lines if line.strip())
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        return "\t" if "\t" in lines[0] else ","


def main() -> None:
    if not TXT_FILE.exists():
        raise FileNotFoundError(f"Missing expected raw file: {TXT_FILE}")

    encoding = detect_encoding(TXT_FILE)
    preview_lines = read_preview(TXT_FILE, encoding)
    delimiter = detect_delimiter(preview_lines)

    with TXT_FILE.open("r", encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("Raw file is empty.")

    header = rows[0]
    data_rows = rows[1:]
    widths = sorted({len(row) for row in rows})
    samples = data_rows[:5]

    raw_bytes = TXT_FILE.stat().st_size
    file_hash = sha256(TXT_FILE)

    report_lines = [
        "# HMDD v4.0 raw-file inspection",
        "",
        f"- Inspection date: {date.today().isoformat()}",
        f"- File: `{TXT_FILE.as_posix()}`",
        f"- File size (bytes): {raw_bytes}",
        f"- SHA-256: `{file_hash}`",
        f"- Detected encoding: `{encoding}`",
        f"- Detected delimiter: `{repr(delimiter)}`",
        f"- Total physical rows including header: {len(rows)}",
        f"- Data rows excluding header: {len(data_rows)}",
        f"- Distinct row widths: {widths}",
        "",
        "## Header",
        "",
        "```text",
        repr(header),
        "```",
        "",
        "## First five raw data rows",
        "",
        "```text",
    ]
    report_lines.extend(repr(row) for row in samples)
    report_lines.extend(["```", "", "## First twelve physical lines", "", "```text"])
    report_lines.extend(preview_lines)
    report_lines.append("```")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {}
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    manifest.setdefault("raw_files", [])
    manifest["inspection_date"] = date.today().isoformat()
    manifest["raw_files"] = [
        {
            "filename": TXT_FILE.name,
            "relative_path": TXT_FILE.as_posix(),
            "file_size_bytes": raw_bytes,
            "sha256": file_hash,
            "detected_encoding": encoding,
            "detected_delimiter": delimiter,
            "physical_row_count_including_header": len(rows),
            "data_row_count_excluding_header": len(data_rows),
            "header": header,
        }
    ]
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Raw file: {TXT_FILE}")
    print(f"Encoding: {encoding}")
    print(f"Delimiter: {repr(delimiter)}")
    print(f"Header: {header}")
    print(f"Data rows: {len(data_rows)}")
    print(f"Distinct row widths: {widths}")
    print(f"SHA-256: {file_hash}")
    print(f"Report written: {REPORT_FILE}")
    print(f"Manifest updated: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()