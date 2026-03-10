from __future__ import annotations

import argparse
import csv
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


def aria_root() -> Path:
    # aria/dev/src/datagen/<this file>
    return Path(__file__).resolve().parents[3]


DEFAULT_PRUNED_XLSX = (
    aria_root()
    / "data"
    / "ucsf_aria"
    / "search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)
DEFAULT_ANNOTATIONS_XLSX = aria_root() / "data" / "ucsf_aria" / "labeled" / "combined_annotations.xlsx"
DEFAULT_OUTPUT_CSV = aria_root() / "data" / "ucsf_aria" / "ucsf-aria_mrn-acc.csv"


def _col_letters(cell_ref: str) -> str:
    return "".join([c for c in cell_ref if c.isalpha()])


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n


def _read_shared_strings(z: zipfile.ZipFile) -> list[str] | None:
    try:
        xml = z.read("xl/sharedStrings.xml")
    except KeyError:
        return None
    root = ET.fromstring(xml)
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out: list[str] = []
    for si in root.findall("s:si", ns):
        texts = [t.text or "" for t in si.findall(".//s:t", ns)]
        out.append("".join(texts))
    return out


def _cell_text(cell: ET.Element, shared: list[str] | None, ns: dict[str, str]) -> str | None:
    t = cell.attrib.get("t")
    v_el = cell.find("m:v", ns)

    if t == "s":
        if v_el is None or shared is None:
            return None
        try:
            return shared[int(v_el.text or "")]
        except Exception:
            return None

    if t == "inlineStr":
        is_el = cell.find("m:is", ns)
        if is_el is None:
            return None
        ts = [t_el.text or "" for t_el in is_el.findall(".//m:t", ns)]
        return "".join(ts)

    if v_el is not None:
        return v_el.text

    return None


def _normalize_id(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    # Excel often stores integer IDs as floats like "12345.0"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def read_xlsx_column(path: Path, col_name: str, sheet_xml: str = "xl/worksheets/sheet1.xml") -> list[str | None]:
    if not path.exists():
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path, "r") as z:
        shared = _read_shared_strings(z)
        xml = z.read(sheet_xml)

    root = ET.fromstring(xml)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    sheet_data = root.find("m:sheetData", ns)
    if sheet_data is None:
        raise ValueError(f"Missing sheetData in {path}")

    rows = sheet_data.findall("m:row", ns)
    if not rows:
        return []

    # Header row mapping: column letter -> header text
    header_row = rows[0]
    header_by_col: dict[str, str] = {}
    for cell in header_row.findall("m:c", ns):
        col = _col_letters(cell.attrib.get("r", ""))
        header_by_col[col] = _cell_text(cell, shared, ns) or ""

    target_cols = [c for c, h in header_by_col.items() if h == col_name]
    if not target_cols:
        available = [h for _, h in sorted(header_by_col.items(), key=lambda kv: _col_index(kv[0]))]
        raise KeyError(f"Column {col_name!r} not found in {path}. Available: {available}")
    if len(target_cols) > 1:
        target_cols = sorted(target_cols, key=_col_index)[:1]
    target_col = target_cols[0]

    out: list[str | None] = []
    for row in rows[1:]:
        value: str | None = None
        for cell in row.findall("m:c", ns):
            col = _col_letters(cell.attrib.get("r", ""))
            if col == target_col:
                value = _cell_text(cell, shared, ns)
                break
        out.append(_normalize_id(value))
    return out


def build_accession_to_mrn_map(pruned_xlsx: Path) -> dict[str, str]:
    accessions = read_xlsx_column(pruned_xlsx, "Accession Number")
    mrns = read_xlsx_column(pruned_xlsx, "Patient MRN")

    if len(accessions) != len(mrns):
        raise ValueError(f"Row mismatch in {pruned_xlsx}: accessions={len(accessions)} mrns={len(mrns)}")

    mapping: dict[str, str] = {}
    for acc, mrn in zip(accessions, mrns, strict=False):
        if acc is None or mrn is None:
            continue
        prev = mapping.get(acc)
        if prev is not None and prev != mrn:
            print(f"WARNING: accession {acc} maps to multiple MRNs: {prev} vs {mrn} (keeping {prev})")
            continue
        mapping[acc] = mrn
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ucsf-aria_mrn-acc.csv from combined annotations + pruned search XLSX.")
    parser.add_argument("--annotations-xlsx", type=Path, default=DEFAULT_ANNOTATIONS_XLSX)
    parser.add_argument("--pruned-xlsx", type=Path, default=DEFAULT_PRUNED_XLSX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if not args.annotations_xlsx.exists():
        raise FileNotFoundError(f"Missing annotations XLSX: {args.annotations_xlsx}")
    if not args.pruned_xlsx.exists():
        raise FileNotFoundError(f"Missing pruned search XLSX: {args.pruned_xlsx}")

    accession_list = read_xlsx_column(args.annotations_xlsx, "Accession")
    # Preserve first-seen order
    seen: set[str] = set()
    accessions: list[str] = []
    for acc in accession_list:
        if acc is None or acc in seen:
            continue
        seen.add(acc)
        accessions.append(acc)

    acc_to_mrn = build_accession_to_mrn_map(args.pruned_xlsx)

    rows: list[tuple[str | None, str]] = []
    missing = 0
    for acc in accessions:
        mrn = acc_to_mrn.get(acc)
        if mrn is None:
            missing += 1
        rows.append((mrn, acc))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient mrn", "accession"])
        for mrn, acc in rows:
            w.writerow([mrn or "", acc])

    print(f"Accessions in combined_annotations.xlsx: {len(accessions)}")
    print(f"Mapped to MRN via pruned XLSX: {len(accessions) - missing}")
    print(f"Missing MRN mappings: {missing}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()

