from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aria_root() -> Path:
    # aria/dev/src/datagen/<this file>
    return Path(__file__).resolve().parents[3]


DEFAULT_XLSX = (
    aria_root()
    / "data"
    / "ucsf_aria"
    / "search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)


def build_accession_to_mrn_map(df: pd.DataFrame, accession_col: str, mrn_col: str) -> dict[str, str]:
    if accession_col not in df.columns:
        raise KeyError(f"Missing column: {accession_col}")
    if mrn_col not in df.columns:
        raise KeyError(f"Missing column: {mrn_col}")

    tmp = df[[accession_col, mrn_col]].copy()
    tmp[accession_col] = tmp[accession_col].astype(str).str.strip()
    tmp[mrn_col] = tmp[mrn_col].astype(str).str.strip()

    tmp = tmp[(tmp[accession_col] != "") & (tmp[accession_col].str.lower() != "nan")]
    tmp = tmp[(tmp[mrn_col] != "") & (tmp[mrn_col].str.lower() != "nan")]

    accession_to_mrn: dict[str, str] = {}
    for accession, sub in tmp.groupby(accession_col, sort=False):
        mrns = sorted(set(sub[mrn_col].tolist()))
        if not mrns:
            continue
        if len(mrns) > 1:
            # Keep deterministic output; user can inspect conflicts in the source file.
            print(f"WARNING: accession {accession} maps to multiple MRNs: {mrns} (using {mrns[0]})")
        accession_to_mrn[str(accession)] = str(mrns[0])
    return accession_to_mrn


def add_patient_mrn_column(
    df: pd.DataFrame,
    accession_to_mrn: dict[str, str],
    accession_col: str,
    out_col: str,
) -> pd.DataFrame:
    if accession_col not in df.columns:
        raise KeyError(f"Missing column: {accession_col}")

    out = df.copy()
    accessions = out[accession_col].astype(str).str.strip()
    out[out_col] = accessions.map(accession_to_mrn)

    # Move to far left
    cols = [out_col] + [c for c in out.columns if c != out_col]
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV from the pruned ARIA anti-amyloid search XLSX, inserting a leftmost "
            "'patient mrn' column mapped from Accession Number."
        )
    )
    parser.add_argument("--map-xlsx", type=Path, default=DEFAULT_XLSX, help="XLSX that contains Accession Number and Patient MRN.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="File to augment (CSV or XLSX) that contains an Accession Number column; defaults to --map-xlsx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path; defaults to <input_stem>_with_patient_mrn.csv next to the input.",
    )
    parser.add_argument("--accession-col", default="Accession Number", help="Accession column name.")
    parser.add_argument("--mrn-col", default="Patient MRN", help="MRN column name in --map-xlsx.")
    parser.add_argument("--out-mrn-col", default="patient mrn", help="Output column name to insert on the left.")
    args = parser.parse_args()

    map_xlsx = args.map_xlsx
    input_path = args.input or map_xlsx

    if not map_xlsx.exists():
        raise FileNotFoundError(f"Mapping XLSX not found: {map_xlsx}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_map = pd.read_excel(map_xlsx)
    accession_to_mrn = build_accession_to_mrn_map(df_map, accession_col=args.accession_col, mrn_col=args.mrn_col)
    if not accession_to_mrn:
        raise ValueError("No accession→MRN mappings found (check column names and missingness).")

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df_in = pd.read_excel(input_path)
    else:
        df_in = pd.read_csv(input_path)

    df_out = add_patient_mrn_column(
        df_in,
        accession_to_mrn=accession_to_mrn,
        accession_col=args.accession_col,
        out_col=args.out_mrn_col,
    )

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix("").with_name(f"{input_path.stem}_with_patient_mrn.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

