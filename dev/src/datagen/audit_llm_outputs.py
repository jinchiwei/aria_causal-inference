#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_CSV_DIR = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled-llm")
DEFAULT_OUT_CSV = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled-llm_audit.csv")
DEFAULT_OUT_JSON = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/labeled-llm_audit_summary.json")

VARIABLES = [
    "aria-e",
    "aria-h",
    "edema",
    "effusion",
    "microhemorrhage",
    "superficial siderosis",
]

VAR_ALIASES = {
    "aria-e": ["aria_e", "aria-e", "ariae"],
    "aria-h": ["aria_h", "aria-h", "ariah"],
    "edema": ["edema"],
    "effusion": ["effusion"],
    "microhemorrhage": ["microhemorrhage", "micro_hemorrhage", "micro-hemorrhage"],
    "superficial siderosis": ["superficial_siderosis", "superficial siderosis", "superficialsiderosis"],
}


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _is_blank_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        stripped = s.astype(str).str.strip()
        return s.isna() | (stripped == "") | (stripped.str.lower().isin({"nan", "none", "null"}))
    return s.isna()


def _find_pred_col(df: pd.DataFrame, variable: str) -> Optional[str]:
    aliases = VAR_ALIASES.get(variable, [variable])
    candidates: list[str] = []
    for c in df.columns:
        cn = _norm(c)
        if cn in {"accession", "accessionnumber"}:
            continue
        if any(_norm(a) in cn for a in aliases):
            candidates.append(str(c))
    if not candidates:
        return None
    # Prefer exact-ish matches
    candidates.sort(key=lambda x: (len(_norm(x)), x))
    return candidates[0]


@dataclass
class VarAudit:
    variable: str
    column: Optional[str]
    n_blank: int
    frac_blank: float
    unique_values: list[str]


def audit_one_file(path: Path) -> dict:
    df = pd.read_csv(path)

    # Normalize accession column if present; not required for blank-check
    if "accession number" in df.columns:
        df = df.rename(columns={"accession number": "accession"})
    elif "Accession Number" in df.columns:
        df = df.rename(columns={"Accession Number": "accession"})

    n_rows = len(df)
    per_var: list[VarAudit] = []

    for var in VARIABLES:
        col = _find_pred_col(df, var)
        if col is None:
            per_var.append(
                VarAudit(variable=var, column=None, n_blank=n_rows, frac_blank=1.0, unique_values=[])
            )
            continue

        s = df[col]
        blank_mask = _is_blank_series(s)
        n_blank = int(blank_mask.sum())
        frac_blank = float(n_blank / n_rows) if n_rows else float("nan")

        # Sample uniques (excluding blanks) for debugging
        nonblank = s.loc[~blank_mask]
        if nonblank.empty:
            uniq = []
        else:
            uniq_vals = nonblank.dropna().unique()
            uniq = [str(x) for x in uniq_vals[:20]]

        per_var.append(
            VarAudit(
                variable=var,
                column=col,
                n_blank=n_blank,
                frac_blank=frac_blank,
                unique_values=uniq,
            )
        )

    return {
        "file": str(path),
        "rows": n_rows,
        "vars": [va.__dict__ for va in per_var],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit labeled-llm CSVs for missing/blank prediction cells."
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSVs found in {args.csv_dir}")

    audits = [audit_one_file(p) for p in csv_files]

    # Flatten to a table for easy sorting/filtering.
    rows = []
    for a in audits:
        for v in a["vars"]:
            rows.append(
                {
                    "file": a["file"],
                    "rows": a["rows"],
                    "variable": v["variable"],
                    "column": v["column"],
                    "n_blank": v["n_blank"],
                    "frac_blank": v["frac_blank"],
                    "unique_values_sample": json.dumps(v["unique_values"]),
                }
            )

    df_report = pd.DataFrame(rows)
    df_report = df_report.sort_values(
        by=["frac_blank", "n_blank", "file", "variable"], ascending=[False, False, True, True]
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(args.out_csv, index=False)

    # Summary: how many files have any blanks per variable
    summary = {}
    for var in VARIABLES:
        subset = df_report[df_report["variable"] == var]
        summary[var] = {
            "files_total": int(subset["file"].nunique()),
            "files_with_any_blank": int((subset.groupby("file")["n_blank"].max() > 0).sum()),
            "median_frac_blank": float(np.nanmedian(subset["frac_blank"].values)),
            "max_frac_blank": float(np.nanmax(subset["frac_blank"].values)),
        }

    with open(args.out_json, "w") as f:
        json.dump({"csv_dir": str(args.csv_dir), "summary": summary}, f, indent=2)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

