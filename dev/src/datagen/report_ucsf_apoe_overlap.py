from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONTROL_APOE = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf-aria_control_mrn-apoe4.xlsx")
DEFAULT_TREATED_REPORTS = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)
DEFAULT_TREATED_APOE = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf-aria_mrn-apoe4_Nabaan_01.17.26.xlsx")
DEFAULT_OUTPUT_MD = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_apoe_overlap_report.md")
DEFAULT_OUTPUT_CSV = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/ucsf_apoe_overlap_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report APOE overlap between UCSF treated and curated control cohorts.")
    parser.add_argument("--control-apoe", type=Path, default=DEFAULT_CONTROL_APOE)
    parser.add_argument("--treated-reports", type=Path, default=DEFAULT_TREATED_REPORTS)
    parser.add_argument("--treated-apoe", type=Path, default=DEFAULT_TREATED_APOE)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def _smd_binary(p1: float, p0: float) -> float:
    denom = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float((p1 - p0) / denom)


def _load_control(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    sheet_name = "relaxed_n2_ranked" if "relaxed_n2_ranked" in workbook.sheet_names else workbook.sheet_names[0]
    controls = pd.read_excel(path, sheet_name=sheet_name).copy()
    controls["Patient MRN"] = pd.to_numeric(controls["Patient MRN"], errors="coerce").astype("Int64")
    controls["apoe_status"] = (
        controls["apoe4"]
        .astype(str)
        .replace({"nan": "Missing", "-1": "Missing", "": "Missing"})
        .fillna("Missing")
    )
    return controls


def _load_treated(reports_path: Path, apoe_path: Path) -> pd.DataFrame:
    reports = pd.read_excel(reports_path)
    reports["Patient MRN"] = pd.to_numeric(reports["Patient MRN"], errors="coerce").astype("Int64")
    treated = reports[["Patient MRN"]].dropna().drop_duplicates()
    apoe = pd.read_excel(apoe_path).rename(columns={"Pt MRN": "Patient MRN", "APOE Genotype": "apoe_status"})
    apoe["Patient MRN"] = pd.to_numeric(apoe["Patient MRN"], errors="coerce").astype("Int64")
    apoe["apoe_status"] = apoe["apoe_status"].fillna("Missing").astype(str).replace({"???": "Missing"})
    treated = treated.merge(apoe[["Patient MRN", "apoe_status"]], on="Patient MRN", how="left")
    treated["apoe_status"] = treated["apoe_status"].fillna("Missing")
    return treated


def _add_apoe_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["apoe_missing"] = out["apoe_status"].eq("Missing")
    out["apoe4_carrier"] = out["apoe_status"].str.upper().str.contains("E4", na=False).astype(int)
    out["apoe_group"] = np.where(
        out["apoe_missing"],
        "missing",
        np.where(out["apoe4_carrier"] == 1, "carrier", "noncarrier"),
    )
    return out


def _summarize(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    observed = df.loc[~df["apoe_missing"]]
    carrier = int((observed["apoe4_carrier"] == 1).sum())
    noncarrier = int((observed["apoe4_carrier"] == 0).sum())
    missing = int(df["apoe_missing"].sum())
    return {
        "group": label,
        "n": int(len(df)),
        "carrier": carrier,
        "noncarrier": noncarrier,
        "missing": missing,
        "observed_n": int(len(observed)),
        "carrier_rate_observed": float(carrier / len(observed)) if len(observed) else np.nan,
        "carrier_rate_all": float(df["apoe4_carrier"].mean()) if len(df) else np.nan,
        "missing_rate": float(df["apoe_missing"].mean()) if len(df) else np.nan,
    }


def main() -> int:
    args = parse_args()
    controls = _add_apoe_fields(_load_control(args.control_apoe))
    treated = _add_apoe_fields(_load_treated(args.treated_reports, args.treated_apoe))

    summary = pd.DataFrame([
        _summarize(treated, "treated"),
        _summarize(controls, "controls"),
    ])
    comparison = pd.DataFrame(
        [
            {
                "metric": "missing_rate",
                "treated": float(treated["apoe_missing"].mean()),
                "control": float(controls["apoe_missing"].mean()),
                "smd": _smd_binary(float(treated["apoe_missing"].mean()), float(controls["apoe_missing"].mean())),
            },
            {
                "metric": "carrier_rate_observed",
                "treated": float(treated.loc[~treated["apoe_missing"], "apoe4_carrier"].mean()),
                "control": float(controls.loc[~controls["apoe_missing"], "apoe4_carrier"].mean()),
                "smd": _smd_binary(
                    float(treated.loc[~treated["apoe_missing"], "apoe4_carrier"].mean()),
                    float(controls.loc[~controls["apoe_missing"], "apoe4_carrier"].mean()),
                ),
            },
            {
                "metric": "carrier_rate_all",
                "treated": float(treated["apoe4_carrier"].mean()),
                "control": float(controls["apoe4_carrier"].mean()),
                "smd": _smd_binary(float(treated["apoe4_carrier"].mean()), float(controls["apoe4_carrier"].mean())),
            },
        ]
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [
            summary.assign(table="summary"),
            comparison.assign(table="comparison"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(args.output_csv, index=False)

    control_counts = controls["apoe_status"].value_counts(dropna=False)
    treated_counts = treated["apoe_status"].value_counts(dropna=False)
    md = "\n".join(
        [
            "# UCSF APOE Overlap",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Comparison",
            "",
            comparison.to_markdown(index=False),
            "",
            "## Control APOE Counts",
            "",
            control_counts.to_markdown(),
            "",
            "## Treated APOE Counts",
            "",
            treated_counts.to_markdown(),
            "",
        ]
    )
    args.output_md.write_text(md)

    print(f"wrote markdown: {args.output_md}")
    print(f"wrote csv: {args.output_csv}")
    print(summary.to_string(index=False))
    print("\nComparison")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
