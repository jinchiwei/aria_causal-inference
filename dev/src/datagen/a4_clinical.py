#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a clean A4 clinical summary table from ADQS and optional MRI reads."
    )
    parser.add_argument("--adqs", type=Path, required=True, help="Path to ADQS.csv.")
    parser.add_argument("--mri-reads", type=Path, default=None, help="Optional path to imaging_MRI_reads.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV or parquet path.")
    parser.add_argument("--baseline-visit-code", type=int, default=1, help="VISITCD to treat as baseline.")
    return parser


def load_baseline(adqs_path: Path, baseline_visit_code: int) -> pd.DataFrame:
    columns = [
        "BID",
        "VISITCD",
        "TX",
        "AGEYR",
        "SEX",
        "RACE",
        "EDCCNTU",
        "ETHNIC",
        "APOEGN",
        "BMIBL",
        "AMYLCENT",
        "SUVRCER",
    ]
    adqs = pd.read_csv(adqs_path, usecols=columns, low_memory=False)
    adqs["VISITCD"] = pd.to_numeric(adqs["VISITCD"], errors="coerce")
    baseline = adqs.loc[adqs["VISITCD"] == baseline_visit_code].copy()
    if baseline.empty:
        baseline = adqs.copy()
    baseline = baseline.sort_values(["BID", "VISITCD"]).drop_duplicates("BID", keep="first")
    baseline = baseline.rename(
        columns={
            "BID": "patient_id",
            "TX": "treatment_label",
            "AGEYR": "age",
            "SEX": "sex",
            "RACE": "race",
            "EDCCNTU": "education",
            "ETHNIC": "ethnicity",
            "APOEGN": "apoe_status",
            "BMIBL": "baseline_bmi",
            "AMYLCENT": "baseline_amyloid_centiloid",
            "SUVRCER": "baseline_amyloid_suvr",
        }
    )
    baseline["apoe4_carrier"] = baseline["apoe_status"].fillna("").astype(str).str.upper().str.contains("E4").astype(int)
    baseline["diagnosis"] = "preclinical_ad"
    return baseline.reset_index(drop=True)


def summarize_mri(mri_path: Path) -> pd.DataFrame:
    columns = ["BID", "STUDYDATE_DAYS_T0", "Definite.MCH", "Lobar", "Deep", "Definite.SS"]
    mri = pd.read_csv(mri_path, usecols=columns)
    mri["STUDYDATE_DAYS_T0"] = pd.to_numeric(mri["STUDYDATE_DAYS_T0"], errors="coerce")
    for col in ("Definite.MCH", "Lobar", "Deep", "Definite.SS"):
        mri[col] = pd.to_numeric(mri[col], errors="coerce").fillna(0)

    baseline = mri.loc[mri["STUDYDATE_DAYS_T0"] < 0].copy()
    followup = mri.loc[mri["STUDYDATE_DAYS_T0"] >= 0].copy()

    baseline_summary = baseline.groupby("BID").agg(
        baseline_definite_mch=("Definite.MCH", "max"),
        baseline_definite_ss=("Definite.SS", "max"),
        baseline_lobar=("Lobar", "max"),
        baseline_deep=("Deep", "max"),
    )

    followup_summary = followup.groupby("BID").agg(
        any_followup_mch=("Definite.MCH", lambda s: int((s > 0).any())),
        any_followup_ss=("Definite.SS", lambda s: int((s > 0).any())),
        first_followup_day_t0=("STUDYDATE_DAYS_T0", "min"),
        last_followup_day_t0=("STUDYDATE_DAYS_T0", "max"),
        n_followup_mri=("STUDYDATE_DAYS_T0", "size"),
    )

    summary = baseline_summary.join(followup_summary, how="outer").reset_index()
    summary = summary.rename(columns={"BID": "patient_id"})
    summary["aria_h_any_followup"] = (
        summary["any_followup_mch"].fillna(0).astype(int) | summary["any_followup_ss"].fillna(0).astype(int)
    ).astype(int)
    summary["baseline_aria_h_positive"] = (
        (summary["baseline_definite_mch"].fillna(0) > 0) | (summary["baseline_definite_ss"].fillna(0) > 0)
    ).astype(int)
    return summary


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> int:
    args = build_parser().parse_args()

    baseline = load_baseline(args.adqs, args.baseline_visit_code)
    merged = baseline.copy()
    if args.mri_reads is not None:
        mri_summary = summarize_mri(args.mri_reads)
        merged = baseline.merge(mri_summary, on="patient_id", how="left")

    print(f"baseline rows: {len(baseline)}")
    print(f"final rows: {len(merged)}")
    print(f"apoe4 carriers: {int(merged['apoe4_carrier'].fillna(0).sum())}")
    print(f"treated labels: {merged['treatment_label'].value_counts(dropna=False).to_dict()}")

    if args.output is not None:
        save_output(merged, args.output)
        print(f"saved output: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
