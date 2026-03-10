from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_AD_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_AlzheimerMCIDementia_acc-mrn.xlsx")
DEFAULT_TX_PATH = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)
DEFAULT_CONTROLS_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_acc-mrn.xlsx")
DEFAULT_CONTROL_MRN_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_mrn.xlsx")
DEFAULT_SHORTLIST_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_shortlist_acc-mrn.xlsx")
DEFAULT_SHORTLIST_MRN_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_shortlist_mrn.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UCSF control pool and shortlist from search exports.")
    parser.add_argument("--ad-path", type=Path, default=DEFAULT_AD_PATH)
    parser.add_argument("--tx-path", type=Path, default=DEFAULT_TX_PATH)
    parser.add_argument("--controls-path", type=Path, default=DEFAULT_CONTROLS_PATH)
    parser.add_argument("--control-mrn-path", type=Path, default=DEFAULT_CONTROL_MRN_PATH)
    parser.add_argument("--shortlist-path", type=Path, default=DEFAULT_SHORTLIST_PATH)
    parser.add_argument("--shortlist-mrn-path", type=Path, default=DEFAULT_SHORTLIST_MRN_PATH)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expanded-k", type=int, default=43)
    return parser.parse_args()


def load_search_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path).copy()
    df["Accession Number"] = df["Accession Number"].astype(str)
    df["Patient MRN"] = pd.to_numeric(df["Patient MRN"], errors="coerce").astype("Int64")
    df["Exam Started Date"] = pd.to_datetime(df["Exam Started Date"], errors="coerce")
    df["Ordered Date"] = pd.to_datetime(df["Ordered Date"], errors="coerce")
    df["Patient Age"] = pd.to_numeric(df["Patient Age"], errors="coerce")
    return df


def summarize_patients(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.sort_values(["Patient MRN", "Exam Started Date"])
        .groupby("Patient MRN", dropna=True)
        .agg(
            first_exam=("Exam Started Date", "min"),
            last_exam=("Exam Started Date", "max"),
            n_exams=("Accession Number", "size"),
            age=("Patient Age", "median"),
            sex=(
                "Patient Sex",
                lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else "Missing",
            ),
            point_of_care=(
                "Point of Care",
                lambda s: s.dropna().astype(str).mode().iloc[0] if not s.dropna().empty else "Missing",
            ),
        )
        .reset_index()
    )
    summary["year"] = summary["first_exam"].dt.year
    summary["span_days"] = (summary["last_exam"] - summary["first_exam"]).dt.days
    return summary


def build_ranked_candidates(treated_patients: pd.DataFrame, control_patients: pd.DataFrame) -> pd.DataFrame:
    repeated_controls = control_patients.loc[
        (control_patients["n_exams"] >= 2) & (control_patients["span_days"] >= 30)
    ].copy()

    combined = pd.concat(
        [treated_patients.assign(group=1), repeated_controls.assign(group=0)],
        ignore_index=True,
    )
    x = combined[["age", "sex", "point_of_care", "year", "n_exams", "span_days"]]
    y = combined["group"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["age", "year", "n_exams", "span_days"],
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["sex", "point_of_care"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(x, y)

    repeated_controls["treated_like_score"] = model.predict_proba(
        repeated_controls[["age", "sex", "point_of_care", "year", "n_exams", "span_days"]]
    )[:, 1]
    target_prevalence = len(treated_patients) / (len(treated_patients) + len(repeated_controls))
    repeated_controls["score_distance"] = (repeated_controls["treated_like_score"] - target_prevalence).abs()
    repeated_controls = repeated_controls.sort_values(
        ["score_distance", "treated_like_score", "span_days", "n_exams"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    repeated_controls["rank"] = np.arange(1, len(repeated_controls) + 1)
    return repeated_controls


def smd_numeric(first: pd.Series, second: pd.Series) -> float:
    a = pd.to_numeric(first, errors="coerce").dropna()
    b = pd.to_numeric(second, errors="coerce").dropna()
    if a.empty or b.empty:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def smd_binary(first: pd.Series, second: pd.Series, level: str) -> float:
    a = (first.astype(str) == level).mean()
    b = (second.astype(str) == level).mean()
    pooled = np.sqrt((a * (1 - a) + b * (1 - b)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((a - b) / pooled)


def summarize_subset(
    treated_patients: pd.DataFrame,
    subset_patients: pd.DataFrame,
    exam_rows: pd.DataFrame,
    label: str,
) -> dict[str, float | int | str]:
    return {
        "subset": label,
        "n_patients": int(len(subset_patients)),
        "n_exam_rows": int(len(exam_rows)),
        "age_smd_abs": abs(smd_numeric(treated_patients["age"], subset_patients["age"])),
        "year_smd_abs": abs(smd_numeric(treated_patients["year"], subset_patients["year"])),
        "female_smd_abs": abs(smd_binary(treated_patients["sex"], subset_patients["sex"], "Female")),
        "male_smd_abs": abs(smd_binary(treated_patients["sex"], subset_patients["sex"], "Male")),
        "n_exams_smd_abs": abs(smd_numeric(treated_patients["n_exams"], subset_patients["n_exams"])),
    }


def main() -> int:
    args = parse_args()

    ad = load_search_table(args.ad_path)
    tx = load_search_table(args.tx_path)

    overlap_mrns = set(tx["Patient MRN"].dropna())
    controls = ad.loc[~ad["Patient MRN"].isin(overlap_mrns)].copy()
    controls.to_excel(args.controls_path, index=False)
    control_mrns = (
        controls[["Patient MRN"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Patient MRN")
        .reset_index(drop=True)
    )
    control_mrns.to_excel(args.control_mrn_path, index=False)

    treated_patients = summarize_patients(tx)
    control_patients = summarize_patients(controls)
    ranked_candidates = build_ranked_candidates(treated_patients, control_patients)

    top_k_patients = ranked_candidates.head(args.top_k).copy()
    expanded_patients = ranked_candidates.head(args.expanded_k).copy()

    top_k_rows = controls.loc[controls["Patient MRN"].isin(set(top_k_patients["Patient MRN"]))].copy()
    expanded_rows = controls.loc[controls["Patient MRN"].isin(set(expanded_patients["Patient MRN"]))].copy()

    subset_summary = pd.DataFrame(
        [
            summarize_subset(treated_patients, top_k_patients, top_k_rows, f"top_{args.top_k}"),
            summarize_subset(treated_patients, expanded_patients, expanded_rows, f"top_{args.expanded_k}"),
            summarize_subset(
                treated_patients,
                ranked_candidates,
                controls.loc[controls["Patient MRN"].isin(set(ranked_candidates["Patient MRN"]))].copy(),
                "all_repeated_imaging_candidates",
            ),
        ]
    )

    with pd.ExcelWriter(args.shortlist_path) as writer:
        ranked_candidates.to_excel(writer, sheet_name="patient_summary_ranked", index=False)
        top_k_patients.to_excel(writer, sheet_name=f"top_{args.top_k}_patients", index=False)
        top_k_rows.to_excel(writer, sheet_name=f"top_{args.top_k}_exam_rows", index=False)
        expanded_patients.to_excel(writer, sheet_name=f"top_{args.expanded_k}_patients", index=False)
        expanded_rows.to_excel(writer, sheet_name=f"top_{args.expanded_k}_exam_rows", index=False)
        subset_summary.to_excel(writer, sheet_name="subset_balance_summary", index=False)

    with pd.ExcelWriter(args.shortlist_mrn_path) as writer:
        ranked_candidates[["Patient MRN", "rank"]].to_excel(writer, sheet_name="ranked_mrn", index=False)
        top_k_patients[["Patient MRN", "rank"]].to_excel(writer, sheet_name=f"top_{args.top_k}_mrn", index=False)
        expanded_patients[["Patient MRN", "rank"]].to_excel(
            writer,
            sheet_name=f"top_{args.expanded_k}_mrn",
            index=False,
        )

    print(f"wrote controls file: {args.controls_path}")
    print(f"wrote control mrn file: {args.control_mrn_path}")
    print(f"wrote shortlist workbook: {args.shortlist_path}")
    print(f"wrote shortlist mrn workbook: {args.shortlist_mrn_path}")
    print(subset_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
