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


DEFAULT_CONTROLS_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_acc-mrn.xlsx")
DEFAULT_TX_PATH = Path(
    "/data/rauschecker2/jkw/aria/data/ucsf_aria/search-pruned_aria _ lecanemab _ donanemab _ solanezumab _ aducanumab _ gantenerumab _ remternetug.xlsx"
)
DEFAULT_REVIEWED_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_shortlist_mrn_apoe4.xlsx")
DEFAULT_OUTPUT_PATH = Path("/data/rauschecker2/jkw/aria/data/ucsf_aria/search_controls_shortlist_expanded.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the UCSF control shortlist after APOE review.")
    parser.add_argument("--controls-path", type=Path, default=DEFAULT_CONTROLS_PATH)
    parser.add_argument("--tx-path", type=Path, default=DEFAULT_TX_PATH)
    parser.add_argument("--reviewed-path", type=Path, default=DEFAULT_REVIEWED_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--next-k", type=int, default=20)
    parser.add_argument("--next-expanded-k", type=int, default=40)
    return parser.parse_args()


def format_mrn(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):08d}"


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
    summary["patient_mrn_text"] = summary["Patient MRN"].map(format_mrn)
    return summary


def build_ranked_candidates(treated_patients: pd.DataFrame, control_patients: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [treated_patients.assign(group=1), control_patients.assign(group=0)],
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

    ranked = control_patients.copy()
    ranked["treated_like_score"] = model.predict_proba(
        ranked[["age", "sex", "point_of_care", "year", "n_exams", "span_days"]]
    )[:, 1]
    target_prevalence = len(treated_patients) / (len(treated_patients) + len(control_patients))
    ranked["score_distance"] = (ranked["treated_like_score"] - target_prevalence).abs()
    ranked = ranked.sort_values(
        ["score_distance", "treated_like_score", "span_days", "n_exams"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    ranked["rank_relaxed_n2"] = np.arange(1, len(ranked) + 1)
    return ranked


def load_reviewed_apoe(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet_name = "ranked_mrn"
    if "relaxed_n2_ranked" in xl.sheet_names:
        sheet_name = "relaxed_n2_ranked"
    reviewed = pd.read_excel(path, sheet_name=sheet_name).copy()
    reviewed["Patient MRN"] = pd.to_numeric(reviewed["Patient MRN"], errors="coerce").astype("Int64")
    reviewed["apoe4"] = reviewed["apoe4"].astype(str).replace({"nan": np.nan})
    reviewed["Note"] = reviewed["Note"].astype(str).replace({"nan": ""})
    reviewed["reviewed"] = reviewed["apoe4"].notna() | reviewed["Note"].astype(str).str.strip().ne("")
    reviewed["known_apoe"] = reviewed["apoe4"].notna() & (reviewed["apoe4"] != "-1")
    reviewed["missing_apoe"] = reviewed["reviewed"] & ~reviewed["known_apoe"]
    reviewed["patient_mrn_text"] = reviewed["Patient MRN"].map(format_mrn)
    return reviewed[["Patient MRN", "patient_mrn_text", "apoe4", "Note", "known_apoe", "missing_apoe", "reviewed"]]


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


def summarize_subset(treated: pd.DataFrame, subset: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    return {
        "subset": label,
        "n_patients": int(len(subset)),
        "age_smd_abs": abs(smd_numeric(treated["age"], subset["age"])),
        "year_smd_abs": abs(smd_numeric(treated["year"], subset["year"])),
        "female_smd_abs": abs(smd_binary(treated["sex"], subset["sex"], "Female")),
        "male_smd_abs": abs(smd_binary(treated["sex"], subset["sex"], "Male")),
        "n_exams_smd_abs": abs(smd_numeric(treated["n_exams"], subset["n_exams"])),
        "span_days_smd_abs": abs(smd_numeric(treated["span_days"], subset["span_days"])),
    }


def get_exam_rows(controls: pd.DataFrame, patient_subset: pd.DataFrame) -> pd.DataFrame:
    rows = controls.loc[controls["Patient MRN"].isin(set(patient_subset["Patient MRN"]))].copy()
    rows["Patient MRN Text"] = rows["Patient MRN"].map(format_mrn)
    mrn_text = rows.pop("Patient MRN Text")
    insert_at = rows.columns.get_loc("Patient MRN") + 1
    rows.insert(insert_at, "Patient MRN Text", mrn_text)
    return rows


def main() -> int:
    args = parse_args()

    controls = load_search_table(args.controls_path)
    treated = load_search_table(args.tx_path)
    reviewed = load_reviewed_apoe(args.reviewed_path)

    treated_patients = summarize_patients(treated)
    control_patients = summarize_patients(controls)

    relaxed_controls = control_patients.loc[control_patients["n_exams"] >= 2].copy()
    ranked = build_ranked_candidates(treated_patients, relaxed_controls)
    ranked = ranked.merge(reviewed, on="Patient MRN", how="left")
    if "patient_mrn_text_x" in ranked.columns or "patient_mrn_text_y" in ranked.columns:
        ranked["patient_mrn_text"] = ranked.get("patient_mrn_text_x").fillna(ranked.get("patient_mrn_text_y"))
        ranked = ranked.drop(columns=[c for c in ["patient_mrn_text_x", "patient_mrn_text_y"] if c in ranked.columns])
    ranked["reviewed"] = ranked["reviewed"].fillna(False).astype(bool)
    ranked["known_apoe"] = ranked["known_apoe"].fillna(False).astype(bool)
    ranked["missing_apoe"] = ranked["missing_apoe"].fillna(False).astype(bool)
    ranked["review_status"] = np.select(
        [ranked["known_apoe"], ranked["missing_apoe"]],
        ["reviewed_known_apoe", "reviewed_missing_apoe"],
        default="unreviewed",
    )

    reviewed_known = ranked.loc[ranked["review_status"] == "reviewed_known_apoe"].copy()
    reviewed_missing = ranked.loc[ranked["review_status"] == "reviewed_missing_apoe"].copy()
    unreviewed = ranked.loc[ranked["review_status"] == "unreviewed"].copy()

    next_batch = unreviewed.head(args.next_k).copy()
    expanded_batch = unreviewed.head(args.next_expanded_k).copy()
    all_unreviewed = unreviewed.copy()
    keep_known_plus_next = pd.concat([reviewed_known, next_batch], ignore_index=True)
    keep_known_plus_expanded = pd.concat([reviewed_known, expanded_batch], ignore_index=True)
    keep_known_plus_all = pd.concat([reviewed_known, all_unreviewed], ignore_index=True)

    summary = pd.DataFrame(
        [
            {"subset": "reviewed_known_apoe", "n_patients": int(len(reviewed_known))},
            {"subset": "reviewed_missing_apoe", "n_patients": int(len(reviewed_missing))},
            {"subset": f"next_{args.next_k}_unreviewed", "n_patients": int(len(next_batch))},
            {"subset": f"next_{args.next_expanded_k}_unreviewed", "n_patients": int(len(expanded_batch))},
            {"subset": "next_all_unreviewed", "n_patients": int(len(all_unreviewed))},
            summarize_subset(treated_patients, reviewed_known, "reviewed_known_apoe_balance"),
            summarize_subset(treated_patients, next_batch, f"next_{args.next_k}_unreviewed_balance"),
            summarize_subset(
                treated_patients,
                keep_known_plus_next,
                f"reviewed_known_plus_next_{args.next_k}_balance",
            ),
            summarize_subset(
                treated_patients,
                keep_known_plus_expanded,
                f"reviewed_known_plus_next_{args.next_expanded_k}_balance",
            ),
            summarize_subset(
                treated_patients,
                keep_known_plus_all,
                "reviewed_known_plus_next_all_balance",
            ),
        ]
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_path) as writer:
        ranked.to_excel(writer, sheet_name="relaxed_n2_ranked", index=False)
        reviewed_known.to_excel(writer, sheet_name="reviewed_known_apoe", index=False)
        reviewed_missing.to_excel(writer, sheet_name="reviewed_missing_apoe", index=False)
        next_batch.to_excel(writer, sheet_name=f"next_{args.next_k}_unreviewed", index=False)
        expanded_batch.to_excel(writer, sheet_name=f"next_{args.next_expanded_k}_unreviewed", index=False)
        all_unreviewed.to_excel(writer, sheet_name="next_all_unreviewed", index=False)
        keep_known_plus_next.to_excel(writer, sheet_name=f"keep_known_plus_next_{args.next_k}", index=False)
        keep_known_plus_expanded.to_excel(
            writer,
            sheet_name=f"keep_known_plus_next_{args.next_expanded_k}",
            index=False,
        )
        keep_known_plus_all.to_excel(writer, sheet_name="keep_known_plus_next_all", index=False)
        get_exam_rows(controls, next_batch).to_excel(writer, sheet_name=f"next_{args.next_k}_exam_rows", index=False)
        get_exam_rows(controls, expanded_batch).to_excel(
            writer,
            sheet_name=f"next_{args.next_expanded_k}_exam_rows",
            index=False,
        )
        get_exam_rows(controls, all_unreviewed).to_excel(writer, sheet_name="next_all_exam_rows", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"wrote expanded shortlist workbook: {args.output_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
