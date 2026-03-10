from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from causal.utils import read_table


ARIA_CONTEXT_PATTERN = re.compile(
    r"(?:\baria\b|microhemorrhage|superficial siderosis|sulcal effusion|edema)",
    flags=re.IGNORECASE,
)

TREATMENT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:lecanemab|lecanamab|lecanumab|levanemab|donanemab|aducanumab|"
    r"gantenerumab|solanezumab|remternetug|anti[- ]?amyloid|amyloid therapy|"
    r"amyloid infusion)\b|"
    r"\baria (?:monitoring|screening|follow ?up|safety|rule out|evaluation)\b|"
    r"\bbaseline for aria monitoring\b|"
    r"\bprior to dose\b|"
    r"\bbefore .*dose\b|"
    r"\bafter .*dose\b|"
    r"\bpost dose\b|"
    r"\binfusion\b"
    r")",
    flags=re.IGNORECASE,
)

TREATMENT_AGENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lecanemab", re.compile(r"\b(?:lecanemab|lecanamab|lecanumab|levanemab)\b", flags=re.IGNORECASE)),
    ("donanemab", re.compile(r"\bdonanemab\b", flags=re.IGNORECASE)),
    ("aducanumab", re.compile(r"\baducanumab\b", flags=re.IGNORECASE)),
    ("gantenerumab", re.compile(r"\bgantenerumab\b", flags=re.IGNORECASE)),
    ("solanezumab", re.compile(r"\bsolanezumab\b", flags=re.IGNORECASE)),
    ("remternetug", re.compile(r"\bremternetug\b", flags=re.IGNORECASE)),
)


def build_ucsf_proto_cohort(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    paths = dataset_cfg["paths"]
    followup_windows_months = _resolve_followup_windows_months(dataset_cfg)
    max_window_days = max(_months_to_days(months) for months in followup_windows_months)
    require_aria_context = bool(dataset_cfg.get("require_aria_context", True))
    require_followup = bool(dataset_cfg.get("require_followup_within_window", True))

    annotations = _load_annotations(paths["annotations"])
    reports = _load_reports(paths["reports"])
    apoe = _load_apoe(paths["apoe"])
    exams = annotations.merge(reports, on="accession", how="left")
    exams = exams.merge(apoe, on="patient_id", how="left")
    exams["exam_date"] = _coalesce_datetime_columns(
        exams,
        ["exam_started_date", "ordered_date", "report_finalized_date"],
    )
    exams["age"] = pd.to_numeric(exams["age"], errors="coerce")

    if require_aria_context:
        exams = exams.loc[exams["report_text"].fillna("").str.contains(ARIA_CONTEXT_PATTERN, na=False)].copy()

    exams["treatment"] = exams["report_text"].fillna("").str.contains(TREATMENT_PATTERN, na=False).astype(int)
    exams["treatment_label"] = exams["report_text"].map(_extract_treatment_label)
    exams["diagnosis"] = exams["report_text"].map(_extract_diagnosis)

    for source_column, target_column in (
        ("ARIA-E", "aria_e_label"),
        ("ARIA-H", "aria_h_label"),
    ):
        exams[target_column] = _coerce_annotation_label(exams[source_column])

    exams["aria_any_label"] = _combine_binary_labels(exams["aria_e_label"], exams["aria_h_label"])
    exams["sex"] = exams["sex"].fillna("Missing").astype(str)
    exams["race"] = "Missing"
    exams["education"] = np.nan
    exams["apoe_status"] = exams["apoe_status"].fillna("Missing").astype(str)

    exams = exams.dropna(subset=["patient_id", "exam_date"]).copy()
    exams["patient_id"] = exams["patient_id"].astype(str).str.strip()
    exams = exams.sort_values(["patient_id", "exam_date", "accession"]).reset_index(drop=True)

    patient_rows: list[dict[str, Any]] = []
    for patient_id, patient_exams in exams.groupby("patient_id", sort=False):
        baseline = patient_exams.iloc[0]
        followup = patient_exams.loc[
            (patient_exams["exam_date"] > baseline["exam_date"])
            & (patient_exams["exam_date"] <= baseline["exam_date"] + pd.Timedelta(days=max_window_days))
        ].copy()
        if require_followup and followup.empty:
            continue

        baseline_aria_e = _binary_indicator_from_label(baseline["aria_e_label"])
        baseline_aria_h = _binary_indicator_from_label(baseline["aria_h_label"])
        baseline_aria_any = int(bool(baseline_aria_e or baseline_aria_h))
        first_followup_date = followup["exam_date"].min() if not followup.empty else pd.NaT
        outcome_payload = _make_windowed_outcomes(
            followup=followup,
            baseline_aria_e=baseline_aria_e,
            baseline_aria_h=baseline_aria_h,
            baseline_aria_any=baseline_aria_any,
            followup_windows_months=followup_windows_months,
            baseline_date=baseline["exam_date"],
        )
        patient_rows.append(
            {
                "patient_id": patient_id,
                "baseline_accession": baseline["accession"],
                "baseline_exam_date": baseline["exam_date"],
                "treatment": int(baseline["treatment"]),
                "treatment_label": baseline["treatment_label"],
                "age": baseline["age"],
                "sex": baseline["sex"],
                "race": baseline["race"],
                "education": baseline["education"],
                "apoe_status": baseline["apoe_status"],
                "diagnosis": baseline["diagnosis"],
                "baseline_exam_year": float(baseline["exam_date"].year),
                "n_exams_total": int(len(patient_exams)),
                "n_followup_exams_6mo": int(len(followup)),
                "days_to_first_followup_6mo": (
                    float((first_followup_date - baseline["exam_date"]).days)
                    if pd.notna(first_followup_date)
                    else np.nan
                ),
                "baseline_aria_e_positive": baseline_aria_e,
                "baseline_aria_h_positive": baseline_aria_h,
                "baseline_aria_any_positive": baseline_aria_any,
                "source_dataset": "ucsf_aria_proto",
                **outcome_payload,
            }
        )

    cohort = pd.DataFrame(patient_rows)
    if cohort.empty:
        raise ValueError("UCSF prototype cohort is empty after filtering.")
    return cohort.sort_values("patient_id").reset_index(drop=True)


def _load_annotations(path: str) -> pd.DataFrame:
    annotations = read_table(path).copy()
    if "Accession" not in annotations.columns and "Accession Number" in annotations.columns:
        annotations = annotations.rename(columns={"Accession Number": "Accession"})

    aria_e_candidates = [col for col in annotations.columns if col == "ARIA-E" or col.startswith("aria_e_")]
    aria_h_candidates = [col for col in annotations.columns if col == "ARIA-H" or col.startswith("aria_h_")]
    if "ARIA-E" not in annotations.columns and aria_e_candidates:
        annotations = annotations.rename(columns={aria_e_candidates[0]: "ARIA-E"})
    if "ARIA-H" not in annotations.columns and aria_h_candidates:
        annotations = annotations.rename(columns={aria_h_candidates[0]: "ARIA-H"})

    required = {"Accession", "ARIA-E", "ARIA-H"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation file is missing required columns after normalization: {missing}")

    annotations["accession"] = annotations["Accession"].astype(str).str.strip()
    return annotations


def _load_reports(path: str) -> pd.DataFrame:
    report_columns = {
        "Accession Number": "accession",
        "Report Text": "report_text",
        "Patient MRN": "patient_id",
        "Patient Age": "age",
        "Patient Sex": "sex",
        "Ordered Date": "ordered_date",
        "Exam Started Date": "exam_started_date",
        "Report Finalized Date": "report_finalized_date",
    }
    reports = read_table(path).copy()
    reports = reports.rename(columns=report_columns)
    reports = reports[list(report_columns.values())].copy()
    reports["accession"] = reports["accession"].astype(str).str.strip()
    reports["patient_id"] = pd.to_numeric(reports["patient_id"], errors="coerce").astype("Int64").astype(str)
    return reports


def _load_apoe(path: str) -> pd.DataFrame:
    apoe = read_table(path).copy()
    apoe = apoe.rename(
        columns={
            "Pt MRN": "patient_id",
            "APOE Genotype": "apoe_status",
            "Note/Reason Missing": "apoe_note",
        }
    )
    apoe["patient_id"] = pd.to_numeric(apoe["patient_id"], errors="coerce").astype("Int64").astype(str)
    apoe["apoe_status"] = apoe["apoe_status"].fillna("Missing").astype(str).replace({"???": "Missing"})
    apoe["apoe_note"] = apoe["apoe_note"].fillna("").astype(str)
    apoe = apoe.sort_values(["patient_id", "apoe_status"]).drop_duplicates("patient_id", keep="first")
    return apoe[["patient_id", "apoe_status", "apoe_note"]].copy()


def _coalesce_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    coalesced = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for column in columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        coalesced = coalesced.fillna(parsed)
    return coalesced


def _coerce_annotation_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.isin([0, 1]), np.nan)
    return numeric


def _combine_binary_labels(first: pd.Series, second: pd.Series) -> pd.Series:
    positive = (first == 1) | (second == 1)
    observed = first.notna() | second.notna()
    combined = pd.Series(np.nan, index=first.index, dtype=float)
    combined.loc[observed & ~positive] = 0.0
    combined.loc[positive] = 1.0
    return combined


def _binary_indicator_from_label(value: Any) -> int:
    return int(pd.notna(value) and float(value) == 1.0)


def _reduce_followup_binary(series: pd.Series) -> float:
    observed = series.dropna()
    if observed.empty:
        return np.nan
    return float((observed == 1).any())


def _incident_outcome(followup_outcome: float, baseline_positive: int) -> float:
    if baseline_positive == 1:
        return np.nan
    return followup_outcome


def _resolve_followup_windows_months(dataset_cfg: dict[str, Any]) -> list[int]:
    windows = dataset_cfg.get("followup_windows_months")
    if windows:
        return sorted({int(window) for window in windows})
    return [int(round(int(dataset_cfg.get("window_days", 183)) / 30.44))]


def _months_to_days(months: int) -> int:
    return int(round(float(months) * 30.44))


def _make_windowed_outcomes(
    followup: pd.DataFrame,
    baseline_aria_e: int,
    baseline_aria_h: int,
    baseline_aria_any: int,
    followup_windows_months: list[int],
    baseline_date: pd.Timestamp,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for months in followup_windows_months:
        suffix = f"{int(months)}mo"
        window_followup = followup.loc[
            followup["exam_date"] <= baseline_date + pd.Timedelta(days=_months_to_days(months))
        ].copy()
        aria_e = _reduce_followup_binary(window_followup["aria_e_label"])
        aria_h = _reduce_followup_binary(window_followup["aria_h_label"])
        aria_any = _reduce_followup_binary(window_followup["aria_any_label"])
        first_followup_date = window_followup["exam_date"].min() if not window_followup.empty else pd.NaT
        payload[f"n_followup_exams_{suffix}"] = int(len(window_followup))
        payload[f"days_to_first_followup_{suffix}"] = (
            float((first_followup_date - baseline_date).days)
            if pd.notna(first_followup_date)
            else np.nan
        )
        payload[f"aria_e_{suffix}"] = aria_e
        payload[f"aria_h_{suffix}"] = aria_h
        payload[f"aria_any_{suffix}"] = aria_any
        payload[f"aria_e_incident_{suffix}"] = _incident_outcome(aria_e, baseline_aria_e)
        payload[f"aria_h_incident_{suffix}"] = _incident_outcome(aria_h, baseline_aria_h)
        payload[f"aria_any_incident_{suffix}"] = _incident_outcome(aria_any, baseline_aria_any)
    return payload


def _extract_treatment_label(report_text: Any) -> str:
    text = str(report_text or "")
    for agent, pattern in TREATMENT_AGENT_PATTERNS:
        if pattern.search(text):
            return agent
    if TREATMENT_PATTERN.search(text):
        return "anti_amyloid_proxy"
    return "none_detected"


def _extract_diagnosis(report_text: Any) -> str:
    text = str(report_text or "").lower()
    if "mild cognitive impairment" in text or re.search(r"\bmci\b", text):
        return "mci"
    if "alzheimer" in text:
        return "alzheimer_disease"
    if "cognitive decline" in text or "dementia" in text:
        return "cognitive_disorder"
    return "unknown"
