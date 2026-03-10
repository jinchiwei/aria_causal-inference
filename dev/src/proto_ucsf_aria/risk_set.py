from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from proto_ucsf_aria.builder import (
    TREATMENT_AGENT_PATTERNS,
    _coerce_annotation_label,
    _combine_binary_labels,
    _load_annotations,
    _load_apoe,
    _load_reports,
)


STRICT_TREATMENT_START_PATTERN = re.compile(
    r"(?:"
    r"\b(?:lecanemab|lecanamab|lecanumab|levanemab|donanemab|aducanumab|"
    r"gantenerumab|solanezumab|remternetug)\b.*\b(?:dose|infusion|therapy|treatment|"
    r"safety|screening|follow[- ]?up|post)\b|"
    r"\b(?:post dose|after (?:the )?(?:first|second|third|fourth|\d+)(?:st|nd|rd|th)? "
    r"(?:dose|infusion)|following (?:the )?(?:first|second|third|fourth|\d+)(?:st|nd|rd|th)? "
    r"(?:dose|infusion)|s/p .*infusion|dose ?#?\d+|first infusion was on|scheduled for infusion #\d+)\b"
    r")",
    flags=re.IGNORECASE,
)


def build_ucsf_risk_set_cohort(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    baseline_window_days = int(dataset_cfg.get("baseline_window_days", 365))
    followup_windows_months = _resolve_followup_windows_months(dataset_cfg)
    max_followup_window_days = max(_months_to_days(months) for months in followup_windows_months)
    controls_per_treated = int(dataset_cfg.get("controls_per_treated", 2))
    min_controls_per_treated = int(dataset_cfg.get("min_controls_per_treated", controls_per_treated))
    random_seed = int(config.get("run", {}).get("random_seed", 42))

    exams = _load_exam_history(dataset_cfg["paths"])
    patient_histories = {
        patient_id: history.sort_values(["exam_date", "accession"]).reset_index(drop=True)
        for patient_id, history in exams.groupby("patient_id", sort=False)
    }
    first_treatment_dates = exams.loc[exams["treatment_start_candidate"] == 1].groupby("patient_id")["exam_date"].min()

    matched_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    treated_counter = 0

    for treated_patient_id, index_date in first_treatment_dates.sort_values().items():
        treated_row = _build_analysis_row(
            patient_id=treated_patient_id,
            patient_history=patient_histories[treated_patient_id],
            index_date=index_date,
            own_treatment_date=index_date,
            baseline_window_days=baseline_window_days,
            followup_windows_months=followup_windows_months,
            max_followup_window_days=max_followup_window_days,
            treat=1,
        )
        if treated_row is None:
            continue

        control_candidates: list[dict[str, Any]] = []
        for control_patient_id, control_history in patient_histories.items():
            if control_patient_id == treated_patient_id:
                continue
            control_treatment_date = first_treatment_dates.get(control_patient_id, pd.NaT)
            if pd.notna(control_treatment_date) and control_treatment_date <= index_date:
                continue
            control_row = _build_analysis_row(
                patient_id=control_patient_id,
                patient_history=control_history,
                index_date=index_date,
                own_treatment_date=control_treatment_date,
                baseline_window_days=baseline_window_days,
                followup_windows_months=followup_windows_months,
                max_followup_window_days=max_followup_window_days,
                treat=0,
            )
            if control_row is None:
                continue
            control_candidates.append(control_row)

        if len(control_candidates) < min_controls_per_treated:
            continue

        treated_counter += 1
        matched_set_id = f"treated_{treated_counter:04d}"
        treated_row["matched_set_id"] = matched_set_id
        treated_row["available_control_count"] = int(len(control_candidates))
        treated_row["analysis_id"] = f"{matched_set_id}_treated"
        matched_rows.append(treated_row)

        chosen_indices = rng.choice(
            len(control_candidates),
            size=min(controls_per_treated, len(control_candidates)),
            replace=False,
        )
        for control_order, chosen_index in enumerate(np.atleast_1d(chosen_indices), start=1):
            control_row = dict(control_candidates[int(chosen_index)])
            control_row["matched_set_id"] = matched_set_id
            control_row["available_control_count"] = int(len(control_candidates))
            control_row["analysis_id"] = f"{matched_set_id}_control_{control_order:02d}"
            matched_rows.append(control_row)

    cohort = pd.DataFrame(matched_rows)
    if cohort.empty:
        raise ValueError("UCSF risk-set prototype cohort is empty after eligibility and matching.")
    return cohort.sort_values(["matched_set_id", "treatment"], ascending=[True, False]).reset_index(drop=True)


def _load_exam_history(paths: dict[str, str]) -> pd.DataFrame:
    treated_exams = _load_treated_exams(paths)
    control_exams = _load_external_control_exams(paths)
    exams = pd.concat([treated_exams, control_exams], ignore_index=True, sort=False)
    exams["patient_id"] = exams["patient_id"].astype(str).str.strip()
    exams["exam_date"] = _parse_mixed_date(exams["exam_started_date"])
    exams["order_date"] = _parse_mixed_date(exams["ordered_date"])
    exams["age"] = pd.to_numeric(exams["age"], errors="coerce")
    exams["report_text"] = exams["report_text"].fillna(exams.get("Report Text", "")).fillna("").astype(str)
    exams["sex"] = exams["sex"].fillna("Missing").astype(str)
    exams["race"] = "Missing"
    exams["education"] = np.nan
    exams["apoe_status"] = exams["apoe_status"].fillna("Missing").astype(str)
    exams["treatment_start_candidate"] = np.where(
        exams["source_patient_pool"].eq("external_control"),
        0,
        exams["report_text"].str.contains(STRICT_TREATMENT_START_PATTERN, na=False).astype(int),
    )
    exams["treatment_label"] = np.where(
        exams["source_patient_pool"].eq("external_control"),
        "none_detected",
        exams["report_text"].map(_extract_treatment_label),
    )
    exams["diagnosis"] = exams["report_text"].map(_extract_diagnosis)
    exams = exams.dropna(subset=["patient_id", "exam_date"]).copy()
    return exams.sort_values(["patient_id", "exam_date", "accession"]).reset_index(drop=True)


def _load_treated_exams(paths: dict[str, str]) -> pd.DataFrame:
    annotations = _load_annotations(paths["annotations"])
    reports = _load_reports(paths["reports"])
    apoe = _load_apoe(paths["apoe"])

    exams = annotations.merge(reports, on="accession", how="left")
    exams = exams.merge(apoe, on="patient_id", how="left")
    exams["aria_e_label"] = _coerce_annotation_label(exams["ARIA-E"])
    exams["aria_h_label"] = _coerce_annotation_label(exams["ARIA-H"])
    exams["aria_any_label"] = _combine_binary_labels(exams["aria_e_label"], exams["aria_h_label"])
    exams["source_patient_pool"] = "treated_search"
    return exams


def _load_external_control_exams(paths: dict[str, str]) -> pd.DataFrame:
    control_reports_path = paths.get("control_reports")
    control_annotations_path = paths.get("control_annotations")
    control_apoe_path = paths.get("control_apoe_curated")
    if not control_reports_path or not control_annotations_path or not control_apoe_path:
        return pd.DataFrame(
            columns=[
                "accession",
                "patient_id",
                "age",
                "sex",
                "ordered_date",
                "exam_started_date",
                "report_finalized_date",
                "report_text",
                "apoe_status",
                "apoe_note",
                "aria_e_label",
                "aria_h_label",
                "aria_any_label",
                "source_patient_pool",
            ]
        )

    reports = _load_reports(control_reports_path)
    annotations = _load_control_annotations(control_annotations_path)
    apoe = _load_curated_control_apoe(control_apoe_path)

    exams = annotations.merge(reports, on="accession", how="left")
    exams = exams.merge(apoe, on="patient_id", how="inner")
    exams["source_patient_pool"] = "external_control"
    return exams


def _load_control_annotations(path: str) -> pd.DataFrame:
    annotations = pd.read_csv(path).copy()
    annotations["accession"] = annotations["Accession Number"].astype(str).str.strip()
    aria_e_col = next(col for col in annotations.columns if col.startswith("aria_e_"))
    aria_h_col = next(col for col in annotations.columns if col.startswith("aria_h_"))
    annotations["aria_e_label"] = _coerce_annotation_label(annotations[aria_e_col])
    annotations["aria_h_label"] = _coerce_annotation_label(annotations[aria_h_col])
    annotations["aria_any_label"] = _combine_binary_labels(annotations["aria_e_label"], annotations["aria_h_label"])
    return annotations[["accession", "aria_e_label", "aria_h_label", "aria_any_label"]].copy()


def _load_curated_control_apoe(path: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    sheet_name = "relaxed_n2_ranked" if "relaxed_n2_ranked" in workbook.sheet_names else workbook.sheet_names[0]
    apoe = pd.read_excel(path, sheet_name=sheet_name).copy()
    apoe["Patient MRN"] = pd.to_numeric(apoe["Patient MRN"], errors="coerce").astype("Int64")
    apoe["patient_id"] = apoe["Patient MRN"].astype(str)
    apoe["apoe_status"] = (
        apoe["apoe4"]
        .astype(str)
        .replace({"nan": "Missing", "-1": "Missing", "": "Missing"})
        .fillna("Missing")
    )
    apoe["apoe_note"] = apoe.get("Note", "").fillna("").astype(str)
    apoe = apoe.sort_values(["patient_id", "apoe_status"]).drop_duplicates("patient_id", keep="first")
    return apoe[["patient_id", "apoe_status", "apoe_note"]].copy()


def _build_analysis_row(
    patient_id: str,
    patient_history: pd.DataFrame,
    index_date: pd.Timestamp,
    own_treatment_date: pd.Timestamp | float | None,
    baseline_window_days: int,
    followup_windows_months: list[int],
    max_followup_window_days: int,
    treat: int,
) -> dict[str, Any] | None:
    if pd.isna(index_date):
        return None
    if pd.notna(own_treatment_date) and pd.Timestamp(own_treatment_date) < index_date:
        return None

    baseline_window_start = index_date - pd.Timedelta(days=baseline_window_days)
    baseline_exams = patient_history.loc[
        (patient_history["exam_date"] >= baseline_window_start) & (patient_history["exam_date"] <= index_date)
    ].copy()
    if baseline_exams.empty:
        return None

    prior_exams = patient_history.loc[patient_history["exam_date"] < index_date].copy()
    if (prior_exams["aria_any_label"] == 1).any():
        return None

    followup_exams = patient_history.loc[
        (patient_history["exam_date"] > index_date)
        & (patient_history["exam_date"] <= index_date + pd.Timedelta(days=max_followup_window_days))
    ].copy()
    if followup_exams.empty:
        return None

    baseline_reference = baseline_exams.sort_values(["exam_date", "accession"]).iloc[-1]
    age = pd.to_numeric(baseline_reference.get("age"), errors="coerce")
    diagnosis = baseline_reference.get("diagnosis", "unknown")
    own_treatment_timestamp = pd.Timestamp(own_treatment_date) if pd.notna(own_treatment_date) else pd.NaT
    outcome_payload = _make_windowed_outcomes(followup_exams, followup_windows_months, index_date)

    return {
        "patient_id": str(patient_id),
        "treatment": int(treat),
        "treatment_label": baseline_reference.get("treatment_label", "none_detected"),
        "treatment_start_date": own_treatment_timestamp,
        "t0": index_date,
        "age": float(age) if pd.notna(age) else np.nan,
        "sex": baseline_reference.get("sex", "Missing"),
        "race": baseline_reference.get("race", "Missing"),
        "education": baseline_reference.get("education", np.nan),
        "apoe_status": baseline_reference.get("apoe_status", "Missing"),
        "diagnosis": diagnosis,
        "diagnosis_alzheimer_flag": int("alzheimer" in diagnosis),
        "diagnosis_mci_flag": int("mci" in diagnosis),
        "diagnosis_dementia_flag": int("dementia" in diagnosis),
        "baseline_exam_year": float(index_date.year),
        "baseline_imaging_count_365d": int(len(baseline_exams)),
        "days_since_last_baseline_exam": float((index_date - baseline_reference["exam_date"]).days),
        "source_dataset": "ucsf_aria_risk_set_proto",
        **outcome_payload,
    }


def _window_outcome(series: pd.Series) -> int:
    observed = series.dropna()
    if observed.empty:
        return np.nan
    return int((observed == 1).any())


def _resolve_followup_windows_months(dataset_cfg: dict[str, Any]) -> list[int]:
    windows = dataset_cfg.get("followup_windows_months")
    if windows:
        return sorted({int(window) for window in windows})
    return [int(round(int(dataset_cfg.get("followup_window_days", 183)) / 30.44))]


def _months_to_days(months: int) -> int:
    return int(round(float(months) * 30.44))


def _make_windowed_outcomes(
    followup_exams: pd.DataFrame,
    followup_windows_months: list[int],
    index_date: pd.Timestamp,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for months in followup_windows_months:
        suffix = f"{int(months)}mo"
        window_followup = followup_exams.loc[
            followup_exams["exam_date"] <= index_date + pd.Timedelta(days=_months_to_days(months))
        ].copy()
        payload[f"followup_exam_count_{suffix}"] = int(len(window_followup))
        payload[f"aria_e_{suffix}"] = _window_outcome(window_followup["aria_e_label"])
        payload[f"aria_h_{suffix}"] = _window_outcome(window_followup["aria_h_label"])
        payload[f"aria_any_{suffix}"] = _window_outcome(window_followup["aria_any_label"])
    return payload


def _parse_mixed_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    needs_numeric_parse = parsed.isna() & series.notna()
    if needs_numeric_parse.any():
        numeric = pd.to_numeric(series.where(needs_numeric_parse), errors="coerce")
        numeric_text = numeric.dropna().astype("Int64").astype(str)
        reparsed = pd.to_datetime(numeric_text, format="%Y%m%d", errors="coerce")
        parsed.loc[reparsed.index] = reparsed
    return parsed


def _extract_treatment_label(report_text: Any) -> str:
    text = str(report_text or "")
    for agent, pattern in TREATMENT_AGENT_PATTERNS:
        if pattern.search(text):
            return agent
    return "none_detected"


def _extract_diagnosis(report_text: Any) -> str:
    text = str(report_text or "").lower()
    flags: list[str] = []
    if "alzheimer" in text:
        flags.append("alzheimer")
    if "mild cognitive impairment" in text or re.search(r"\bamci\b|\bmci\b", text):
        flags.append("mci")
    if "dementia" in text:
        flags.append("dementia")
    if not flags:
        return "unknown"
    return "+".join(flags)
