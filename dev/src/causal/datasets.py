from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from causal.utils import normalize_string, read_table


ANTI_AMYLOID_KEYWORDS = (
    "lecanemab",
    "donanemab",
    "aducanumab",
    "gantenerumab",
    "solanezumab",
    "remternetug",
)


def build_cohort(config: dict[str, Any]) -> pd.DataFrame:
    builder = config["dataset"]["builder"]
    if builder == "a4":
        return build_a4_cohort(config)
    if builder == "ucsf_proto":
        from proto_ucsf_aria.builder import build_ucsf_proto_cohort

        return build_ucsf_proto_cohort(config)
    if builder == "ucsf_risk_set":
        from proto_ucsf_aria.risk_set import build_ucsf_risk_set_cohort

        return build_ucsf_risk_set_cohort(config)
    if builder == "prebuilt":
        return build_prebuilt_cohort(config)
    raise ValueError(f"Unsupported dataset builder: {builder}")


def build_prebuilt_cohort(config: dict[str, Any]) -> pd.DataFrame:
    path = config["dataset"]["paths"]["cohort"]
    cohort = read_table(path)
    return cohort.copy()


def build_a4_cohort(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    paths = dataset_cfg["paths"]
    followup_windows_months = _resolve_followup_windows_months(dataset_cfg)
    max_window_days = max(_months_to_days(months) for months in followup_windows_months)
    exclude_baseline_positive = bool(dataset_cfg.get("exclude_baseline_positive_for_incident", True))

    baseline = _load_a4_baseline(paths["adqs"], dataset_cfg.get("baseline_visit_code", 1))
    dose = _aggregate_a4_dose(paths["dose"], max_window_days)
    mri = _aggregate_a4_mri(paths["mri_reads"], followup_windows_months)

    cohort = baseline.merge(dose, on="patient_id", how="left").merge(mri, on="patient_id", how="left")

    active_labels = set(dataset_cfg.get("active_treatment_labels", ["Solanezumab"]))
    cohort["treatment"] = cohort["treatment_label"].isin(active_labels).astype(int)

    for column in (
        "first_dose_mg",
        "max_dose_mg_6mo",
        "cumulative_dose_mg_6mo",
        "dose_events_6mo",
    ):
        if column in cohort.columns:
            cohort[column] = np.where(cohort["treatment"] == 1, cohort[column].fillna(0.0), 0.0)

    for column in (
        "baseline_definite_mch",
        "baseline_definite_ss",
        "baseline_lobar",
        "baseline_deep",
    ):
        if column in cohort.columns:
            cohort[column] = cohort[column].fillna(0).astype(int)
    for months in followup_windows_months:
        suffix = f"{int(months)}mo"
        for column in (
            f"microhemorrhage_{suffix}",
            f"superficial_siderosis_{suffix}",
            f"aria_h_{suffix}",
            f"incident_microhemorrhage_{suffix}",
            f"incident_superficial_siderosis_{suffix}",
            f"aria_h_incident_{suffix}",
        ):
            if column in cohort.columns:
                cohort[column] = cohort[column].fillna(0).astype(int)

    cohort["baseline_aria_h_positive"] = (
        (cohort["baseline_definite_mch"] == 1) | (cohort["baseline_definite_ss"] == 1)
    ).astype(int)

    if exclude_baseline_positive:
        cohort = cohort.loc[cohort["baseline_aria_h_positive"] == 0].copy()

    cohort["diagnosis"] = "preclinical_ad"
    cohort["source_dataset"] = "a4"
    return cohort.sort_values("patient_id").reset_index(drop=True)


def _load_a4_baseline(adqs_path: str, baseline_visit_code: int) -> pd.DataFrame:
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
    adqs = read_table(adqs_path, usecols=columns, low_memory=False)
    adqs["VISITCD"] = pd.to_numeric(adqs["VISITCD"], errors="coerce")
    preferred = adqs.loc[adqs["VISITCD"] == baseline_visit_code].copy()
    if preferred.empty:
        preferred = adqs.copy()
    preferred = preferred.sort_values(["BID", "VISITCD"]).drop_duplicates("BID", keep="first")

    preferred = preferred.rename(
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
    return preferred[
        [
            "patient_id",
            "treatment_label",
            "age",
            "sex",
            "race",
            "education",
            "ethnicity",
            "apoe_status",
            "baseline_bmi",
            "baseline_amyloid_centiloid",
            "baseline_amyloid_suvr",
        ]
    ].copy()


def _aggregate_a4_dose(dose_path: str, window_days: int) -> pd.DataFrame:
    columns = [
        "BID",
        "DONE",
        "STARTDATE_DAYS_T0",
        "DOSELEVEL",
        "BLINDDOSE",
        "BLINDCUMDOSE",
    ]
    dose = read_table(dose_path, usecols=columns)
    dose["STARTDATE_DAYS_T0"] = pd.to_numeric(dose["STARTDATE_DAYS_T0"], errors="coerce")
    dose["BLINDDOSE"] = pd.to_numeric(dose["BLINDDOSE"], errors="coerce")
    dose["BLINDCUMDOSE"] = pd.to_numeric(dose["BLINDCUMDOSE"], errors="coerce")
    dose["parsed_doselevel_mg"] = dose["DOSELEVEL"].map(_parse_mg_string)
    dose["dose_mg"] = dose["parsed_doselevel_mg"].fillna(dose["BLINDDOSE"])
    dose = dose.loc[dose["DONE"].astype(str).str.lower() == "yes"].copy()

    dose = dose.sort_values(["BID", "STARTDATE_DAYS_T0"])
    within_window = dose.loc[
        dose["STARTDATE_DAYS_T0"].between(0, window_days, inclusive="both")
    ].copy()

    summary = within_window.groupby("BID").agg(
        first_dose_day_t0=("STARTDATE_DAYS_T0", "min"),
        first_dose_mg=("dose_mg", "first"),
        max_dose_mg_6mo=("dose_mg", "max"),
        cumulative_dose_mg_6mo=("BLINDCUMDOSE", "max"),
        dose_events_6mo=("dose_mg", "size"),
    )
    summary = summary.reset_index().rename(columns={"BID": "patient_id"})
    return summary


def _aggregate_a4_mri(mri_path: str, followup_windows_months: list[int]) -> pd.DataFrame:
    columns = [
        "BID",
        "STUDYDATE_DAYS_T0",
        "Definite.MCH",
        "Lobar",
        "Deep",
        "Definite.SS",
    ]
    mri = read_table(mri_path, usecols=columns)
    mri["STUDYDATE_DAYS_T0"] = pd.to_numeric(mri["STUDYDATE_DAYS_T0"], errors="coerce")
    for column in ("Definite.MCH", "Lobar", "Deep", "Definite.SS"):
        mri[column] = pd.to_numeric(mri[column], errors="coerce").fillna(0)

    baseline = mri.loc[mri["STUDYDATE_DAYS_T0"] < 0].copy()
    max_window_days = max(_months_to_days(months) for months in followup_windows_months)
    followup = mri.loc[mri["STUDYDATE_DAYS_T0"].between(0, max_window_days, inclusive="both")].copy()

    baseline_summary = baseline.groupby("BID").agg(
        baseline_definite_mch=("Definite.MCH", "max"),
        baseline_definite_ss=("Definite.SS", "max"),
        baseline_lobar=("Lobar", "max"),
        baseline_deep=("Deep", "max"),
    )

    summary = baseline_summary.copy()
    if not followup.empty:
        for months in followup_windows_months:
            suffix = f"{int(months)}mo"
            window_days = _months_to_days(months)
            window_followup = followup.loc[
                followup["STUDYDATE_DAYS_T0"].between(0, window_days, inclusive="both")
            ].copy()
            if window_followup.empty:
                continue
            window_followup[f"microhemorrhage_{suffix}"] = (window_followup["Definite.MCH"] > 0).astype(int)
            window_followup[f"superficial_siderosis_{suffix}"] = (window_followup["Definite.SS"] > 0).astype(int)
            window_followup[f"aria_h_{suffix}"] = (
                (window_followup[f"microhemorrhage_{suffix}"] == 1)
                | (window_followup[f"superficial_siderosis_{suffix}"] == 1)
            ).astype(int)
            window_summary = window_followup.groupby("BID").agg(
                **{
                    f"microhemorrhage_{suffix}": (f"microhemorrhage_{suffix}", "max"),
                    f"superficial_siderosis_{suffix}": (f"superficial_siderosis_{suffix}", "max"),
                    f"aria_h_{suffix}": (f"aria_h_{suffix}", "max"),
                    f"first_followup_mri_day_t0_{suffix}": ("STUDYDATE_DAYS_T0", "min"),
                }
            )
            summary = summary.join(window_summary, how="outer")

    summary = summary.reset_index()
    summary = summary.rename(columns={"BID": "patient_id"})
    for months in followup_windows_months:
        suffix = f"{int(months)}mo"
        for column in (
            f"microhemorrhage_{suffix}",
            f"superficial_siderosis_{suffix}",
            f"aria_h_{suffix}",
            f"first_followup_mri_day_t0_{suffix}",
        ):
            if column not in summary.columns:
                summary[column] = np.nan
        summary[f"incident_microhemorrhage_{suffix}"] = (
            (summary[f"microhemorrhage_{suffix}"].fillna(0) == 1)
            & (summary["baseline_definite_mch"].fillna(0) == 0)
        ).astype(int)
        summary[f"incident_superficial_siderosis_{suffix}"] = (
            (summary[f"superficial_siderosis_{suffix}"].fillna(0) == 1)
            & (summary["baseline_definite_ss"].fillna(0) == 0)
        ).astype(int)
        summary[f"aria_h_incident_{suffix}"] = (
            (summary[f"incident_microhemorrhage_{suffix}"] == 1)
            | (summary[f"incident_superficial_siderosis_{suffix}"] == 1)
        ).astype(int)
    return summary


def _parse_mg_string(value: Any) -> float | None:
    text = normalize_string(value)
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*mg", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _resolve_followup_windows_months(dataset_cfg: dict[str, Any]) -> list[int]:
    windows = dataset_cfg.get("followup_windows_months")
    if windows:
        return sorted({int(window) for window in windows})
    return [int(round(int(dataset_cfg.get("window_days", 183)) / 30.44))]


def _months_to_days(months: int) -> int:
    return int(round(float(months) * 30.44))
