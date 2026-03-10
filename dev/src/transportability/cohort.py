"""Build a fused cohort for transportability analysis.

Stacks UCSF treated patients (target population) with A4 placebo patients
(source trial controls). Harmonises columns so both sources share a common
schema before the estimator sees the data.

Only ARIA-H outcomes are valid for the fused analysis because A4 does not
capture ARIA-E.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal.datasets import build_a4_cohort
from causal.utils import load_yaml


def build_fused_cohort(config: dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame with columns usable by the transportability estimator.

    Rows are either:
      * UCSF treated patients  (site=1, treatment=1)
      * A4 placebo patients    (site=0, treatment=0)

    The ``site`` indicator doubles as the treatment indicator in the
    transportability framework: every UCSF row is treated, every A4 row
    is untreated.
    """
    transport_cfg = config["transportability"]
    a4_cfg = _build_a4_sub_config(config)
    ucsf_cfg = config.copy()

    # ---- A4 placebo arm ------------------------------------------------
    a4 = build_a4_cohort(a4_cfg)
    a4 = a4.loc[a4["treatment"] == 0].copy()  # placebo only
    a4["site"] = 0
    a4["source_dataset"] = "a4_placebo"

    # ---- UCSF treated arm -----------------------------------------------
    ucsf_builder = transport_cfg["ucsf_builder"]
    if ucsf_builder == "ucsf_proto":
        from proto_ucsf_aria.builder import build_ucsf_proto_cohort
        ucsf = build_ucsf_proto_cohort(config)
    elif ucsf_builder == "ucsf_risk_set":
        from proto_ucsf_aria.risk_set import build_ucsf_risk_set_cohort
        ucsf = build_ucsf_risk_set_cohort(config)
    else:
        raise ValueError(f"Unsupported UCSF builder for transportability: {ucsf_builder}")

    ucsf = ucsf.loc[ucsf["treatment"] == 1].copy()  # treated only
    ucsf["site"] = 1
    if "source_dataset" not in ucsf.columns:
        ucsf["source_dataset"] = "ucsf_treated"

    # ---- harmonise columns ----------------------------------------------
    shared_covariates: list[str] = transport_cfg["shared_covariates"]
    outcome_columns: list[str] = config["analysis"]["outcome_columns"]
    keep_columns = (
        ["patient_id", "site", "treatment"]
        + shared_covariates
        + outcome_columns
        + ["apoe_status", "source_dataset"]
    )
    # add optional id columns if present
    if "analysis_id" in ucsf.columns:
        keep_columns.append("analysis_id")
    keep_columns = list(dict.fromkeys(keep_columns))  # dedupe, preserve order

    # ensure all columns exist in both frames (fill missing with NaN)
    for col in keep_columns:
        if col not in a4.columns:
            a4[col] = np.nan
        if col not in ucsf.columns:
            ucsf[col] = np.nan

    fused = pd.concat(
        [ucsf[keep_columns], a4[keep_columns]],
        ignore_index=True,
    )

    # harmonise sex encoding across datasets
    if "sex" in fused.columns:
        fused["sex"] = _harmonise_sex(fused["sex"])

    # harmonise diagnosis if it's a shared covariate
    if "diagnosis" in shared_covariates:
        fused["diagnosis"] = _harmonise_diagnosis(fused["diagnosis"], fused["site"])

    n_ucsf = int((fused["site"] == 1).sum())
    n_a4 = int((fused["site"] == 0).sum())
    print(f"Fused cohort: {n_ucsf} UCSF treated + {n_a4} A4 placebo = {len(fused)} total")
    return fused.reset_index(drop=True)


def _build_a4_sub_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract an A4-compatible config dict from the transportability config."""
    a4_paths = config["transportability"]["a4_paths"]
    a4_settings = config["transportability"].get("a4_settings", {})
    return {
        "dataset": {
            "name": "a4",
            "builder": "a4",
            "paths": a4_paths,
            "window_days": a4_settings.get("window_days", 183),
            "followup_windows_months": a4_settings.get("followup_windows_months", [6]),
            "baseline_visit_code": a4_settings.get("baseline_visit_code", 1),
            "exclude_baseline_positive_for_incident": a4_settings.get(
                "exclude_baseline_positive_for_incident", True
            ),
            "active_treatment_labels": a4_settings.get(
                "active_treatment_labels", ["Solanezumab"]
            ),
            "placebo_labels": a4_settings.get("placebo_labels", ["Placebo"]),
        },
    }


def _harmonise_sex(sex: pd.Series) -> pd.Series:
    """Unify sex encoding: A4 uses 1/2, UCSF uses Female/Male."""
    mapping = {
        "1": "Female",
        "2": "Male",
        "female": "Female",
        "male": "Male",
        "Female": "Female",
        "Male": "Male",
        "F": "Female",
        "M": "Male",
        1: "Female",
        2: "Male",
    }
    return sex.map(lambda v: mapping.get(v, mapping.get(str(v).strip(), "Missing")))


def _harmonise_diagnosis(diagnosis: pd.Series, site: pd.Series) -> pd.Series:
    """Map A4's ``preclinical_ad`` to a category comparable with UCSF labels."""
    out = diagnosis.copy().fillna("unknown").astype(str).str.lower()
    # A4 participants are cognitively normal with elevated amyloid
    # Map to a shared vocabulary
    out = out.replace({
        "preclinical_ad": "preclinical_ad",
        "alzheimer_disease": "alzheimer_disease",
        "alzheimer": "alzheimer_disease",
        "mci": "mci",
        "cognitive_disorder": "cognitive_disorder",
    })
    return out
