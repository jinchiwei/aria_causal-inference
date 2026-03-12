from __future__ import annotations

from typing import Any

import pandas as pd

from causal.datasets import build_cohort


def build_prediction_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort = build_cohort(config)
    analysis_cfg = config["analysis"]

    outcome_column = analysis_cfg["outcome_column"]
    if outcome_column not in cohort.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in cohort.")

    treatment_filter = analysis_cfg.get("treatment_filter")
    if treatment_filter is not None:
        if "treatment" not in cohort.columns:
            raise ValueError("Config requested treatment_filter but cohort has no 'treatment' column.")
        cohort = cohort.loc[cohort["treatment"] == int(treatment_filter)].copy()

    cohort = cohort.loc[cohort[outcome_column].notna()].copy()
    cohort[outcome_column] = pd.to_numeric(cohort[outcome_column], errors="coerce")
    cohort = cohort.loc[cohort[outcome_column].isin([0, 1])].copy()
    cohort[outcome_column] = cohort[outcome_column].astype(int)

    time_column = analysis_cfg.get("time_column")
    if time_column:
        if time_column not in cohort.columns:
            raise ValueError(f"Time column '{time_column}' not found in cohort.")
        cohort[time_column] = pd.to_datetime(cohort[time_column], errors="coerce")

    metadata = {
        "n_rows": int(len(cohort)),
        "n_positive": int(cohort[outcome_column].sum()),
        "outcome_prevalence": float(cohort[outcome_column].mean()) if len(cohort) else None,
        "outcome_column": outcome_column,
        "time_column": time_column,
        "treatment_filter": treatment_filter,
    }
    return cohort.reset_index(drop=True), metadata
