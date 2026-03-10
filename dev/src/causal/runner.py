from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pandas as pd

from causal.datasets import build_cohort
from causal.diagnostics import (
    compute_smd_table,
    propensity_warnings,
    save_propensity_overlap_plot,
    save_smd_plot,
)
from causal.estimation import run_dr_analysis
from causal.preprocessing import prepare_balance_frame
from causal.utils import copy_config, create_run_dir, ensure_dir, load_yaml, write_json


def _extract_window_months(outcome_column: str) -> int | None:
    m = re.search(r"(\d+)mo$", outcome_column)
    return int(m.group(1)) if m else None


def _make_window_config(config: dict[str, Any], window_months: int) -> dict[str, Any]:
    """Deep-copy config with followup_windows_months restricted to a single window."""
    wc = copy.deepcopy(config)
    wc["dataset"]["followup_windows_months"] = [window_months]
    return wc


def main(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    run_cfg = config["run"]
    analysis_cfg = config["analysis"]

    run_dir = create_run_dir(run_cfg["output_root"], run_cfg["run_descriptor"])
    copied_config = copy_config(config_path, run_dir)

    outcome_columns = analysis_cfg["outcome_columns"]
    random_seed = int(run_cfg.get("random_seed", 42))
    bootstrap_iterations = int(analysis_cfg.get("bootstrap_iterations", 0))

    # Build a separate matched cohort for each follow-up window so that
    # eligibility (≥1 follow-up exam within the window) is enforced
    # independently per window.  This prevents the max-window cohort from
    # silently including controls who only have distant follow-up, which
    # biases the short-window analyses after dropna.
    unique_windows: list[int] = sorted(
        {w for col in outcome_columns if (w := _extract_window_months(col)) is not None}
    )
    per_window_cohorts: dict[int, pd.DataFrame] = {}
    if unique_windows:
        for w in unique_windows:
            wc = _make_window_config(config, w)
            cohort_w = build_cohort(wc)
            cohort_w.to_csv(run_dir / f"analysis_cohort_{w}mo.csv", index=False)
            per_window_cohorts[w] = cohort_w
        fallback_cohort = per_window_cohorts[unique_windows[-1]]
    else:
        # No windowed outcomes — fall back to single cohort (legacy path)
        fallback_cohort = build_cohort(config)
        fallback_cohort.to_csv(run_dir / "analysis_cohort.csv", index=False)

    write_json(
        {
            "config_copy": str(copied_config),
            "per_window_cohort_sizes": {
                f"{w}mo": int(len(df)) for w, df in per_window_cohorts.items()
            } if per_window_cohorts else {"all": int(len(fallback_cohort))},
            "outcomes": outcome_columns,
            "source_dataset": config["dataset"]["name"],
        },
        run_dir / "run_metadata.json",
    )

    summary_rows: list[dict[str, Any]] = []
    for outcome_column in outcome_columns:
        w = _extract_window_months(outcome_column)
        cohort = per_window_cohorts.get(w, fallback_cohort) if w is not None else fallback_cohort

        outcome_dir = ensure_dir(run_dir / outcome_column)
        result = run_dr_analysis(
            cohort=cohort,
            outcome_column=outcome_column,
            analysis_cfg=analysis_cfg,
            random_seed=random_seed,
            bootstrap_iterations=bootstrap_iterations,
        )

        low = float(analysis_cfg["overlap_warning_thresholds"]["low"])
        high = float(analysis_cfg["overlap_warning_thresholds"]["high"])
        warnings = result.warnings + propensity_warnings(
            result.patient_level["propensity_score"],
            low,
            high,
        )

        use_apoe = bool(analysis_cfg.get("use_apoe", True))
        balance_categorical_covariates = list(analysis_cfg["categorical_covariates"])
        if use_apoe:
            balance_categorical_covariates.append("apoe_group")
        balance_frame = prepare_balance_frame(
            result.patient_level,
            balance_categorical_covariates,
            analysis_cfg["numeric_covariates"],
        )
        smd_table = compute_smd_table(
            balance_frame,
            result.patient_level["treatment"],
            result.patient_level["balance_weight"],
        )

        result.patient_level.to_csv(outcome_dir / "patient_level_estimates.csv", index=False)
        result.subgroup_estimates.to_csv(outcome_dir / "subgroup_cate_estimates.csv", index=False)
        smd_table.to_csv(outcome_dir / "standardized_mean_differences.csv", index=False)
        save_propensity_overlap_plot(
            result.patient_level["propensity_score"],
            result.patient_level["treatment"],
            outcome_dir / "propensity_overlap.png",
        )
        save_smd_plot(smd_table, outcome_dir / "covariate_balance.png")

        summary = dict(result.summary)
        summary["warnings"] = " | ".join(warnings) if warnings else ""
        summary_rows.append(summary)

        write_json(
            {
                "summary": summary,
                "warnings": warnings,
            },
            outcome_dir / "summary.json",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nRun directory: {run_dir}")
    return 0
