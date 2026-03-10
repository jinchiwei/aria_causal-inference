"""Runner for transportability analysis.

Mirrors the structure of ``causal.runner`` but uses the fused-cohort
builder and the augmented IOSW estimator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from causal.preprocessing import prepare_balance_frame
from causal.utils import copy_config, create_run_dir, ensure_dir, load_yaml, write_json
from transportability.cohort import build_fused_cohort
from transportability.diagnostics import (
    compute_smd_table,
    participation_warnings,
    save_participation_overlap_plot,
    save_smd_plot,
)
from transportability.estimation import run_transport_analysis


def main(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    run_cfg = config["run"]
    analysis_cfg = config["analysis"]

    run_dir = create_run_dir(run_cfg["output_root"], run_cfg["run_descriptor"])
    copied_config = copy_config(config_path, run_dir)

    fused = build_fused_cohort(config)
    fused.to_csv(run_dir / "fused_cohort.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    outcome_columns = analysis_cfg["outcome_columns"]
    random_seed = int(run_cfg.get("random_seed", 42))
    bootstrap_iterations = int(analysis_cfg.get("bootstrap_iterations", 0))

    write_json(
        {
            "config_copy": str(copied_config),
            "n_rows_fused_cohort": int(len(fused)),
            "n_target_ucsf": int((fused["site"] == 1).sum()),
            "n_source_a4_placebo": int((fused["site"] == 0).sum()),
            "outcomes": outcome_columns,
            "method": "transportability_aiosw",
        },
        run_dir / "run_metadata.json",
    )

    for outcome_column in outcome_columns:
        outcome_dir = ensure_dir(run_dir / outcome_column)
        result = run_transport_analysis(
            fused=fused,
            outcome_column=outcome_column,
            analysis_cfg=analysis_cfg,
            random_seed=random_seed,
            bootstrap_iterations=bootstrap_iterations,
        )

        low = float(analysis_cfg["overlap_warning_thresholds"]["low"])
        high = float(analysis_cfg["overlap_warning_thresholds"]["high"])
        warnings = result.warnings + participation_warnings(
            result.patient_level["participation_prob"],
            result.patient_level["site"],
            low,
            high,
        )

        balance_frame = prepare_balance_frame(
            result.patient_level,
            analysis_cfg["categorical_covariates"] + ["apoe4_carrier"],
            analysis_cfg["numeric_covariates"],
        )
        smd_table = compute_smd_table(
            balance_frame,
            result.patient_level["site"],
            result.patient_level["iosw"],
        )

        result.patient_level.to_csv(outcome_dir / "patient_level_estimates.csv", index=False)
        result.subgroup_estimates.to_csv(outcome_dir / "subgroup_cate_estimates.csv", index=False)
        smd_table.to_csv(outcome_dir / "standardized_mean_differences.csv", index=False)
        save_participation_overlap_plot(
            result.patient_level["participation_prob"],
            result.patient_level["site"],
            outcome_dir / "participation_overlap.png",
        )
        save_smd_plot(smd_table, outcome_dir / "covariate_balance.png")

        summary = dict(result.summary)
        summary["warnings"] = " | ".join(warnings) if warnings else ""
        summary_rows.append(summary)

        write_json(
            {"summary": summary, "warnings": warnings},
            outcome_dir / "summary.json",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nRun directory: {run_dir}")
    return 0
