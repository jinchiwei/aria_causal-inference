from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from causal.utils import copy_config, create_run_dir, ensure_dir, load_yaml, write_json
from prediction.datasets import build_prediction_dataset
from prediction.evaluation import (
    compute_binary_metrics,
    estimate_calibration_intercept_slope,
    make_calibration_table,
    save_calibration_curve,
    save_roc_curve,
    write_metrics,
)
from prediction.modeling import build_estimator, resolve_experiment_features


def _make_split(
    cohort: pd.DataFrame,
    outcome_column: str,
    split_cfg: dict[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    method = split_cfg.get("method", "random")
    test_size = float(split_cfg.get("test_size", 0.25))

    if method == "temporal":
        time_column = split_cfg.get("time_column")
        if not time_column:
            raise ValueError("Temporal split requires analysis.split.time_column.")
        dated = cohort.loc[cohort[time_column].notna()].sort_values(time_column).copy()
        if dated.empty:
            raise ValueError(f"No non-missing values available for temporal split column '{time_column}'.")
        cutoff_index = max(1, int(len(dated) * (1.0 - test_size)))
        cutoff_time = dated.iloc[cutoff_index - 1][time_column]
        train = cohort.loc[cohort[time_column].notna() & (cohort[time_column] <= cutoff_time)].copy()
        test = cohort.loc[cohort[time_column].notna() & (cohort[time_column] > cutoff_time)].copy()
        if train.empty or test.empty:
            raise ValueError("Temporal split produced an empty train or test set.")
        metadata = {
            "method": "temporal",
            "time_column": time_column,
            "cutoff_time": str(cutoff_time),
        }
        return train, test, metadata

    train_idx, test_idx = train_test_split(
        cohort.index,
        test_size=test_size,
        random_state=random_seed,
        stratify=cohort[outcome_column],
    )
    train = cohort.loc[train_idx].copy()
    test = cohort.loc[test_idx].copy()
    metadata = {
        "method": "random",
        "test_size": test_size,
    }
    return train, test, metadata


def main(config_path: str | Path) -> int:
    config = load_yaml(config_path)
    run_cfg = config["run"]
    analysis_cfg = config["analysis"]

    run_dir = create_run_dir(run_cfg["output_root"], run_cfg["run_descriptor"])
    copied_config = copy_config(config_path, run_dir)

    random_seed = int(run_cfg.get("random_seed", 42))
    n_calibration_bins = int(analysis_cfg.get("n_calibration_bins", 10))
    outcome_column = analysis_cfg["outcome_column"]
    id_column = analysis_cfg.get("id_column", "patient_id")
    numeric_features = list(analysis_cfg.get("numeric_features", []))
    categorical_features = list(analysis_cfg.get("categorical_features", []))

    cohort, cohort_metadata = build_prediction_dataset(config)
    cohort.to_csv(run_dir / "analysis_cohort.csv", index=False)

    train_df, test_df, split_metadata = _make_split(
        cohort=cohort,
        outcome_column=outcome_column,
        split_cfg=analysis_cfg.get("split", {}),
        random_seed=random_seed,
    )

    write_json(
        {
            "config_copy": str(copied_config),
            "outcome_column": outcome_column,
            "cohort_metadata": cohort_metadata,
            "split_metadata": split_metadata,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "source_dataset": config["dataset"]["name"],
        },
        run_dir / "run_metadata.json",
    )

    summary_rows: list[dict[str, Any]] = []
    experiments = list(analysis_cfg.get("experiments", []))
    if not experiments:
        raise ValueError("analysis.experiments must contain at least one experiment.")

    for experiment_cfg in experiments:
        experiment_name = experiment_cfg["name"]
        experiment_dir = ensure_dir(run_dir / experiment_name)

        exp_numeric, exp_categorical, missing_features = resolve_experiment_features(
            experiment_cfg,
            numeric_candidates=numeric_features,
            categorical_candidates=categorical_features,
        )

        estimator = build_estimator(
            model_type=experiment_cfg["model_type"],
            numeric_features=exp_numeric,
            categorical_features=exp_categorical,
            random_seed=random_seed,
        )

        used_features = exp_numeric + exp_categorical
        estimator.fit(train_df[used_features], train_df[outcome_column])

        train_pred = estimator.predict_proba(train_df[used_features])[:, 1]
        test_pred = estimator.predict_proba(test_df[used_features])[:, 1]

        patient_level = pd.concat(
            [
                pd.DataFrame(
                    {
                        id_column: train_df[id_column].values if id_column in train_df.columns else train_df.index,
                        "split": "train",
                        "outcome": train_df[outcome_column].values,
                        "predicted_risk": train_pred,
                    }
                ),
                pd.DataFrame(
                    {
                        id_column: test_df[id_column].values if id_column in test_df.columns else test_df.index,
                        "split": "test",
                        "outcome": test_df[outcome_column].values,
                        "predicted_risk": test_pred,
                    }
                ),
            ],
            ignore_index=True,
        )
        patient_level.to_csv(experiment_dir / "patient_level_predictions.csv", index=False)

        train_metrics = compute_binary_metrics(train_df[outcome_column], pd.Series(train_pred))
        test_metrics = compute_binary_metrics(test_df[outcome_column], pd.Series(test_pred))
        cal_intercept, cal_slope = estimate_calibration_intercept_slope(
            test_df[outcome_column],
            pd.Series(test_pred),
        )
        calibration_table = make_calibration_table(
            test_df[outcome_column],
            pd.Series(test_pred),
            n_bins=n_calibration_bins,
        )
        calibration_table.to_csv(experiment_dir / "calibration_table.csv", index=False)
        save_roc_curve(test_df[outcome_column], pd.Series(test_pred), experiment_dir / "roc_curve.png")
        save_calibration_curve(calibration_table, experiment_dir / "calibration_curve.png")

        metrics = {
            "experiment_name": experiment_name,
            "model_type": experiment_cfg["model_type"],
            "used_features": used_features,
            "missing_features": missing_features,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_calibration_intercept": cal_intercept,
            "test_calibration_slope": cal_slope,
        }
        write_metrics(metrics, experiment_dir / "metrics.json")

        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "model_type": experiment_cfg["model_type"],
                "n_features": len(used_features),
                "missing_features": " | ".join(missing_features),
                "train_auroc": train_metrics["auroc"],
                "test_auroc": test_metrics["auroc"],
                "test_average_precision": test_metrics["average_precision"],
                "test_brier_score": test_metrics["brier_score"],
                "test_calibration_intercept": cal_intercept,
                "test_calibration_slope": cal_slope,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nRun directory: {run_dir}")
    return 0
