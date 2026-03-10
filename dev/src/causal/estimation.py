from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal.preprocessing import make_age_quartile, prepare_model_frame


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    patient_level: pd.DataFrame
    subgroup_estimates: pd.DataFrame
    design_matrix: pd.DataFrame
    warnings: list[str]


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        n_rows = len(x)
        probs = np.full(n_rows, self.probability)
        return np.column_stack([1.0 - probs, probs])


def run_dr_analysis(
    cohort: pd.DataFrame,
    outcome_column: str,
    analysis_cfg: dict[str, Any],
    random_seed: int,
    bootstrap_iterations: int,
) -> AnalysisResult:
    estimand = str(analysis_cfg.get("estimand", "ate")).lower()
    treatment_column = analysis_cfg["treatment_column"]
    categorical_covariates = analysis_cfg["categorical_covariates"]
    numeric_covariates = analysis_cfg["numeric_covariates"]
    use_apoe = bool(analysis_cfg.get("use_apoe", True))
    apoe_column = analysis_cfg.get("apoe_column")

    required = [treatment_column, outcome_column] + categorical_covariates + numeric_covariates
    if use_apoe:
        if not apoe_column:
            raise ValueError("analysis.use_apoe is true, but no analysis.apoe_column was provided.")
        required.append(apoe_column)
    row_identifier_columns = ["patient_id"]
    if "analysis_id" in cohort.columns:
        row_identifier_columns = ["analysis_id", "patient_id"]
    working = cohort[required + row_identifier_columns].copy()
    working = working.dropna(subset=[treatment_column, outcome_column]).reset_index(drop=True)
    working["treatment"] = pd.to_numeric(working[treatment_column], errors="coerce").astype(int)
    working["outcome"] = pd.to_numeric(working[outcome_column], errors="coerce").astype(int)
    if use_apoe:
        apoe_text = working[apoe_column].fillna("Missing").astype(str).replace({"nan": "Missing", "": "Missing"})
        working["apoe_missing"] = apoe_text.eq("Missing").astype(int)
        working["apoe4_carrier"] = apoe_text.str.upper().str.contains("E4").astype(int)
        working["apoe_group"] = np.where(
            working["apoe_missing"] == 1,
            "missing",
            np.where(working["apoe4_carrier"] == 1, "carrier", "noncarrier"),
        )
    else:
        working["apoe_missing"] = 0
        working["apoe4_carrier"] = np.nan
        working["apoe_group"] = "not_used"
    working["age_quartile"] = make_age_quartile(working["age"])
    n_treated = int((working["treatment"] == 1).sum())
    n_control = int((working["treatment"] == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Treatment groups are not both present for {outcome_column}: "
            f"n_treated={n_treated}, n_control={n_control}."
        )

    model_categorical_covariates = list(categorical_covariates)
    if use_apoe:
        model_categorical_covariates.append("apoe_group")
    design, _ = prepare_model_frame(working, model_categorical_covariates, numeric_covariates)
    propensity = _fit_propensity_model(
        design,
        working["treatment"],
        analysis_cfg["propensity_model"],
        random_seed,
    )
    clip_cfg = analysis_cfg["propensity_clip"]
    propensity = propensity.clip(lower=float(clip_cfg["min"]), upper=float(clip_cfg["max"]))

    mu1, mu0 = _fit_outcome_models(
        design,
        working["treatment"],
        working["outcome"],
        analysis_cfg["outcome_model"],
        random_seed,
    )
    tau = _dr_pseudo_outcome(working["treatment"], working["outcome"], propensity, mu1, mu0)
    cate_pred = _fit_cate_model(design, tau, analysis_cfg["cate_model"], random_seed)
    target_weight = _target_weight(propensity, estimand)
    balance_weight = _balance_weight(working["treatment"], propensity, estimand)

    estimate = _weighted_mean(tau, target_weight)
    ci_lower, ci_upper = _bootstrap_effect(
        cohort=cohort,
        outcome_column=outcome_column,
        analysis_cfg=analysis_cfg,
        random_seed=random_seed,
        iterations=bootstrap_iterations,
    )

    patient_level_columns = row_identifier_columns + ["treatment", "outcome"] + categorical_covariates + numeric_covariates
    if use_apoe:
        patient_level_columns.extend([apoe_column, "apoe_missing", "apoe4_carrier", "apoe_group"])
    else:
        patient_level_columns.extend(["apoe_missing", "apoe4_carrier", "apoe_group"])
    patient_level_columns = list(dict.fromkeys(patient_level_columns))
    patient_level = working.reindex(columns=patient_level_columns).copy()
    patient_level["propensity_score"] = propensity
    patient_level["mu1_hat"] = mu1
    patient_level["mu0_hat"] = mu0
    patient_level["tau_hat"] = tau
    patient_level["cate_hat"] = cate_pred
    patient_level["target_weight"] = target_weight
    patient_level["balance_weight"] = balance_weight
    patient_level["estimand"] = estimand
    patient_level["age_quartile"] = working["age_quartile"].astype(str)

    subgroup_estimates = _make_subgroup_estimates(patient_level, include_apoe=use_apoe)
    warnings = []
    min_group_size_warning = int(analysis_cfg.get("min_group_size_warning", 5))
    if min(n_treated, n_control) < min_group_size_warning:
        warnings.append(
            f"Small treatment arm: n_treated={n_treated}, n_control={n_control}. "
            "Estimates may be unstable."
        )

    summary = {
        "outcome": outcome_column,
        "n": int(len(working)),
        "n_treated": n_treated,
        "n_control": n_control,
        "outcome_rate_treated": float(working.loc[working["treatment"] == 1, "outcome"].mean()),
        "outcome_rate_control": float(working.loc[working["treatment"] == 0, "outcome"].mean()),
        "estimand": estimand,
        "use_apoe": use_apoe,
        "estimate": estimate,
        "ate": estimate,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }
    return AnalysisResult(
        summary=summary,
        patient_level=patient_level,
        subgroup_estimates=subgroup_estimates,
        design_matrix=design,
        warnings=warnings,
    )


def _fit_propensity_model(
    x: pd.DataFrame,
    treatment: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    model = _make_classifier(model_cfg, random_seed)
    model.fit(x, treatment)
    return pd.Series(model.predict_proba(x)[:, 1], index=x.index)


def _fit_outcome_models(
    x: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> tuple[pd.Series, pd.Series]:
    treated_model = _fit_binary_outcome_model(x.loc[treatment == 1], outcome.loc[treatment == 1], model_cfg, random_seed)
    control_model = _fit_binary_outcome_model(x.loc[treatment == 0], outcome.loc[treatment == 0], model_cfg, random_seed)
    mu1 = pd.Series(treated_model.predict_proba(x)[:, 1], index=x.index)
    mu0 = pd.Series(control_model.predict_proba(x)[:, 1], index=x.index)
    return mu1, mu0


def _fit_binary_outcome_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> Any:
    if y.nunique(dropna=True) < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = _make_classifier(model_cfg, random_seed)
    model.fit(x, y)
    return model


def _make_classifier(model_cfg: dict[str, Any], random_seed: int) -> Any:
    model_type = model_cfg["type"]
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            random_state=random_seed,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            learning_rate=float(model_cfg.get("learning_rate", 0.05)),
            max_depth=int(model_cfg.get("max_depth", 2)),
            random_state=random_seed,
        )
    raise ValueError(f"Unsupported classifier type: {model_type}")


def _fit_cate_model(
    x: pd.DataFrame,
    tau: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    model_type = model_cfg["type"]
    if model_type != "random_forest":
        raise ValueError(f"Unsupported CATE model type: {model_type}")
    model = RandomForestRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 10)),
        random_state=random_seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
    model.fit(x, tau)
    return pd.Series(model.predict(x), index=x.index)


def _dr_pseudo_outcome(
    treatment: pd.Series,
    outcome: pd.Series,
    propensity: pd.Series,
    mu1: pd.Series,
    mu0: pd.Series,
) -> pd.Series:
    return (
        mu1
        - mu0
        + (treatment / propensity) * (outcome - mu1)
        - ((1 - treatment) / (1 - propensity)) * (outcome - mu0)
    )


def _target_weight(propensity: pd.Series, estimand: str) -> pd.Series:
    if estimand == "ate":
        return pd.Series(np.ones(len(propensity)), index=propensity.index, dtype=float)
    if estimand == "att":
        return propensity.astype(float)
    if estimand == "overlap":
        return (propensity * (1.0 - propensity)).astype(float)
    raise ValueError(f"Unsupported estimand: {estimand}")


def _balance_weight(treatment: pd.Series, propensity: pd.Series, estimand: str) -> pd.Series:
    if estimand == "ate":
        return treatment / propensity + (1 - treatment) / (1 - propensity)
    if estimand == "att":
        return treatment + (1 - treatment) * (propensity / (1 - propensity))
    if estimand == "overlap":
        return treatment * (1 - propensity) + (1 - treatment) * propensity
    raise ValueError(f"Unsupported estimand: {estimand}")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("Target weights sum to zero; cannot estimate effect.")
    return float(np.average(values, weights=weights))


def _bootstrap_effect(
    cohort: pd.DataFrame,
    outcome_column: str,
    analysis_cfg: dict[str, Any],
    random_seed: int,
    iterations: int,
) -> tuple[float | None, float | None]:
    if iterations <= 0:
        return None, None

    rng = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = cohort.iloc[rng.integers(0, len(cohort), size=len(cohort))].reset_index(drop=True)
        try:
            result = run_dr_analysis(
                cohort=sampled,
                outcome_column=outcome_column,
                analysis_cfg=analysis_cfg,
                random_seed=random_seed,
                bootstrap_iterations=0,
            )
        except Exception:
            continue
        estimates.append(float(result.summary["estimate"]))

    if not estimates:
        return None, None
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _make_subgroup_estimates(patient_level: pd.DataFrame, include_apoe: bool = True) -> pd.DataFrame:
    subgroup_frames = []
    if include_apoe:
        subgroup_frames.append(_aggregate_group(patient_level, "apoe4_carrier"))
    subgroup_frames.append(_aggregate_group(patient_level, "sex"))
    subgroup_frames.append(_aggregate_group(patient_level, "age_quartile"))
    return pd.concat(subgroup_frames, ignore_index=True)


def _aggregate_group(patient_level: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = patient_level.groupby(column, dropna=False).apply(_summarize_subgroup).reset_index()
    grouped["subgroup"] = column
    grouped["level"] = grouped[column].astype(str)
    return grouped[
        [
            "subgroup",
            "level",
            "n",
            "mean_tau_hat",
            "weighted_mean_tau_hat",
            "mean_cate_hat",
            "weighted_mean_cate_hat",
            "observed_outcome_rate",
            "weighted_outcome_rate",
        ]
    ]


def _summarize_subgroup(group: pd.DataFrame) -> pd.Series:
    weights = pd.to_numeric(group["target_weight"], errors="coerce").fillna(0.0)
    weight_sum = float(weights.sum())
    if weight_sum > 0:
        weighted_tau = float(np.average(group["tau_hat"], weights=weights))
        weighted_cate = float(np.average(group["cate_hat"], weights=weights))
        weighted_outcome = float(np.average(group["outcome"], weights=weights))
    else:
        weighted_tau = np.nan
        weighted_cate = np.nan
        weighted_outcome = np.nan
    return pd.Series(
        {
            "n": int(len(group)),
            "mean_tau_hat": float(group["tau_hat"].mean()),
            "weighted_mean_tau_hat": weighted_tau,
            "mean_cate_hat": float(group["cate_hat"].mean()),
            "weighted_mean_cate_hat": weighted_cate,
            "observed_outcome_rate": float(group["outcome"].mean()),
            "weighted_outcome_rate": weighted_outcome,
        }
    )
