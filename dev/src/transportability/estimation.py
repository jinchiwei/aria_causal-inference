"""Transportability estimator: augmented inverse-odds-of-sampling weighting.

Framework (Dahabreh et al. 2020, Westreich et al. 2017):

We have two populations:
  * Target (S=1): UCSF patients, all treated (T=1).
  * Source (S=0): A4 trial placebo arm, all untreated (T=0).

Estimand — ATE in the *target* population:
    tau_target = E_target[Y(1)] - E_target[Y(0)]

Identification:
  * E_target[Y(1)] is directly estimable from UCSF treated outcomes.
  * E_target[Y(0)] is not observed — no untreated patients in UCSF.

    We *transport* E[Y(0)|X] from A4 placebo to the UCSF covariate
    distribution using inverse-odds-of-sampling weights (IOSW):

        w(X) = P(S=1|X) / P(S=0|X)          (odds of being in target)

    With augmentation (AIOSW / doubly robust):

        E_target[Y(0)] = (1/N_target) * sum over target of mu0_hat(X_i)
                       + (1/N_target) * sum over source of w(X_j)*(Y_j - mu0_hat(X_j))

Key assumptions:
  1. Mean exchangeability over S: E[Y(0)|X, S=1] = E[Y(0)|X, S=0]
  2. Positivity of sampling: P(S=0|X) > 0 for all X with P(S=1|X) > 0
  3. Outcome model or participation model is correctly specified (DR)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from causal.preprocessing import make_age_quartile, prepare_model_frame


@dataclass
class TransportResult:
    summary: dict[str, Any]
    patient_level: pd.DataFrame
    subgroup_estimates: pd.DataFrame
    design_matrix: pd.DataFrame
    warnings: list[str]


def run_transport_analysis(
    fused: pd.DataFrame,
    outcome_column: str,
    analysis_cfg: dict[str, Any],
    random_seed: int,
    bootstrap_iterations: int,
) -> TransportResult:
    """Run the augmented IOSW transportability estimator on a fused cohort."""

    categorical_covariates = analysis_cfg["categorical_covariates"]
    numeric_covariates = analysis_cfg["numeric_covariates"]
    apoe_column = analysis_cfg["apoe_column"]

    required = (
        ["site", "treatment", outcome_column, apoe_column]
        + categorical_covariates
        + numeric_covariates
    )
    row_id_cols = ["patient_id"]
    if "analysis_id" in fused.columns:
        row_id_cols = ["analysis_id", "patient_id"]

    working = fused[required + row_id_cols + ["source_dataset"]].copy()
    working = working.dropna(subset=["site", outcome_column]).reset_index(drop=True)
    working["site"] = working["site"].astype(int)
    working["treatment"] = working["site"]  # site==1 ↔ treated, site==0 ↔ control
    working["outcome"] = pd.to_numeric(working[outcome_column], errors="coerce").astype(float)
    working["apoe4_carrier"] = (
        working[apoe_column].fillna("").astype(str).str.upper().str.contains("E4").astype(int)
    )
    working["age_quartile"] = make_age_quartile(working["age"])

    target = working.loc[working["site"] == 1]
    source = working.loc[working["site"] == 0]
    n_target = len(target)
    n_source = len(source)
    if n_target == 0 or n_source == 0:
        raise ValueError(
            f"Fused cohort must have rows from both sites for {outcome_column}: "
            f"n_target(UCSF)={n_target}, n_source(A4)={n_source}."
        )

    # ---- design matrix (shared covariates) ------------------------------
    design, _ = prepare_model_frame(
        working, categorical_covariates + ["apoe4_carrier"], numeric_covariates
    )

    # ---- participation model: P(S=1 | X) --------------------------------
    participation_cfg = analysis_cfg.get(
        "participation_model",
        analysis_cfg.get("propensity_model", {"type": "logistic_regression"}),
    )
    participation_prob = _fit_participation_model(
        design, working["site"], participation_cfg, random_seed
    )
    clip_cfg = analysis_cfg["propensity_clip"]
    participation_prob = participation_prob.clip(
        lower=float(clip_cfg["min"]), upper=float(clip_cfg["max"])
    )
    # IOSW: w(X) = P(S=1|X) / P(S=0|X) = p / (1-p)
    iosw = participation_prob / (1.0 - participation_prob)

    # ---- outcome model: E[Y(0) | X] fitted on A4 placebo ---------------
    outcome_cfg = analysis_cfg["outcome_model"]
    mu0_hat = _fit_control_outcome_model(
        design, working["site"], working["outcome"], outcome_cfg, random_seed
    )

    # ---- point estimates ------------------------------------------------
    # E_target[Y(1)]: simple mean of outcomes among UCSF treated
    ey1_target = float(target["outcome"].mean())

    # E_target[Y(0)]: augmented IOSW (doubly robust)
    # = (1/N_target) * [ sum_{target} mu0(X_i) + sum_{source} w(X_j)*(Y_j - mu0(X_j)) ]
    outcome_model_term = mu0_hat.loc[target.index].sum()
    bias_correction_term = (iosw.loc[source.index] * (
        working.loc[source.index, "outcome"] - mu0_hat.loc[source.index]
    )).sum()
    ey0_target = float((outcome_model_term + bias_correction_term) / n_target)

    ate = ey1_target - ey0_target

    # ---- individual-level pseudo-outcomes for CATE ----------------------
    # For target (UCSF) patients: tau_i = Y_i - mu0(X_i)
    # For source (A4) patients: not directly interpretable, but we store mu0
    tau_target = target["outcome"] - mu0_hat.loc[target.index]
    tau_source = pd.Series(np.nan, index=source.index)
    tau = pd.concat([tau_target, tau_source]).reindex(working.index)

    # ---- CATE model on target patients only -----------------------------
    cate_cfg = analysis_cfg["cate_model"]
    cate_pred = _fit_cate_on_target(
        design.loc[target.index], tau_target, cate_cfg, random_seed
    )
    # predict for all rows (including source) for completeness
    cate_all = pd.Series(np.nan, index=working.index)
    cate_all.loc[target.index] = cate_pred

    # ---- bootstrap CI ---------------------------------------------------
    ci_lower, ci_upper = _bootstrap_transport_ate(
        fused=fused,
        outcome_column=outcome_column,
        analysis_cfg=analysis_cfg,
        random_seed=random_seed,
        iterations=bootstrap_iterations,
    )

    # ---- patient-level output -------------------------------------------
    out_cols = row_id_cols + ["site", "treatment", "outcome", "source_dataset"] + categorical_covariates + numeric_covariates + [apoe_column, "apoe4_carrier"]
    out_cols = list(dict.fromkeys(out_cols))
    patient_level = working.reindex(columns=out_cols).copy()
    patient_level["participation_prob"] = participation_prob
    patient_level["iosw"] = iosw
    patient_level["mu0_hat"] = mu0_hat
    patient_level["tau_hat"] = tau
    patient_level["cate_hat"] = cate_all
    patient_level["age_quartile"] = working["age_quartile"].astype(str)

    # ---- subgroup estimates (target only) --------------------------------
    subgroup_estimates = _make_subgroup_estimates(
        patient_level.loc[patient_level["site"] == 1]
    )

    # ---- warnings -------------------------------------------------------
    warnings: list[str] = []
    min_group = int(analysis_cfg.get("min_group_size_warning", 5))
    if min(n_target, n_source) < min_group:
        warnings.append(
            f"Small group: n_target={n_target}, n_source={n_source}."
        )
    # check participation score overlap
    source_participation = participation_prob.loc[source.index]
    n_extreme_low = int((source_participation < float(clip_cfg["min"])).sum())
    n_extreme_high = int((source_participation > float(clip_cfg["max"])).sum())
    if n_extreme_low > 0:
        warnings.append(
            f"{n_extreme_low} A4 patients have participation P(S=1|X) near 0 "
            "(poor covariate overlap — these patients look unlike any UCSF patient)."
        )
    if n_extreme_high > 0:
        warnings.append(
            f"{n_extreme_high} A4 patients have participation P(S=1|X) near 1 "
            "(extreme IOSW — these patients dominate the transported estimate)."
        )

    summary = {
        "outcome": outcome_column,
        "n_total": int(len(working)),
        "n_target_ucsf": n_target,
        "n_source_a4_placebo": n_source,
        "ey1_target": ey1_target,
        "ey0_target_transported": ey0_target,
        "ate": ate,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }
    return TransportResult(
        summary=summary,
        patient_level=patient_level,
        subgroup_estimates=subgroup_estimates,
        design_matrix=design,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_participation_model(
    x: pd.DataFrame,
    site: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    """Fit P(S=1 | X) — probability of being in the target (UCSF) population."""
    model = _make_classifier(model_cfg, random_seed)
    model.fit(x, site)
    return pd.Series(model.predict_proba(x)[:, 1], index=x.index)


def _fit_control_outcome_model(
    x: pd.DataFrame,
    site: pd.Series,
    outcome: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    """Fit E[Y(0) | X] using only source (A4 placebo) patients, predict for all."""
    source_mask = site == 0
    x_source = x.loc[source_mask]
    y_source = outcome.loc[source_mask]

    if y_source.nunique(dropna=True) < 2:
        # degenerate: return constant
        prob = float(y_source.mean()) if len(y_source) > 0 else 0.0
        return pd.Series(prob, index=x.index)

    model = _make_classifier(model_cfg, random_seed)
    model.fit(x_source, y_source.astype(int))
    return pd.Series(model.predict_proba(x)[:, 1], index=x.index)


def _fit_cate_on_target(
    x_target: pd.DataFrame,
    tau_target: pd.Series,
    model_cfg: dict[str, Any],
    random_seed: int,
) -> pd.Series:
    """Fit a CATE model on target-population pseudo-outcomes."""
    if model_cfg["type"] != "random_forest":
        raise ValueError(f"Unsupported CATE model: {model_cfg['type']}")
    model = RandomForestRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 10)),
        random_state=random_seed,
        n_jobs=int(model_cfg.get("n_jobs", -1)),
    )
    clean_mask = tau_target.notna()
    if clean_mask.sum() < 5:
        return pd.Series(float(tau_target.mean()), index=x_target.index)
    model.fit(x_target.loc[clean_mask], tau_target.loc[clean_mask])
    return pd.Series(model.predict(x_target), index=x_target.index)


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


def _bootstrap_transport_ate(
    fused: pd.DataFrame,
    outcome_column: str,
    analysis_cfg: dict[str, Any],
    random_seed: int,
    iterations: int,
) -> tuple[float | None, float | None]:
    if iterations <= 0:
        return None, None

    rng = np.random.default_rng(random_seed)
    estimates: list[float] = []

    # stratified bootstrap: resample within each site to preserve structure
    target_idx = fused.index[fused["site"] == 1].to_numpy()
    source_idx = fused.index[fused["site"] == 0].to_numpy()

    for _ in range(iterations):
        boot_target = fused.iloc[rng.choice(target_idx, size=len(target_idx), replace=True)]
        boot_source = fused.iloc[rng.choice(source_idx, size=len(source_idx), replace=True)]
        boot_fused = pd.concat([boot_target, boot_source], ignore_index=True)
        try:
            result = run_transport_analysis(
                fused=boot_fused,
                outcome_column=outcome_column,
                analysis_cfg=analysis_cfg,
                random_seed=random_seed,
                bootstrap_iterations=0,
            )
        except Exception:
            continue
        estimates.append(float(result.summary["ate"]))

    if not estimates:
        return None, None
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def _make_subgroup_estimates(target_level: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in ("apoe4_carrier", "sex", "age_quartile"):
        if col in target_level.columns:
            frames.append(_aggregate_group(target_level, col))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_group(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(
            n=("patient_id", "size"),
            mean_tau_hat=("tau_hat", "mean"),
            mean_cate_hat=("cate_hat", "mean"),
            observed_outcome_rate=("outcome", "mean"),
        )
        .reset_index()
    )
    grouped["subgroup"] = column
    grouped["level"] = grouped[column].astype(str)
    return grouped[["subgroup", "level", "n", "mean_tau_hat", "mean_cate_hat", "observed_outcome_rate"]]
