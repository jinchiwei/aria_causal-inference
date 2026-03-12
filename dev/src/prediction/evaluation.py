from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: pd.Series, pred_prob: pd.Series) -> dict[str, float | int | None]:
    y_true = pd.Series(y_true).astype(int)
    pred_prob = pd.Series(pred_prob).astype(float)

    metrics: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else None,
        "brier_score": float(brier_score_loss(y_true, pred_prob)) if len(y_true) else None,
    }

    if y_true.nunique() < 2:
        metrics["auroc"] = None
        metrics["average_precision"] = None
        return metrics

    metrics["auroc"] = float(roc_auc_score(y_true, pred_prob))
    metrics["average_precision"] = float(average_precision_score(y_true, pred_prob))
    return metrics


def make_calibration_table(
    y_true: pd.Series,
    pred_prob: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true.astype(int), "pred_prob": pred_prob.astype(float)})
    frame = frame.sort_values("pred_prob").reset_index(drop=True)

    try:
        frame["bin"] = pd.qcut(frame["pred_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.Series(["all"] * len(frame), index=frame.index)

    summary = (
        frame.groupby("bin", observed=False)
        .agg(
            n=("y_true", "size"),
            predicted_risk_mean=("pred_prob", "mean"),
            observed_risk_mean=("y_true", "mean"),
        )
        .reset_index()
    )
    summary["bin"] = summary["bin"].astype(str)
    return summary


def save_roc_curve(y_true: pd.Series, pred_prob: pd.Series, output_path: str | Path) -> None:
    if pd.Series(y_true).nunique() < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="Model ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_calibration_curve(calibration_table: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        calibration_table["predicted_risk_mean"],
        calibration_table["observed_risk_mean"],
        marker="o",
        label="Observed",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Mean Predicted Risk")
    ax.set_ylabel("Observed Risk")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def estimate_calibration_intercept_slope(
    y_true: pd.Series,
    pred_prob: pd.Series,
) -> tuple[float | None, float | None]:
    frame = pd.DataFrame({"y_true": y_true.astype(int), "pred_prob": pred_prob.astype(float)})
    eps = np.finfo(float).eps
    frame["pred_prob"] = frame["pred_prob"].clip(eps, 1 - eps)

    if frame["y_true"].nunique() < 2:
        return None, None

    logit_pred = np.log(frame["pred_prob"] / (1 - frame["pred_prob"]))
    design = np.column_stack([np.ones(len(logit_pred)), logit_pred])
    target = frame["y_true"].to_numpy()

    beta = np.zeros(design.shape[1])
    for _ in range(100):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), eps, None)
        z = eta + (target - mu) / w
        xtwx = design.T @ (w[:, None] * design)
        xtwz = design.T @ (w * z)
        try:
            beta_new = np.linalg.solve(xtwx, xtwz)
        except np.linalg.LinAlgError:
            return None, None
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[0]), float(beta[1])
