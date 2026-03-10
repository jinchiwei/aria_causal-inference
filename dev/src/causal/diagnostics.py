from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def propensity_warnings(
    propensity: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> list[str]:
    warnings: list[str] = []
    low_count = int((propensity < low_threshold).sum())
    high_count = int((propensity > high_threshold).sum())
    if low_count:
        warnings.append(
            f"{low_count} observations have propensity scores below {low_threshold:.2f}."
        )
    if high_count:
        warnings.append(
            f"{high_count} observations have propensity scores above {high_threshold:.2f}."
        )
    return warnings


def compute_smd_table(
    design: pd.DataFrame,
    treatment: pd.Series,
    weights: pd.Series,
) -> pd.DataFrame:
    rows = []
    for column in design.columns:
        values = pd.to_numeric(design[column], errors="coerce").fillna(0.0)
        before = _smd(values, treatment)
        after = _weighted_smd(values, treatment, weights)
        rows.append(
            {
                "feature": column,
                "smd_before": before,
                "smd_after_weighting": after,
                "smd_after_ipw": after,
            }
        )
    return pd.DataFrame(rows).sort_values("smd_before", ascending=False).reset_index(drop=True)


def save_propensity_overlap_plot(
    propensity: pd.Series,
    treatment: pd.Series,
    output_path: str | Path,
) -> None:
    plot_df = pd.DataFrame({"propensity": propensity, "treatment": treatment.map({0: "Control", 1: "Treated"})})
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="propensity",
        hue="treatment",
        bins=25,
        stat="density",
        common_norm=False,
        alpha=0.5,
    )
    plt.title("Propensity Score Overlap")
    plt.xlabel("Estimated propensity score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_smd_plot(smd_table: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    plot_df = smd_table.head(top_n).copy()
    plot_df = plot_df.sort_values("smd_before", ascending=True)

    plt.figure(figsize=(9, max(4, 0.35 * len(plot_df))))
    plt.scatter(plot_df["smd_before"], plot_df["feature"], label="Before weighting")
    plt.scatter(plot_df["smd_after_weighting"], plot_df["feature"], label="After weighting")
    plt.axvline(0.1, linestyle="--", color="red", linewidth=1)
    plt.axvline(-0.1, linestyle="--", color="red", linewidth=1)
    plt.xlabel("Standardized mean difference")
    plt.title("Covariate Balance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _smd(values: pd.Series, treatment: pd.Series) -> float:
    treated = values[treatment == 1]
    control = values[treatment == 0]
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_sd)


def _weighted_smd(values: pd.Series, treatment: pd.Series, weights: pd.Series) -> float:
    treated_mask = treatment == 1
    control_mask = treatment == 0
    treated_mean = np.average(values[treated_mask], weights=weights[treated_mask])
    control_mean = np.average(values[control_mask], weights=weights[control_mask])

    treated_var = np.average(
        np.square(values[treated_mask] - treated_mean), weights=weights[treated_mask]
    )
    control_var = np.average(
        np.square(values[control_mask] - control_mean), weights=weights[control_mask]
    )
    pooled_sd = np.sqrt((treated_var + control_var) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((treated_mean - control_mean) / pooled_sd)
