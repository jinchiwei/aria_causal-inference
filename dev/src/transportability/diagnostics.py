"""Diagnostics for transportability analysis.

Provides participation-score overlap plots and covariate balance
assessment between the source (A4) and target (UCSF) populations,
before and after IOSW reweighting.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def participation_warnings(
    participation_prob: pd.Series,
    site: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> list[str]:
    """Flag source patients with extreme participation scores."""
    source_prob = participation_prob[site == 0]
    warnings: list[str] = []
    low_count = int((source_prob < low_threshold).sum())
    high_count = int((source_prob > high_threshold).sum())
    if low_count:
        warnings.append(
            f"{low_count} A4 placebo patients have P(S=1|X) < {low_threshold:.2f} "
            "(look very different from UCSF population)."
        )
    if high_count:
        warnings.append(
            f"{high_count} A4 placebo patients have P(S=1|X) > {high_threshold:.2f} "
            "(receive extreme IOSW weights)."
        )
    return warnings


def compute_smd_table(
    design: pd.DataFrame,
    site: pd.Series,
    iosw: pd.Series,
) -> pd.DataFrame:
    """SMD between target and IOSW-reweighted source, before and after weighting."""
    rows = []
    for column in design.columns:
        values = pd.to_numeric(design[column], errors="coerce").fillna(0.0)
        before = _smd(values, site)
        after = _weighted_smd(values, site, iosw)
        rows.append({"feature": column, "smd_before": before, "smd_after_iosw": after})
    return pd.DataFrame(rows).sort_values("smd_before", ascending=False, key=abs).reset_index(drop=True)


def save_participation_overlap_plot(
    participation_prob: pd.Series,
    site: pd.Series,
    output_path: str | Path,
) -> None:
    """Histogram of participation scores by site."""
    plot_df = pd.DataFrame({
        "P(S=1|X)": participation_prob,
        "Population": site.map({0: "A4 Placebo (source)", 1: "UCSF Treated (target)"}),
    })
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="P(S=1|X)",
        hue="Population",
        bins=25,
        stat="density",
        common_norm=False,
        alpha=0.5,
    )
    plt.title("Participation Score Overlap (Transportability)")
    plt.xlabel("Estimated P(S=1 | X)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_smd_plot(smd_table: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    """Scatter plot of covariate SMDs before/after IOSW."""
    plot_df = smd_table.head(top_n).copy()
    plot_df = plot_df.sort_values("smd_before", ascending=True, key=abs)

    plt.figure(figsize=(9, max(4, 0.35 * len(plot_df))))
    plt.scatter(plot_df["smd_before"], plot_df["feature"], label="Before IOSW")
    plt.scatter(plot_df["smd_after_iosw"], plot_df["feature"], label="After IOSW")
    plt.axvline(0.1, linestyle="--", color="red", linewidth=1)
    plt.axvline(-0.1, linestyle="--", color="red", linewidth=1)
    plt.xlabel("Standardized mean difference")
    plt.title("Covariate Balance: UCSF vs IOSW-reweighted A4 Placebo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _smd(values: pd.Series, site: pd.Series) -> float:
    """Unweighted SMD between target (site=1) and source (site=0)."""
    target = values[site == 1]
    source = values[site == 0]
    pooled_sd = np.sqrt((target.var(ddof=1) + source.var(ddof=1)) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((target.mean() - source.mean()) / pooled_sd)


def _weighted_smd(values: pd.Series, site: pd.Series, iosw: pd.Series) -> float:
    """SMD between target (unweighted) and IOSW-reweighted source."""
    target = values[site == 1]
    source = values[site == 0]
    source_weights = iosw[site == 0]

    target_mean = float(target.mean())
    source_mean = float(np.average(source, weights=source_weights))

    target_var = float(target.var(ddof=1))
    source_var = float(np.average(np.square(source - source_mean), weights=source_weights))

    pooled_sd = np.sqrt((target_var + source_var) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((target_mean - source_mean) / pooled_sd)
