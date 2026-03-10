from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_apoe4_indicator(
    df: pd.DataFrame,
    apoe_column: str,
    output_column: str = "apoe4_carrier",
) -> pd.DataFrame:
    out = df.copy()
    out[output_column] = (
        out[apoe_column]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains("E4")
        .astype(int)
    )
    return out


def prepare_model_frame(
    df: pd.DataFrame,
    categorical_covariates: list[str],
    numeric_covariates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = pd.DataFrame(index=df.index)
    metadata: dict[str, Any] = {"numeric_fill_values": {}, "categorical_levels": {}}

    if numeric_covariates:
        numeric = df[numeric_covariates].apply(pd.to_numeric, errors="coerce").copy()
        fill_values = numeric.median().fillna(0.0)
        numeric = numeric.fillna(fill_values)
        scaler = StandardScaler()
        numeric_scaled = pd.DataFrame(
            scaler.fit_transform(numeric),
            index=df.index,
            columns=numeric_covariates,
        )
        work = pd.concat([work, numeric_scaled], axis=1)
        metadata["numeric_fill_values"] = fill_values.to_dict()

    if categorical_covariates:
        categorical = (
            df[categorical_covariates]
            .copy()
            .fillna("Missing")
            .astype(str)
            .replace({"nan": "Missing"})
        )
        encoded = pd.get_dummies(categorical, prefix=categorical_covariates, drop_first=False)
        work = pd.concat([work, encoded], axis=1)
        metadata["categorical_levels"] = {
            column: sorted(categorical[column].unique().tolist()) for column in categorical_covariates
        }

    return work, metadata


def prepare_balance_frame(
    df: pd.DataFrame,
    categorical_covariates: list[str],
    numeric_covariates: list[str],
) -> pd.DataFrame:
    balance = pd.DataFrame(index=df.index)
    if numeric_covariates:
        numeric = df[numeric_covariates].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median()).fillna(0.0)
        balance = pd.concat([balance, numeric], axis=1)

    if categorical_covariates:
        categorical = df[categorical_covariates].fillna("Missing").astype(str)
        balance = pd.concat(
            [balance, pd.get_dummies(categorical, prefix=categorical_covariates, drop_first=False)],
            axis=1,
        )

    return balance


def coerce_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1).astype(int)


def make_age_quartile(age: pd.Series) -> pd.Series:
    numeric_age = pd.to_numeric(age, errors="coerce")
    return pd.qcut(numeric_age, q=4, duplicates="drop")
