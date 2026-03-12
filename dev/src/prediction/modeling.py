from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_experiment_features(
    experiment_cfg: dict[str, Any],
    numeric_candidates: list[str],
    categorical_candidates: list[str],
) -> tuple[list[str], list[str], list[str]]:
    requested = list(experiment_cfg.get("features", []))
    requested_set = set(requested)

    numeric_features = [f for f in numeric_candidates if f in requested_set]
    categorical_features = [f for f in categorical_candidates if f in requested_set]
    missing_features = [f for f in requested if f not in set(numeric_features + categorical_features)]
    return numeric_features, categorical_features, missing_features


def build_estimator(
    model_type: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int,
) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    preprocessor = ColumnTransformer(transformers=transformers)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
