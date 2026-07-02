from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]
    target_column: str
    train_rows: int
    test_rows: int
    model_name: str


def build_model(model_name: str = "gradient_boosting") -> Pipeline:
    """Build an industry-style tabular regression pipeline.

    ``GradientBoostingRegressor`` is the default because gradient-boosted
    trees are a strong baseline for structured/tabular data like lap timing.
    HistGradientBoosting and random forest are kept as comparison models.
    """

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    if model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=0.02,
            random_state=42,
        )
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            n_estimators=250,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    return {
        "mae_sec": float(mae),
        "rmse_sec": float(rmse),
        "r2": float(r2),
    }


def chronological_split(laps: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split laps so later laps are tested on earlier training data.

    A random split makes the model look better than it is because lap data is a
    time series. This split is closer to how you would predict future race laps.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    ordered = laps.sort_values(["event_name", "session_name", "lap_number", "driver"]).reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * (1 - test_size)))
    split_idx = min(split_idx, len(ordered) - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def train_model(
    laps: pd.DataFrame,
    *,
    model_name: str = "gradient_boosting",
    test_size: float = 0.2,
) -> ModelBundle:
    if len(laps) < 20:
        raise ValueError("Need at least 20 usable laps to train a model.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in laps.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    train_df, test_df = chronological_split(laps, test_size=test_size)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    pipeline = build_model(model_name=model_name)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions)

    return ModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        train_rows=len(train_df),
        test_rows=len(test_df),
        model_name=model_name,
    )


def predict_laps(bundle: ModelBundle, laps: pd.DataFrame) -> pd.DataFrame:
    output = laps.copy()

    output["predicted_lap_time_sec"] = bundle.pipeline.predict(output[bundle.feature_columns])
    output["prediction_error_sec"] = output["predicted_lap_time_sec"] - output[bundle.target_column]
    output["absolute_error_sec"] = output["prediction_error_sec"].abs()

    return output


def save_model(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path) -> ModelBundle:
    return joblib.load(path)
