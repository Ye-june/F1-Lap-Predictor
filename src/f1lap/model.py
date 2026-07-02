from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]
    target_column: str


def build_model() -> Pipeline:
    numeric_features = [
        "lap_number",
        "stint",
        "tyre_life",
        "prev_lap_time_sec",
        "rolling_median_3_sec",
        "is_pit_lap",
    ]

    categorical_features = [
        "driver",
        "team",
        "compound",
        "track_status",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae_sec": float(mae),
        "rmse_sec": float(rmse),
        "r2": float(r2),
    }


def train_model(laps: pd.DataFrame) -> ModelBundle:
    x = laps[FEATURE_COLUMNS]
    y = laps[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_model()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions)

    return ModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )


def predict_laps(bundle: ModelBundle, laps: pd.DataFrame) -> pd.DataFrame:
    output = laps.copy()

    output["predicted_lap_time_sec"] = bundle.pipeline.predict(
        output[bundle.feature_columns]
    )

    output["prediction_error_sec"] = (
        output["predicted_lap_time_sec"] - output[bundle.target_column]
    )

    output["absolute_error_sec"] = output["prediction_error_sec"].abs()

    return output


def save_model(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path) -> ModelBundle:
    return joblib.load(path)