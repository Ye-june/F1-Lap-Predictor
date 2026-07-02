from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "lap_time_sec"

FEATURE_COLUMNS = [
    "lap_number",
    "stint",
    "tyre_life",
    "prev_lap_time_sec",
    "rolling_median_3_sec",
    "is_pit_lap",
    "driver",
    "team",
    "compound",
    "track_status",
]


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert timedelta values to seconds."""

    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def prepare_lap_frame(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """Convert raw FastF1-style laps into model-ready features."""

    if raw_laps.empty:
        raise ValueError("No lap data was provided.")

    df = raw_laps.copy()

    rename_map = {
        "LapNumber": "lap_number",
        "Stint": "stint",
        "TyreLife": "tyre_life",
        "Driver": "driver",
        "Team": "team",
        "Compound": "compound",
        "TrackStatus": "track_status",
    }

    df = df.rename(columns=rename_map)

    if "LapTime" in df.columns:
        df[TARGET_COLUMN] = _to_seconds(df["LapTime"])

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Missing LapTime or lap_time_sec column.")

    if "PitInTime" in df.columns:
        df["is_pit_lap"] = df["PitInTime"].notna().astype(int)
    else:
        df["is_pit_lap"] = 0

    defaults = {
        "stint": 1,
        "tyre_life": np.nan,
        "driver": "Unknown",
        "team": "Unknown",
        "compound": "Unknown",
        "track_status": "1",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce")
    df["stint"] = pd.to_numeric(df["stint"], errors="coerce")
    df["tyre_life"] = pd.to_numeric(df["tyre_life"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna(subset=["lap_number", TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].between(30, 300)]

    df = df.sort_values(["driver", "lap_number"]).reset_index(drop=True)

    driver_groups = df.groupby("driver")[TARGET_COLUMN]

    df["prev_lap_time_sec"] = driver_groups.shift(1)
    df["rolling_median_3_sec"] = driver_groups.transform(
        lambda laps: laps.shift(1).rolling(3, min_periods=1).median()
    )

    global_median = df[TARGET_COLUMN].median()

    df["prev_lap_time_sec"] = df["prev_lap_time_sec"].fillna(global_median)
    df["rolling_median_3_sec"] = df["rolling_median_3_sec"].fillna(global_median)

    for column in ["driver", "team", "compound", "track_status"]:
        df[column] = df[column].astype(str).fillna("Unknown")

    return df.reset_index(drop=True)