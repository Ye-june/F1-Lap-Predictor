from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "lap_time_sec"

NUMERIC_FEATURES = [
    "lap_number",
    "stint",
    "tyre_life",
    "prev_lap_time_sec",
    "rolling_median_3_sec",
    "is_pit_lap",
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
    "rainfall",
]

CATEGORICAL_FEATURES = [
    "driver",
    "team",
    "compound",
    "track_status",
    "event_name",
    "session_name",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert timedelta-like values to seconds."""

    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def prepare_lap_frame(raw_laps: pd.DataFrame, *, quick_laps_only: bool = False) -> pd.DataFrame:
    """Convert raw FastF1-style laps into model-ready features.

    FastF1 returns extended pandas DataFrames with columns such as ``LapTime``,
    ``Driver``, ``Team``, ``Compound`` and weather columns. The Streamlit demo
    data intentionally mirrors those names, so this function works for both.
    """

    if raw_laps.empty:
        raise ValueError("No lap data was provided.")

    df = raw_laps.copy()

    rename_map = {
        "Year": "year",
        "EventName": "event_name",
        "SessionName": "session_name",
        "LapNumber": "lap_number",
        "Stint": "stint",
        "TyreLife": "tyre_life",
        "Driver": "driver",
        "Team": "team",
        "Compound": "compound",
        "TrackStatus": "track_status",
        "AirTemp": "air_temp",
        "TrackTemp": "track_temp",
        "Humidity": "humidity",
        "WindSpeed": "wind_speed",
        "Rainfall": "rainfall",
    }
    df = df.rename(columns=rename_map)

    lap_time_column = _first_existing(df, ["LapTime", TARGET_COLUMN])
    if lap_time_column is None:
        raise ValueError("Missing LapTime or lap_time_sec column.")

    df[TARGET_COLUMN] = _to_seconds(df[lap_time_column])

    if "PitInTime" in df.columns:
        df["is_pit_lap"] = df["PitInTime"].notna().astype(int)
    elif "PitOutTime" in df.columns:
        df["is_pit_lap"] = df["PitOutTime"].notna().astype(int)
    else:
        df["is_pit_lap"] = 0

    defaults = {
        "year": np.nan,
        "event_name": "Unknown Grand Prix",
        "session_name": "Unknown Session",
        "stint": 1,
        "tyre_life": np.nan,
        "driver": "Unknown",
        "team": "Unknown",
        "compound": "Unknown",
        "track_status": "1",
        "air_temp": np.nan,
        "track_temp": np.nan,
        "humidity": np.nan,
        "wind_speed": np.nan,
        "rainfall": 0,
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    for column in ["lap_number", "stint", "tyre_life", TARGET_COLUMN, "air_temp", "track_temp", "humidity", "wind_speed"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["rainfall"] = df["rainfall"].fillna(False).astype(bool).astype(int)

    # Remove invalid laps. Real FastF1 laps can include deleted, in/out, or empty rows.
    df = df.dropna(subset=["lap_number", TARGET_COLUMN])
    df = df[df[TARGET_COLUMN].between(30, 300)]

    if "IsAccurate" in df.columns:
        accurate_mask = df["IsAccurate"].fillna(True).astype(bool)
        df = df[accurate_mask]

    if quick_laps_only and not df.empty:
        fastest = df[TARGET_COLUMN].min()
        df = df[df[TARGET_COLUMN] <= fastest * 1.07]

    df = df.sort_values(["event_name", "session_name", "driver", "lap_number"]).reset_index(drop=True)

    group_cols = ["event_name", "session_name", "driver"]
    driver_groups = df.groupby(group_cols, dropna=False)[TARGET_COLUMN]

    df["prev_lap_time_sec"] = driver_groups.shift(1)
    df["rolling_median_3_sec"] = driver_groups.transform(
        lambda laps: laps.shift(1).rolling(3, min_periods=1).median()
    )

    global_median = df[TARGET_COLUMN].median()
    df["prev_lap_time_sec"] = df["prev_lap_time_sec"].fillna(global_median)
    df["rolling_median_3_sec"] = df["rolling_median_3_sec"].fillna(global_median)

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype("string").fillna("Unknown").astype(str)

    return df.reset_index(drop=True)
