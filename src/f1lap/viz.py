from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .features import TARGET_COLUMN


def _filter_driver(df: pd.DataFrame, driver: str | None) -> pd.DataFrame:
    if driver and driver != "All":
        return df[df["driver"] == driver].copy()
    return df.copy()


def prediction_trace(predictions: pd.DataFrame, driver: str | None = None) -> go.Figure:
    df = _filter_driver(predictions, driver)
    fig = go.Figure()

    for selected_driver, driver_df in df.groupby("driver"):
        driver_df = driver_df.sort_values("lap_number")
        fig.add_trace(
            go.Scatter(
                x=driver_df["lap_number"],
                y=driver_df[TARGET_COLUMN],
                mode="lines+markers",
                name=f"{selected_driver} actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=driver_df["lap_number"],
                y=driver_df["predicted_lap_time_sec"],
                mode="lines+markers",
                name=f"{selected_driver} predicted",
                line={"dash": "dash"},
            )
        )

    fig.update_layout(
        title="Actual vs predicted lap times",
        xaxis_title="Lap number",
        yaxis_title="Lap time (seconds)",
        hovermode="x unified",
    )
    return fig


def residual_chart(predictions: pd.DataFrame, driver: str | None = None) -> go.Figure:
    df = _filter_driver(predictions, driver)
    fig = px.scatter(
        df,
        x="lap_number",
        y="prediction_error_sec",
        color="driver",
        hover_data=["compound", "tyre_life", "team"],
        title="Prediction error by lap",
        labels={
            "lap_number": "Lap number",
            "prediction_error_sec": "Prediction error (seconds)",
        },
    )
    fig.add_hline(y=0, line_dash="dash")
    return fig


def driver_error_bar(predictions: pd.DataFrame) -> go.Figure:
    summary = (
        predictions.groupby("driver", as_index=False)["absolute_error_sec"]
        .mean()
        .sort_values("absolute_error_sec")
    )
    return px.bar(
        summary,
        x="driver",
        y="absolute_error_sec",
        title="Mean absolute error by driver",
        labels={"absolute_error_sec": "Mean absolute error (sec)", "driver": "Driver"},
    )
