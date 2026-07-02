from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .features import TARGET_COLUMN


def prediction_trace(predictions: pd.DataFrame, driver: str | None = None) -> go.Figure:
    df = predictions.copy()

    if driver and driver != "All":
        df = df[df["driver"] == driver]

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
        title="Actual vs Predicted Lap Times",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time Seconds",
        hovermode="x unified",
    )

    return fig


def residual_chart(predictions: pd.DataFrame, driver: str | None = None) -> go.Figure:
    df = predictions.copy()

    if driver and driver != "All":
        df = df[df["driver"] == driver]

    fig = go.Figure()

    for selected_driver, driver_df in df.groupby("driver"):
        fig.add_trace(
            go.Scatter(
                x=driver_df["lap_number"],
                y=driver_df["prediction_error_sec"],
                mode="markers",
                name=selected_driver,
            )
        )

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title="Prediction Error by Lap",
        xaxis_title="Lap Number",
        yaxis_title="Prediction Error Seconds",
    )

    return fig