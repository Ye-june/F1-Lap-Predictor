from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1lap.demo import make_demo_laps
from f1lap.features import prepare_lap_frame
from f1lap.model import predict_laps, train_model
from f1lap.viz import prediction_trace, residual_chart


st.set_page_config(
    page_title="F1 Lap Predictor",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Lap Predictor")
st.write("Predict Formula 1 lap times and visualize actual vs predicted performance.")

st.sidebar.header("Data")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Demo Data"],
)

if data_source == "Demo Data":
    n_laps = st.sidebar.slider(
        "Number of demo laps",
        min_value=20,
        max_value=60,
        value=35,
    )

    raw_laps = make_demo_laps(n_laps=n_laps)

prepared_laps = prepare_lap_frame(raw_laps)
bundle = train_model(prepared_laps)
predictions = predict_laps(bundle, prepared_laps)

st.subheader("Model Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{bundle.metrics['mae_sec']:.3f}s")
col2.metric("RMSE", f"{bundle.metrics['rmse_sec']:.3f}s")
col3.metric("R²", f"{bundle.metrics['r2']:.3f}")

drivers = ["All"] + sorted(predictions["driver"].unique().tolist())

selected_driver = st.selectbox(
    "Driver",
    drivers,
)

st.plotly_chart(
    prediction_trace(predictions, selected_driver),
    use_container_width=True,
)

st.plotly_chart(
    residual_chart(predictions, selected_driver),
    use_container_width=True,
)

st.subheader("Prediction Table")

display_columns = [
    "driver",
    "lap_number",
    "compound",
    "tyre_life",
    "lap_time_sec",
    "predicted_lap_time_sec",
    "prediction_error_sec",
    "absolute_error_sec",
]

st.dataframe(
    predictions[display_columns].round(3),
    use_container_width=True,
)