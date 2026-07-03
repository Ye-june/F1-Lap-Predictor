from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1lap.data import available_events, load_event_schedule, load_fastf1_laps
from f1lap.demo import make_demo_laps
from f1lap.features import prepare_lap_frame
from f1lap.model import predict_laps, train_model
from f1lap.track import build_track_map
from f1lap.viz import driver_error_bar, prediction_trace, residual_chart


st.set_page_config(
    page_title="F1 Lap Predictor",
    page_icon="🏎️",
    layout="wide",
)

def _theme_color(option_name: str, fallback: str | None = None) -> str | None:
    try:
        return st.get_option(option_name) or fallback
    except Exception:
        return fallback

@st.cache_data(show_spinner="Loading F1 calendar...")
def cached_schedule(year: int):
    return load_event_schedule(year)


@st.cache_resource(show_spinner="Downloading FastF1 session data...")
def cached_fastf1_session(year: int, event: str, session_name: str, include_telemetry: bool):
    return load_fastf1_laps(
        year,
        event,
        session_name,
        cache_dir=ROOT / ".fastf1_cache",
        include_telemetry=include_telemetry,
    )


st.title("🏎️ F1 Lap Predictor")
st.write(
    "Predict F1 lap times with FastF1 data, compare model errors, and view the selected drivers on an accurate circuit map."
)

st.sidebar.header("Data")
data_source = st.sidebar.radio("Choose data source", ["FastF1", "Demo Data"])

raw_laps = None
session = None
session_label = "Demo Grand Prix"

if data_source == "Demo Data":
    n_laps = st.sidebar.slider("Number of demo laps", min_value=20, max_value=70, value=35)
    raw_laps = make_demo_laps(n_laps=n_laps)
    st.sidebar.info("Demo data is only for offline development. Use FastF1 for the real project.")
else:
    current_year = datetime.now().year
    year = st.sidebar.selectbox("Season", list(range(current_year, 2017, -1)), index=1 if current_year >= 2026 else 0)

    try:
        schedule = cached_schedule(year)
        events = available_events(schedule)
    except Exception as exc:
        st.error(f"Could not load the FastF1 calendar: {exc}")
        events = []

    if events:
        event = st.sidebar.selectbox("Grand Prix", events)
        session_name = st.sidebar.selectbox("Session", ["Race", "Qualifying", "Sprint", "Practice 3", "Practice 2", "Practice 1"])
        include_telemetry = st.sidebar.checkbox("Load telemetry for track map", value=True)
        quick_laps_only = st.sidebar.checkbox("Use only quick laps (107% rule)", value=False)

        if st.sidebar.button("Load FastF1 session", type="primary"):
            try:
                session, raw_laps = cached_fastf1_session(year, event, session_name, include_telemetry)
                session_label = f"{year} {event} {session_name}"
                st.session_state["session"] = session
                st.session_state["raw_laps"] = raw_laps
                st.session_state["session_label"] = session_label
                st.session_state["quick_laps_only"] = quick_laps_only
            except Exception as exc:
                st.error(f"Could not load FastF1 data: {exc}")
        else:
            session = st.session_state.get("session")
            raw_laps = st.session_state.get("raw_laps")
            session_label = st.session_state.get("session_label", session_label)
            quick_laps_only = st.session_state.get("quick_laps_only", False)

if raw_laps is None:
    st.info("Choose a session in the sidebar and load the data to start.")
    st.stop()

try:
    prepared_laps = prepare_lap_frame(raw_laps, quick_laps_only=st.session_state.get("quick_laps_only", False))
except Exception as exc:
    st.error(f"Could not prepare lap features: {exc}")
    st.stop()

st.subheader(session_label)
st.caption(f"Usable laps after cleaning: {len(prepared_laps):,}")

model_name = st.sidebar.selectbox(
    "Model",
    ["gradient_boosting", "hist_gradient_boosting", "random_forest"],
    help="Gradient boosting is the recommended model for tabular lap-time prediction.",
)
test_size = st.sidebar.slider("Validation size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)

try:
    bundle = train_model(prepared_laps, model_name=model_name, test_size=test_size)
    predictions = predict_laps(bundle, prepared_laps)
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

st.subheader("Model metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAE", f"{bundle.metrics['mae_sec']:.3f}s")
col2.metric("RMSE", f"{bundle.metrics['rmse_sec']:.3f}s")
col3.metric("R²", f"{bundle.metrics['r2']:.3f}")
col4.metric("Train rows", f"{bundle.train_rows:,}")
col5.metric("Test rows", f"{bundle.test_rows:,}")

st.caption("Validation uses a chronological split, not a random split, so later laps are tested against earlier laps.")

if session is not None:
    st.subheader("Track map")
    map_drivers = st.multiselect(
        "Drivers on map",
        sorted(prepared_laps["driver"].unique().tolist()),
        default=sorted(prepared_laps["driver"].unique().tolist())[:5],
    )
    lap_min = int(prepared_laps["lap_number"].min())
    lap_max = int(prepared_laps["lap_number"].max())
    map_lap = st.slider("Map lap", min_value=lap_min, max_value=lap_max, value=lap_min)
    lap_progress = st.slider("Lap progress", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

    try:
        st.plotly_chart(
            build_track_map(
                session,
                drivers=map_drivers,
                lap_number=map_lap,
                lap_progress=lap_progress,
                show_corners=True,
                background_color=_theme_color("theme.backgroundColor", "rgba(0,0,0,0)"),
                text_color=_theme_color("theme.textColor"),
                driver_marker_size=7,
            ),
            use_container_width=True,
        )
        st.caption("Hover over the small dots to see driver, team, lap, tyre, and lap-time info.")
    except Exception as exc:
        st.warning(
            "Track map needs telemetry. Re-load the session with 'Load telemetry for track map' checked. "
            f"Details: {exc}"
        )

drivers = ["All"] + sorted(predictions["driver"].unique().tolist())
selected_driver = st.selectbox("Driver", drivers)

chart_col, error_col = st.columns([2, 1])
with chart_col:
    st.plotly_chart(prediction_trace(predictions, selected_driver), use_container_width=True)
with error_col:
    st.plotly_chart(driver_error_bar(predictions), use_container_width=True)

st.plotly_chart(residual_chart(predictions, selected_driver), use_container_width=True)

st.subheader("Prediction table")
display_columns = [
    "driver",
    "team",
    "lap_number",
    "compound",
    "tyre_life",
    "lap_time_sec",
    "predicted_lap_time_sec",
    "prediction_error_sec",
    "absolute_error_sec",
]
st.dataframe(predictions[display_columns].round(3), use_container_width=True)
