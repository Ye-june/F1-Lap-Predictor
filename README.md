# F1 Lap Predictor

Web Link: [https://f1-lap-predicion-ml.streamlit.app/](https://f1-ml-lap-predictor.streamlit.app/)

A Streamlit machine-learning app that predicts Formula 1 lap times using real FastF1 session data. It also draws an accurate circuit map from FastF1 position telemetry and overlays selected driver markers on the track.

## What this version does

- Loads real F1 session data from FastF1 instead of only using random demo data.
- Keeps the demo-data path for offline testing and safe development.
- On Streamlit Cloud, uses a packed FastF1 deploy cache in offline mode so hosted races can still get real tyres/telemetry when live timing is blocked.
- Falls back to Jolpica for other completed Race sessions when FastF1 is unavailable (no telemetry/track map).
- Cleans and prepares lap-time, tyre, pit, driver, team, weather, and track-status features.
- Trains a tabular regression model with scikit-learn.
- Uses `GradientBoostingRegressor` by default because gradient-boosted trees are a strong industry-style baseline for tabular prediction problems. `HistGradientBoostingRegressor` is also available for larger datasets.
- Validates with a chronological split so later laps are tested against earlier laps.
- Shows MAE, RMSE, R², actual-vs-predicted plots, residual plots, and driver-level error summaries.
- Draws a FastF1 telemetry-based track map with corner labels and driver markers.
- Includes unit tests for features, model training, save/load, and track-map geometry.

## Tech stack

- Python
- Streamlit
- FastF1
- pandas / NumPy
- scikit-learn
- Plotly
- Joblib
- pytest

## Project structure

```text
F1-Lap-Predictor/
├── app.py
├── requirements.txt
├── pytest.ini
├── README.md
├── data/
│   └── fastf1_deploy_cache.zip
├── scripts/
│   └── preload_fastf1_cache.py
├── src/
│   └── f1lap/
│       ├── __init__.py
│       ├── data.py
│       ├── demo.py
│       ├── features.py
│       ├── model.py
│       ├── track.py
│       └── viz.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_model.py
    └── test_track.py
```

## How to run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## How to run tests

```bash
pytest
```

The tests use demo data, so they do not need the internet or FastF1 downloads.

## Streamlit Cloud + FastF1 only

Formula 1 blocks datacenter IPs from live timing, so Streamlit Cloud cannot download new FastF1 sessions live.

To serve real FastF1 data (tyres, weather, telemetry, track map) on the hosted app:

1. On a home/residential network, preload and pack a session:

```bash
python scripts/preload_fastf1_cache.py --year 2025 --event "Australian Grand Prix" --session Race --pack
```

2. Commit `data/fastf1_deploy_cache.zip` and redeploy.

3. On Streamlit Cloud, open that same season/event/session with **Force re-download** unchecked.

The app extracts the zip into `.fastf1_cache/` and enables FastF1 offline mode on Streamlit Cloud so it reads the packed cache instead of contacting livetiming.formula1.com.

## Notes

FastF1 data downloads can be slow the first time. The app creates a `.fastf1_cache/` folder so repeated local loads are much faster and less likely to hit rate limits.
