# F1 Lap Predictor

Web Link: [https://f1-lap-predicion-ml.streamlit.app/](https://f1-ml-lap-predictor.streamlit.app/)

A Streamlit machine-learning app that predicts Formula 1 lap times using real FastF1 session data. It also draws an accurate circuit map from FastF1 position telemetry and overlays selected driver markers on the track.

## What this version does

- Loads real F1 session data from FastF1 instead of only using random demo data.
- Keeps the demo-data path for offline testing and safe development.
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

## Notes

FastF1 data downloads can be slow the first time. The app creates a `.fastf1_cache/` folder so repeated loads are much faster and less likely to hit rate limits.
