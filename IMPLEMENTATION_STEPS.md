# FastF1 Upgrade Implementation Steps

This folder contains a drop-in upgrade for `Ye-june/F1-Lap-Predictor`.

## 1. Create a branch

```bash
git checkout main
git pull
git checkout -b fastf1-real-data-track-map
```

## 2. Replace these existing files

Copy these files from this upgrade folder into your repo root:

```text
app.py
README.md
requirements.txt
src/f1lap/__init__.py
src/f1lap/demo.py
src/f1lap/features.py
src/f1lap/model.py
src/f1lap/viz.py
```

## 3. Create these new files

```text
.gitignore
pytest.ini
src/f1lap/data.py
src/f1lap/track.py
tests/test_features.py
tests/test_model.py
tests/test_track.py
```

## 4. Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
pytest
streamlit run app.py
```

## 5. How the app should behave

1. Open the Streamlit app.
2. Leave data source on `FastF1`.
3. Pick a season, Grand Prix, and session.
4. Keep `Load telemetry for track map` checked.
5. Click `Load FastF1 session`.
6. Review MAE/RMSE/R², prediction plots, residuals, driver error summary, prediction table, and track map.

## 6. Why this implementation is better

- It keeps the original demo-data flow so development and tests work offline.
- It adds a real FastF1 data loader with caching.
- It cleans FastF1 lap data into stable modeling columns.
- It uses gradient-boosted decision trees as the default tabular ML model.
- It validates chronologically instead of randomly, which is more honest for lap prediction.
- It draws the circuit using FastF1 position telemetry and circuit corner metadata.
- It includes tests that run without internet access.
