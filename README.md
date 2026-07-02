# F1 Lap Predictor

RUN THE WEBSITE: https://f1-lap-predicion-ml.streamlit.app/

A simple Streamlit app that predicts Formula 1 lap times and shows how close the predictions are to the actual lap times.

## What was implemented

- Streamlit web app interface
- Demo F1 lap data generator
- Lap time prediction model
- Feature preparation for race data
- Driver, team, tyre, stint, pit lap, and track status features
- Machine learning pipeline using scikit-learn
- Model performance metrics:
  - MAE
  - RMSE
  - R² score
- Driver filter for viewing results
- Actual vs predicted lap time chart
- Prediction error chart
- Prediction results table
- Basic model save/load support with joblib

## Tech used

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Joblib

## How to run

Install the dependencies:

```bash
pip install -r requirements.txt
