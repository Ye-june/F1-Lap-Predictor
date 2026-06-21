import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib

from data_loader import race_load

race = race_load(2023, "Bahrain")

features = [
    "Driver", "Team", "LapNumber", "Stint",
    "Compound", "TyreLife", "TrackStatus"
]

df = race[features + ["LapTimeSeconds"]].dropna()
df = pd.get_dummies(df, columns=["Driver", "Team", "Compound", "TrackStatus"])

X = df.drop(columns=["LapTimeSeconds"])
y = df["LapTimeSeconds"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("R2:", r2_score(y_test, preds))

models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "lap_time_model.pkl")
joblib.dump(X.columns.tolist(), models_dir / "model_features.pkl")