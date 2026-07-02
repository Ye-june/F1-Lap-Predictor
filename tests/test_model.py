from pathlib import Path

from f1lap.demo import make_demo_laps
from f1lap.features import prepare_lap_frame
from f1lap.model import chronological_split, load_model, predict_laps, save_model, train_model


def test_chronological_split_preserves_all_rows():
    laps = prepare_lap_frame(make_demo_laps(n_laps=30))
    train_df, test_df = chronological_split(laps, test_size=0.25)

    assert len(train_df) + len(test_df) == len(laps)
    assert len(train_df) > len(test_df)


def test_train_predict_and_save_roundtrip(tmp_path: Path):
    laps = prepare_lap_frame(make_demo_laps(n_laps=40))
    bundle = train_model(laps, model_name="gradient_boosting", test_size=0.2)
    predictions = predict_laps(bundle, laps)

    assert bundle.metrics["mae_sec"] >= 0
    assert "predicted_lap_time_sec" in predictions.columns
    assert predictions["absolute_error_sec"].notna().all()

    model_path = tmp_path / "model.joblib"
    save_model(bundle, model_path)
    loaded = load_model(model_path)
    loaded_predictions = predict_laps(loaded, laps.head(5))

    assert loaded.model_name == "gradient_boosting"
    assert len(loaded_predictions) == 5
