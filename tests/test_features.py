import pandas as pd
import pytest

from f1lap.demo import make_demo_laps
from f1lap.features import FEATURE_COLUMNS, TARGET_COLUMN, prepare_lap_frame


def test_prepare_lap_frame_outputs_required_columns():
    raw = make_demo_laps(n_laps=25)
    prepared = prepare_lap_frame(raw)

    assert not prepared.empty
    assert TARGET_COLUMN in prepared.columns
    for column in FEATURE_COLUMNS:
        assert column in prepared.columns

    assert prepared[TARGET_COLUMN].between(30, 300).all()
    assert prepared["prev_lap_time_sec"].notna().all()
    assert prepared["rolling_median_3_sec"].notna().all()


def test_prepare_lap_frame_rejects_empty_input():
    with pytest.raises(ValueError, match="No lap data"):
        prepare_lap_frame(pd.DataFrame())


def test_quick_laps_filter_removes_big_outliers():
    raw = make_demo_laps(n_laps=35)
    all_laps = prepare_lap_frame(raw, quick_laps_only=False)
    quick_laps = prepare_lap_frame(raw, quick_laps_only=True)

    assert len(quick_laps) < len(all_laps)
    assert quick_laps["lap_time_sec"].max() <= quick_laps["lap_time_sec"].min() * 1.07
