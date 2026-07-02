from __future__ import annotations

import numpy as np
import pandas as pd


DRIVERS = [
    ("VER", "Red Bull Racing", 0.00),
    ("NOR", "McLaren", 0.25),
    ("LEC", "Ferrari", 0.38),
    ("HAM", "Ferrari", 0.55),
    ("RUS", "Mercedes", 0.62),
    ("ALO", "Aston Martin", 0.95),
]


def make_demo_laps(n_laps: int = 35, seed: int = 7) -> pd.DataFrame:
    """Create fake but realistic F1 lap data for demos and tests."""

    rng = np.random.default_rng(seed)
    rows = []

    for driver, team, pace_delta in DRIVERS:
        pit_lap = int(rng.integers(12, n_laps - 5))
        stint = 1
        compound = "MEDIUM"
        tyre_life = 0

        for lap in range(1, n_laps + 1):
            if lap == pit_lap + 1:
                stint += 1
                compound = "HARD"
                tyre_life = 0
            else:
                tyre_life += 1

            is_pit_lap = lap == pit_lap

            fuel_effect = -0.045 * lap
            tyre_degradation = 0.035 * tyre_life
            compound_effect = 0.25 if compound == "HARD" else 0.0
            pit_effect = 22.0 if is_pit_lap else 0.0
            random_noise = rng.normal(0, 0.25)

            lap_time = (
                91.5
                + pace_delta
                + fuel_effect
                + tyre_degradation
                + compound_effect
                + pit_effect
                + random_noise
            )

            rows.append(
                {
                    "Year": 2025,
                    "EventName": "Demo Grand Prix",
                    "SessionName": "Race",
                    "Driver": driver,
                    "Team": team,
                    "LapNumber": lap,
                    "Stint": stint,
                    "Compound": compound,
                    "TyreLife": tyre_life,
                    "TrackStatus": "1",
                    "LapTime": pd.to_timedelta(lap_time, unit="s"),
                    "PitInTime": pd.to_timedelta(lap * 95, unit="s") if is_pit_lap else pd.NaT,
                    "PitOutTime": pd.NaT,
                    "IsAccurate": True,
                }
            )

    return pd.DataFrame(rows)