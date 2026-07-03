from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _fastf1():
    try:
        import fastf1
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError(
            "FastF1 is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return fastf1


def enable_fastf1_cache(cache_dir: str | Path = ".fastf1_cache") -> Path:
    """Create and enable the local FastF1 cache directory."""

    fastf1 = _fastf1()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    return cache_path


def load_event_schedule(year: int) -> pd.DataFrame:
    """Return the FastF1 event schedule for one season."""

    fastf1 = _fastf1()
    schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    return pd.DataFrame(schedule)


def load_fastf1_session(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
):
    """Load a FastF1 session object.

    ``include_telemetry=True`` is needed for the track map because driver/circuit
    coordinates come from position telemetry.
    """

    enable_fastf1_cache(cache_dir)
    fastf1 = _fastf1()
    session = fastf1.get_session(int(year), event, session_name)
    session.load(laps=True, telemetry=include_telemetry, weather=True, messages=False)
    return session


def session_laps_to_frame(session) -> pd.DataFrame:
    """Turn a loaded FastF1 session into a normal pandas DataFrame for modeling."""

    laps = pd.DataFrame(session.laps.copy())
    if laps.empty:
        raise ValueError("FastF1 returned no laps for this session.")

    laps["Year"] = int(session.event["EventDate"].year) if "EventDate" in session.event else None
    laps["EventName"] = str(session.event.get("EventName", session.event.get("Name", "Unknown Grand Prix")))
    laps["SessionName"] = str(getattr(session, "name", "Unknown Session"))

    try:
        weather = session.laps.get_weather_data().reset_index(drop=True)
        weather = weather[[column for column in weather.columns if column not in laps.columns]]
        laps = pd.concat([laps.reset_index(drop=True), weather], axis=1)
    except Exception:
        # Weather is useful, but the predictor should still work if a session has
        # incomplete weather data or the API shape changes slightly.
        pass

    return laps.reset_index(drop=True)


def load_fastf1_laps(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
) -> tuple[object, pd.DataFrame]:
    """Load a session and return ``(session, laps_dataframe)``."""

    session = load_fastf1_session(
        year,
        event,
        session_name,
        cache_dir=cache_dir,
        include_telemetry=include_telemetry,
    )
    return session, session_laps_to_frame(session)


def available_events(schedule: pd.DataFrame) -> list[str]:
    """Return clean event names for a sidebar dropdown."""

    if "EventName" not in schedule.columns:
        return []

    return [str(name) for name in schedule["EventName"].dropna().tolist()]
