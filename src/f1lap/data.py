from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fastf1():
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError(
            "FastF1 is not installed. Run `pip install -r requirements.txt` first."
        ) from exc
    return fastf1


def enable_fastf1_cache(
    cache_dir: str | Path = ".fastf1_cache",
    *,
    force_renew: bool = False,
) -> Path:
    fastf1 = _fastf1()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path), force_renew=force_renew)
    return cache_path


def load_event_schedule(year: int) -> pd.DataFrame:
    fastf1 = _fastf1()
    schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    return pd.DataFrame(schedule)


def _assert_laps_are_loaded(session) -> None:
    try:
        laps = session.laps
    except Exception as exc:
        raise ValueError(
            "FastF1 created the session, but lap timing was not loaded. "
            "Try an older completed event, uncheck telemetry, or force a cache re-download."
        ) from exc

    if laps is None or len(laps) == 0:
        raise ValueError(
            "FastF1 loaded the session, but returned zero laps. "
            "Try an older completed event or force a cache re-download."
        )


def load_fastf1_session(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
    force_renew: bool = False,
):
    enable_fastf1_cache(cache_dir, force_renew=force_renew)
    fastf1 = _fastf1()

    telemetry_attempts = [include_telemetry]
    if include_telemetry:
        telemetry_attempts.append(False)

    errors: list[str] = []

    for telemetry_enabled in telemetry_attempts:
        session = fastf1.get_session(int(year), event, session_name)

        try:
            session.load(
                laps=True,
                telemetry=telemetry_enabled,
                weather=True,
                messages=False,
            )
            _assert_laps_are_loaded(session)
            return session
        except Exception as exc:
            errors.append(f"telemetry={telemetry_enabled}: {exc}")

    raise ValueError(
        "FastF1 could not load usable lap data for this session. "
        "Try a completed older race, uncheck telemetry, enable force re-download, "
        "or clear the Streamlit cache. "
        f"Details: {' | '.join(errors)}"
    )


def session_laps_to_frame(session) -> pd.DataFrame:
    try:
        laps = pd.DataFrame(session.laps.copy())
    except Exception as exc:
        raise ValueError(
            "The FastF1 session exists, but lap timing is unavailable. "
            "Reload the session or choose a completed session with timing data."
        ) from exc

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
        pass

    return laps.reset_index(drop=True)


def load_fastf1_laps(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
    force_renew: bool = False,
) -> tuple[object, pd.DataFrame]:
    session = load_fastf1_session(
        year,
        event,
        session_name,
        cache_dir=cache_dir,
        include_telemetry=include_telemetry,
        force_renew=force_renew,
    )
    return session, session_laps_to_frame(session)


def available_events(schedule: pd.DataFrame) -> list[str]:
    if "EventName" not in schedule.columns:
        return []

    return [str(name) for name in schedule["EventName"].dropna().tolist()]