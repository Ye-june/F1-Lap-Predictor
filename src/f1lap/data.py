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
    """Create and enable the local FastF1 cache directory."""

    fastf1 = _fastf1()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(
        str(cache_path),
        force_renew=force_renew,
    )

    return cache_path


def load_event_schedule(year: int) -> pd.DataFrame:
    """Return the FastF1 event schedule for one season."""

    fastf1 = _fastf1()
    schedule = fastf1.get_event_schedule(
        int(year),
        include_testing=False,
    )

    return pd.DataFrame(schedule)


def _get_session(year: int, event: str | int, session_name: str, backend: str | None):
    """Create a FastF1 session, falling back for older FastF1 versions if needed."""

    fastf1 = _fastf1()

    if backend is None:
        return fastf1.get_session(
            int(year),
            event,
            session_name,
        )

    try:
        return fastf1.get_session(
            int(year),
            event,
            session_name,
            backend=backend,
        )
    except TypeError:
        # Some FastF1 versions may not support the backend keyword.
        return fastf1.get_session(
            int(year),
            event,
            session_name,
        )


def _assert_laps_are_loaded(session, *, backend_label: str) -> None:
    """Fail clearly if FastF1 created a session but did not load lap timing."""

    try:
        laps = session.laps
    except Exception as exc:
        raise ValueError(
            f"{backend_label}: FastF1 created the session, but lap timing was not loaded."
        ) from exc

    if laps is None or len(laps) == 0:
        raise ValueError(
            f"{backend_label}: FastF1 loaded the session, but returned zero laps."
        )

    if "LapTime" not in laps.columns:
        raise ValueError(
            f"{backend_label}: FastF1 returned laps, but no LapTime column."
        )

    if laps["LapTime"].notna().sum() == 0:
        raise ValueError(
            f"{backend_label}: FastF1 returned laps, but every LapTime is empty."
        )


def _load_one_attempt(
    *,
    year: int,
    event: str | int,
    session_name: str,
    backend: str | None,
    telemetry: bool,
    weather: bool,
):
    backend_label = backend or "default"

    session = _get_session(
        year,
        event,
        session_name,
        backend,
    )

    session.load(
        laps=True,
        telemetry=telemetry,
        weather=weather,
        messages=False,
    )

    _assert_laps_are_loaded(
        session,
        backend_label=backend_label,
    )

    # Store this so the app/dataframe can explain how the data was loaded.
    session._f1lap_backend = backend_label
    session._f1lap_telemetry_loaded = telemetry

    return session


def _load_attempt_plan(
    *,
    include_telemetry: bool,
    session_name: str,
) -> list[tuple[str | None, bool, bool]]:
    """Return FastF1 loading attempts in safest order.

    Tuple format:
        (backend, telemetry, weather)
    """

    attempts: list[tuple[str | None, bool, bool]] = []

    # First try normal FastF1 behavior.
    attempts.append((None, include_telemetry, True))

    # If telemetry was requested, retry lap-only before switching backend.
    if include_telemetry:
        attempts.append((None, False, True))

    # Explicit f1timing attempt. This often behaves differently than default fallback.
    attempts.append(("f1timing", False, True))

    # Ergast is mainly useful for race lap timing fallback. It does not provide
    # modern telemetry, tyre, and local timing detail like the official F1 timing API.
    if session_name.lower() in {"race", "r", "sprint", "s"}:
        attempts.append(("ergast", False, False))

    # Remove duplicates while preserving order.
    clean_attempts: list[tuple[str | None, bool, bool]] = []
    seen: set[tuple[str | None, bool, bool]] = set()

    for attempt in attempts:
        if attempt not in seen:
            clean_attempts.append(attempt)
            seen.add(attempt)

    return clean_attempts


def load_fastf1_session(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
    force_renew: bool = False,
):
    """Load a FastF1 session object with backend fallbacks.

    FastF1's normal/live-timing backend gives the best data. If that fails to
    expose lap timing, this tries f1timing explicitly and then Ergast for race
    sessions so the model can still run.
    """

    enable_fastf1_cache(
        cache_dir,
        force_renew=force_renew,
    )

    errors: list[str] = []

    for backend, telemetry, weather in _load_attempt_plan(
        include_telemetry=include_telemetry,
        session_name=session_name,
    ):
        backend_label = backend or "default"

        try:
            return _load_one_attempt(
                year=year,
                event=event,
                session_name=session_name,
                backend=backend,
                telemetry=telemetry,
                weather=weather,
            )
        except Exception as exc:
            errors.append(
                f"{backend_label}, telemetry={telemetry}, weather={weather}: {exc}"
            )

    raise ValueError(
        "FastF1 could not load usable lap data for this session. "
        "This usually means the F1 live timing endpoints failed from the deployed environment "
        "or the selected session has incomplete public timing data. "
        "Try a different completed race, enable force re-download, or use Demo Data. "
        f"Details: {' | '.join(errors)}"
    )


def session_laps_to_frame(session) -> pd.DataFrame:
    """Turn a loaded FastF1 session into a normal pandas DataFrame for modeling."""

    try:
        laps = pd.DataFrame(session.laps.copy())
    except Exception as exc:
        raise ValueError(
            "The FastF1 session exists, but lap timing is unavailable. "
            "Reload the session or choose a completed session with timing data."
        ) from exc

    if laps.empty:
        raise ValueError("FastF1 returned no laps for this session.")

    if "LapTime" not in laps.columns or laps["LapTime"].notna().sum() == 0:
        raise ValueError(
            "FastF1 returned lap rows, but no usable LapTime values."
        )

    laps["Year"] = (
        int(session.event["EventDate"].year)
        if "EventDate" in session.event
        else None
    )

    laps["EventName"] = str(
        session.event.get(
            "EventName",
            session.event.get("Name", "Unknown Grand Prix"),
        )
    )

    laps["SessionName"] = str(
        getattr(session, "name", "Unknown Session")
    )

    laps["DataBackend"] = str(
        getattr(session, "_f1lap_backend", "unknown")
    )

    laps["TelemetryLoaded"] = bool(
        getattr(session, "_f1lap_telemetry_loaded", False)
    )

    try:
        weather = session.laps.get_weather_data().reset_index(drop=True)
        weather = weather[
            [column for column in weather.columns if column not in laps.columns]
        ]
        laps = pd.concat(
            [laps.reset_index(drop=True), weather],
            axis=1,
        )
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
    """Load a session and return ``(session, laps_dataframe)``."""

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
    """Return clean event names for a sidebar dropdown."""

    if "EventName" not in schedule.columns:
        return []

    return [
        str(name)
        for name in schedule["EventName"].dropna().tolist()
    ]