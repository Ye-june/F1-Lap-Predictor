from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd


RACE_SESSION_NAMES = {"race", "r"}
DEFAULT_DEPLOY_CACHE_ZIP = Path("data") / "fastf1_deploy_cache.zip"


def _fastf1():
    try:
        import fastf1
    except ImportError as exc:
        raise ImportError(
            "FastF1 is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    return fastf1


def is_streamlit_cloud() -> bool:
    """Detect Streamlit Community Cloud-style hosted runtimes."""

    if os.environ.get("STREAMLIT_SHARING_MODE") == "true":
        return True

    return Path("/mount/src").exists()


def deploy_cache_zip_path(cache_dir: str | Path) -> Path:
    """Return the packed FastF1 cache zip next to the project root."""

    cache_path = Path(cache_dir)
    root = cache_path.parent if cache_path.name == ".fastf1_cache" else Path.cwd()
    return root / DEFAULT_DEPLOY_CACHE_ZIP


def extract_deploy_cache(
    cache_dir: str | Path = ".fastf1_cache",
    *,
    zip_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Extract a shipped FastF1 cache zip into ``cache_dir`` when needed."""

    cache_path = Path(cache_dir)
    archive = Path(zip_path) if zip_path is not None else deploy_cache_zip_path(cache_path)

    if not archive.is_file():
        return None

    marker = cache_path / ".deploy_cache_ready"
    if marker.exists() and not overwrite:
        return cache_path

    cache_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(cache_path)

    marker.write_text(f"extracted from {archive.name}\n", encoding="utf-8")
    return cache_path


def enable_fastf1_cache(
    cache_dir: str | Path = ".fastf1_cache",
    *,
    force_renew: bool = False,
    offline_mode: bool | None = None,
    use_deploy_cache: bool = True,
) -> Path:
    """Create and enable the local FastF1 cache directory.

    On Streamlit Cloud, Formula 1 live-timing endpoints are blocked for
    datacenter IPs. Shipping ``data/fastf1_deploy_cache.zip`` and enabling
    FastF1 offline mode lets the hosted app serve real FastF1 sessions from
    that packed cache without contacting livetiming.formula1.com.
    """

    fastf1 = _fastf1()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if use_deploy_cache and not force_renew:
        extract_deploy_cache(cache_path)

    fastf1.Cache.enable_cache(
        str(cache_path),
        force_renew=force_renew,
    )

    if offline_mode is None:
        # Only force offline on Streamlit Cloud. Locally, keep live FastF1
        # available for sessions that are not in the packed deploy cache.
        offline_mode = (
            use_deploy_cache
            and not force_renew
            and is_streamlit_cloud()
            and deploy_cache_zip_path(cache_path).is_file()
        )

    if offline_mode:
        fastf1.Cache.offline_mode(True)

    return cache_path


def load_event_schedule(
    year: int,
    *,
    cache_dir: str | Path = ".fastf1_cache",
) -> pd.DataFrame:
    """Return an event schedule, falling back to Jolpica if needed."""

    enable_fastf1_cache(
        cache_dir,
        force_renew=False,
    )

    fastf1 = _fastf1()
    errors: list[str] = []

    for backend in (None, "ergast"):
        backend_label = backend or "default"

        try:
            kwargs = {
                "year": int(year),
                "include_testing": False,
            }
            if backend is not None:
                kwargs["backend"] = backend

            schedule = pd.DataFrame(fastf1.get_event_schedule(**kwargs))

            if schedule.empty:
                raise ValueError("the returned schedule was empty")
            if "EventName" not in schedule.columns:
                raise ValueError("the schedule did not contain EventName")

            return schedule
        except Exception as exc:
            errors.append(f"{backend_label}: {exc}")

    raise ValueError(
        f"Could not load the {year} event schedule. "
        f"Details: {' | '.join(errors)}"
    )


def _get_session(
    year: int,
    event: str | int,
    session_name: str,
    backend: str | None,
):
    """Create a FastF1 session, with compatibility for older releases."""

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
        return fastf1.get_session(
            int(year),
            event,
            session_name,
        )


def _assert_laps_are_loaded(session, *, backend_label: str) -> None:
    """Fail clearly if FastF1 created a session but loaded no lap timing."""

    try:
        laps = session.laps
    except Exception as exc:
        raise ValueError(
            f"{backend_label}: FastF1 created the session, "
            "but lap timing was not loaded."
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

    session._f1lap_backend = backend_label
    session._f1lap_telemetry_loaded = telemetry

    return session


def _load_attempt_plan(
    *,
    include_telemetry: bool,
) -> list[tuple[str | None, bool, bool]]:
    """Return a small FastF1 retry plan before using Jolpica.

    Repeating several schedule backends does not solve data-centre IP blocking,
    because lap timing still comes from Formula 1's live-timing endpoints. The
    second attempt only removes optional telemetry and weather requests.
    """

    attempts = [
        (None, include_telemetry, True),
        (None, False, False),
    ]

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
    offline_mode: bool | None = None,
):
    """Load a FastF1 session before the caller considers other sources."""

    enable_fastf1_cache(
        cache_dir,
        force_renew=force_renew,
        offline_mode=offline_mode,
    )

    errors: list[str] = []

    for backend, telemetry, weather in _load_attempt_plan(
        include_telemetry=include_telemetry,
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
                f"{backend_label}, telemetry={telemetry}, "
                f"weather={weather}: {exc}"
            )

    raise ValueError(
        "FastF1 could not load usable lap data for this session. "
        "The Formula 1 live-timing service may be unavailable from the "
        "deployment network, or the selected session may have incomplete "
        "public timing data. "
        f"Details: {' | '.join(errors)}"
    )


def session_laps_to_frame(session) -> pd.DataFrame:
    """Turn a loaded FastF1 session into a normal pandas DataFrame."""

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
        getattr(session, "_f1lap_backend", "fastf1")
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


def _collect_ergast_pages(response) -> pd.DataFrame:
    """Collect every page from a pandas-style FastF1 Ergast response."""

    frames: list[pd.DataFrame] = []
    current = response

    while True:
        content = getattr(current, "content", None)

        if content is None:
            frame = pd.DataFrame(current)
            if not frame.empty:
                frames.append(frame)
        elif isinstance(content, (list, tuple)):
            for item in content:
                frame = pd.DataFrame(item)
                if not frame.empty:
                    frames.append(frame)
        else:
            frame = pd.DataFrame(content)
            if not frame.empty:
                frames.append(frame)

        get_next_page = getattr(current, "get_next_result_page", None)
        if get_next_page is None:
            break

        try:
            current = get_next_page()
        except ValueError:
            break

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _normalise_event_name(value: str) -> str:
    text = str(value).strip().casefold()

    for suffix in (" grand prix", " gp"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break

    return "".join(character for character in text if character.isalnum())


def _resolve_jolpica_event(
    year: int,
    event: str | int,
) -> tuple[int, str]:
    """Resolve an event name or round number using the Jolpica schedule."""

    fastf1 = _fastf1()
    schedule = pd.DataFrame(
        fastf1.get_event_schedule(
            int(year),
            include_testing=False,
            backend="ergast",
        )
    )

    required = {"RoundNumber", "EventName"}
    missing = required.difference(schedule.columns)
    if schedule.empty or missing:
        raise ValueError(
            "Jolpica returned an invalid event schedule. "
            f"Missing columns: {sorted(missing)}"
        )

    is_round_number = isinstance(event, int) or (
        isinstance(event, str) and event.strip().isdigit()
    )

    if is_round_number:
        round_number = int(event)
        matching = schedule[
            pd.to_numeric(schedule["RoundNumber"], errors="coerce")
            == round_number
        ]
    else:
        target = _normalise_event_name(str(event))
        normalised_names = schedule["EventName"].map(_normalise_event_name)
        matching = schedule[normalised_names == target]

    if matching.empty:
        raise ValueError(
            f"Could not find {event!r} in the {year} Jolpica schedule."
        )

    row = matching.iloc[0]
    return int(row["RoundNumber"]), str(row["EventName"])


def _jolpica_laps_to_frame(
    laps: pd.DataFrame,
    results: pd.DataFrame,
    *,
    year: int,
    event_name: str,
) -> pd.DataFrame:
    """Convert Jolpica lap timing into the FastF1-style columns used here."""

    required = {"number", "driverId", "time"}
    missing = required.difference(laps.columns)
    if laps.empty or missing:
        raise ValueError(
            "Jolpica returned invalid lap data. "
            f"Missing columns: {sorted(missing)}"
        )

    output = laps.copy()
    output["driverId"] = output["driverId"].astype(str)

    driver_map: dict[str, str] = {}
    team_map: dict[str, str] = {}

    if not results.empty and "driverId" in results.columns:
        result_rows = results.drop_duplicates("driverId").copy()
        result_rows["driverId"] = result_rows["driverId"].astype(str)

        display_driver = pd.Series(
            pd.NA,
            index=result_rows.index,
            dtype="object",
        )

        if "driverCode" in result_rows.columns:
            display_driver = result_rows["driverCode"].replace("", pd.NA)

        if "familyName" in result_rows.columns:
            display_driver = display_driver.fillna(
                result_rows["familyName"].replace("", pd.NA)
            )

        display_driver = display_driver.fillna(result_rows["driverId"])
        driver_map = dict(zip(result_rows["driverId"], display_driver))

        if "constructorName" in result_rows.columns:
            team_names = result_rows["constructorName"].fillna("Unknown")
            team_map = dict(zip(result_rows["driverId"], team_names))

    output["Driver"] = output["driverId"].map(driver_map)
    output["Driver"] = output["Driver"].fillna(output["driverId"]).astype(str)

    output["Team"] = output["driverId"].map(team_map).fillna("Unknown")

    output = output.rename(
        columns={
            "number": "LapNumber",
            "time": "LapTime",
            "position": "Position",
        }
    )

    output["LapNumber"] = pd.to_numeric(
        output["LapNumber"],
        errors="coerce",
    )
    lap_time_values = output["LapTime"]
    if not pd.api.types.is_timedelta64_dtype(lap_time_values):
        lap_time_values = lap_time_values.astype("string").map(
            lambda value: (
                f"00:{value}"
                if value is not pd.NA and str(value).count(":") == 1
                else value
            )
        )

    output["LapTime"] = pd.to_timedelta(
        lap_time_values,
        errors="coerce",
    )

    output["Year"] = int(year)
    output["EventName"] = str(event_name)
    output["SessionName"] = "Race"
    output["Stint"] = 1
    output["TyreLife"] = pd.NA
    output["Compound"] = "Unknown"
    output["TrackStatus"] = "1"
    output["AirTemp"] = pd.NA
    output["TrackTemp"] = pd.NA
    output["Humidity"] = pd.NA
    output["WindSpeed"] = pd.NA
    output["Rainfall"] = False
    output["IsAccurate"] = True
    output["DataBackend"] = "jolpica"
    output["TelemetryLoaded"] = False

    output = output.dropna(subset=["LapNumber", "LapTime"])

    if output.empty:
        raise ValueError("Jolpica returned no usable lap-time values.")

    return output.sort_values(
        ["LapNumber", "Driver"],
        ignore_index=True,
    )


def load_jolpica_race_laps(
    year: int,
    event: str | int,
) -> pd.DataFrame:
    """Load basic race lap timing from Jolpica.

    Jolpica provides lap number, driver, position, and lap time. It does not
    provide FastF1 telemetry, tyre compounds, detailed track status, or weather.
    """

    try:
        from fastf1.ergast import Ergast
    except ImportError as exc:
        raise ImportError(
            "The FastF1 Ergast/Jolpica interface is unavailable."
        ) from exc

    round_number, event_name = _resolve_jolpica_event(year, event)

    ergast = Ergast(
        result_type="pandas",
        auto_cast=True,
        limit=1000,
    )

    lap_response = ergast.get_lap_times(
        season=int(year),
        round=round_number,
        limit=1000,
    )
    laps = _collect_ergast_pages(lap_response)

    if laps.empty:
        raise ValueError(
            f"Jolpica returned no laps for {year} round {round_number}."
        )

    try:
        result_response = ergast.get_race_results(
            season=int(year),
            round=round_number,
            limit=100,
        )
        results = _collect_ergast_pages(result_response)
    except Exception:
        results = pd.DataFrame()

    return _jolpica_laps_to_frame(
        laps,
        results,
        year=year,
        event_name=event_name,
    )


def load_fastf1_laps(
    year: int,
    event: str | int,
    session_name: str = "Race",
    *,
    cache_dir: str | Path = ".fastf1_cache",
    include_telemetry: bool = True,
    force_renew: bool = False,
) -> tuple[object | None, pd.DataFrame]:
    """Load FastF1 data, falling back to Jolpica for completed races."""

    try:
        session = load_fastf1_session(
            year,
            event,
            session_name,
            cache_dir=cache_dir,
            include_telemetry=include_telemetry,
            force_renew=force_renew,
        )
        return session, session_laps_to_frame(session)

    except Exception as fastf1_error:
        if session_name.strip().casefold() not in RACE_SESSION_NAMES:
            raise ValueError(
                "FastF1 failed and Jolpica lap-by-lap fallback is only "
                "available for Race sessions. "
                f"FastF1 details: {fastf1_error}"
            ) from fastf1_error

        try:
            laps = load_jolpica_race_laps(year, event)
            return None, laps
        except Exception as jolpica_error:
            raise ValueError(
                "Both FastF1 live timing and the Jolpica race-lap fallback "
                "failed. "
                f"FastF1: {fastf1_error} | "
                f"Jolpica: {jolpica_error}"
            ) from jolpica_error


def available_events(schedule: pd.DataFrame) -> list[str]:
    """Return clean event names for a sidebar dropdown."""

    if "EventName" not in schedule.columns:
        return []

    return [
        str(name)
        for name in schedule["EventName"].dropna().tolist()
    ]
