from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import fastf1


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".fastf1_cache"
DEPLOY_ZIP = ROOT / "data" / "fastf1_deploy_cache.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a FastF1 session into the project cache and optionally "
            "pack it for Streamlit Cloud deployment."
        )
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--session", default="Race")
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Cache lap timing without car and position telemetry.",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help=(
            f"Write {DEPLOY_ZIP.relative_to(ROOT)} so Streamlit Cloud can "
            "load this session in FastF1 offline mode."
        ),
    )
    return parser.parse_args()


def _session_folder_name(session) -> str:
    event_date = session.event["EventDate"].strftime("%Y-%m-%d")
    event_name = str(session.event["EventName"]).replace(" ", "_")
    return f"{event_date}_{event_name}"


def pack_deploy_cache(session) -> Path:
    """Pack the HTTP cache plus this session's pickle cache into the deploy zip."""

    DEPLOY_ZIP.parent.mkdir(parents=True, exist_ok=True)

    year = int(session.event["EventDate"].year)
    session_dir = CACHE_DIR / str(year) / _session_folder_name(session)
    http_cache = CACHE_DIR / "fastf1_http_cache.sqlite"

    if not http_cache.exists():
        raise RuntimeError(f"Missing HTTP cache file: {http_cache}")
    if not session_dir.exists():
        raise RuntimeError(f"Missing session cache directory: {session_dir}")

    with zipfile.ZipFile(
        DEPLOY_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zip_file:
        zip_file.write(
            http_cache,
            http_cache.relative_to(CACHE_DIR).as_posix(),
        )
        for path in session_dir.rglob("*"):
            if path.is_file():
                zip_file.write(
                    path,
                    path.relative_to(CACHE_DIR).as_posix(),
                )

    return DEPLOY_ZIP


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    session = fastf1.get_session(
        args.year,
        args.event,
        args.session,
    )
    session.load(
        laps=True,
        telemetry=not args.no_telemetry,
        weather=True,
        messages=False,
    )

    laps = session.laps
    if laps is None or laps.empty or laps["LapTime"].notna().sum() == 0:
        raise RuntimeError("FastF1 did not return usable lap timing.")

    print(
        f"Cached {len(laps):,} lap rows for "
        f"{args.year} {args.event} {args.session}."
    )
    print(f"Cache directory: {CACHE_DIR}")

    if args.pack:
        archive = pack_deploy_cache(session)
        print(
            f"Packed deploy cache: {archive} "
            f"({archive.stat().st_size / 1_000_000:.1f} MB)"
        )


if __name__ == "__main__":
    main()
