from __future__ import annotations

import argparse
from pathlib import Path

import fastf1


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".fastf1_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a FastF1 session into the project's local cache."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--session", default="Race")
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Cache lap timing without car and position telemetry.",
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
