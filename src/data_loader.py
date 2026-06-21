import fastf1
import pandas as pd
from pathlib import Path

cache_path = Path.cwd().parent / "cache"
fastf1.Cache.enable_cache(str(cache_path))

def race_load(year, grand_prix, session_type="R"):
    session = fastf1.get_session(year, grand_prix, session_type)
    session.load()

    laps = session.laps.copy()
    laps = laps[
        laps["LapTime"].notna() &
        laps["Compound"].notna() &
        laps["TyreLife"].notna()
    ]

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    laps["Year"] = year
    laps["GrandPrix"] = grand_prix

    return laps
