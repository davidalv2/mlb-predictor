"""
data_collector.py
-----------------
Downloads and caches team batting and pitching data from Fangraphs via pybaseball.
Saves CSVs locally to avoid repeated API calls.

Usage:
    python src/data_collector.py
"""

import warnings
import pandas as pd
from pathlib import Path
from pybaseball import team_batting, pitching_stats, cache

warnings.filterwarnings("ignore")
cache.enable()

CACHE_DIR = Path("data/raw")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEAM_NAME_TO_CODE = {
    "Yankees": "NYY", "Red Sox": "BOS", "Blue Jays": "TOR",
    "Rays": "TBR",    "Orioles": "BAL", "White Sox": "CHW",
    "Guardians": "CLE","Tigers": "DET", "Royals": "KCR",
    "Twins": "MIN",   "Astros": "HOU", "Mariners": "SEA",
    "Rangers": "TEX", "Athletics": "OAK","Angels": "LAA",
    "Braves": "ATL",  "Mets": "NYM",   "Phillies": "PHI",
    "Marlins": "MIA", "Nationals": "WSN","Cubs": "CHC",
    "Cardinals": "STL","Brewers": "MIL","Pirates": "PIT",
    "Reds": "CIN",    "Dodgers": "LAD", "Giants": "SFG",
    "Padres": "SDP",  "Rockies": "COL", "Diamondbacks": "ARI",
}


class DataCollector:
    """
    Downloads batting (team-level) and pitching (player-level) stats
    from Fangraphs via pybaseball. Caches to CSV to avoid re-downloading.

    Args:
        years         : list of seasons to download (e.g. [2023, 2024, 2025])
        force_refresh : if True, re-downloads even if cache exists
    """

    def __init__(self, years: list, force_refresh: bool = False):
        self.years = years
        self.force_refresh = force_refresh
        self._batting = None
        self._pitching = None

    def get_batting(self) -> pd.DataFrame:
        """Returns team batting stats for all seasons. Loads from cache if available."""
        if self._batting is not None:
            return self._batting
        frames = []
        for year in self.years:
            path = CACHE_DIR / f"batting_{year}.csv"
            if path.exists() and not self.force_refresh:
                print(f"  [cache] batting {year}")
                df = pd.read_csv(path)
            else:
                print(f"  [download] batting {year} ...")
                df = team_batting(year)
                df["Season"] = year
                df.to_csv(path, index=False)
            frames.append(df)
        self._batting = pd.concat(frames, ignore_index=True)
        return self._batting

    def get_pitching(self, qual: int = 0) -> pd.DataFrame:
        """Returns individual pitcher stats for all seasons. Loads from cache if available."""
        if self._pitching is not None:
            return self._pitching
        frames = []
        for year in self.years:
            path = CACHE_DIR / f"pitching_{year}.csv"
            if path.exists() and not self.force_refresh:
                print(f"  [cache] pitching {year}")
                df = pd.read_csv(path)
            else:
                print(f"  [download] pitching {year} ...")
                df = pitching_stats(year, qual=qual)
                df["Season"] = year
                df.to_csv(path, index=False)
            frames.append(df)
        self._pitching = pd.concat(frames, ignore_index=True)
        return self._pitching

    def resolve_team(self, name: str) -> str:
        """Converts common team name to official code. e.g. 'Yankees' -> 'NYY'"""
        key = name.strip().title()
        if key not in TEAM_NAME_TO_CODE:
            raise ValueError(
                f"Team '{name}' not recognized. "
                f"Valid names: {sorted(TEAM_NAME_TO_CODE.keys())}"
            )
        return TEAM_NAME_TO_CODE[key]


if __name__ == "__main__":
    dc = DataCollector(years=[2023, 2024, 2025])

    print("Downloading batting stats...")
    bat = dc.get_batting()
    print(f"  -> {len(bat)} rows | seasons: {sorted(bat['Season'].unique())}")

    print("\nDownloading pitching stats...")
    pit = dc.get_pitching()
    print(f"  -> {len(pit)} rows | seasons: {sorted(pit['Season'].unique())}")

    print("\n--- Batting sample ---")
    print(bat[["Team", "Season", "AVG", "OBP", "SLG", "OPS", "wRC+"]].head())

    print("\n--- Pitching sample ---")
    print(pit[["Name", "Team", "Season", "ERA", "WHIP", "FIP", "K-BB%"]].head())
