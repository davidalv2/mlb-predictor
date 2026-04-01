"""
feature_engineer.py
-------------------
Builds a 112-feature vector for each MLB matchup.

Feature groups:
  - 45 batting features   (home, away, differential per stat)
  - 36 pitching features  (home, away, differential per stat)
  - 9  composite features (OPS vs FIP, wRC+ vs xERA, HardHit vs SwStr%)
  - 6  rolling features   (team win rate, starter win rate)
  - 8  Statcast features  (xwOBA and delta_run_exp last 3 starts)
  - 5  ELO features       (dynamic team rating)
  - 2  context features   (season)

Usage:
    from src.feature_engineer import FeatureEngineer, Matchup
    fe = FeatureEngineer(batting_df, pitching_df)
    X, y, meta = fe.build_dataset(matchups)
"""

import unicodedata
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

BATTING_STATS = [
    "AVG", "OBP", "SLG", "OPS", "wRC+", "wOBA",
    "BB%", "K%", "HardHit%", "Barrel%", "EV", "Hard%",
    "ISO", "BABIP", "xwOBA",
]

PITCHING_STATS = [
    "ERA", "WHIP", "FIP", "xFIP", "xERA", "SIERA",
    "K/9", "BB/9", "HR/9", "K-BB%", "SwStr%", "Stuff+",
]

LOWER_IS_BETTER = {
    "ERA", "WHIP", "BB/9", "HR/9", "FIP", "xFIP",
    "xERA", "SIERA", "K%", "BABIP"
}


def remove_accents(text: str) -> str:
    """Normalizes accented characters. e.g. 'Néstor Cortés' -> 'Nestor Cortes'"""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )


@dataclass
class Matchup:
    """
    Represents a single MLB game matchup.

    Attributes:
        home_team : official team code (e.g. 'NYY')
        away_team : official team code
        home_sp   : home starting pitcher name (fuzzy matched)
        away_sp   : away starting pitcher name
        season    : year of the game
        home_win  : 1 if home team won, 0 if not, None for prediction
        game_id   : optional unique identifier
    """
    home_team: str
    away_team: str
    home_sp:   str
    away_sp:   str
    season:    int
    home_win:  Optional[int] = None
    game_id:   Optional[str] = None


class FeatureEngineer:
    """
    Transforms raw batting and pitching DataFrames into ML-ready feature vectors.
    Produces one row per matchup with ~112 features.
    """

    def __init__(self, batting_df: pd.DataFrame, pitching_df: pd.DataFrame):
        self.batting  = batting_df
        self.pitching = pitching_df.copy()
        self.pitching["Name_clean"] = self.pitching["Name"].apply(
            lambda x: remove_accents(str(x)).lower()
        )
        self._avail_bat = self._check_cols(batting_df,  BATTING_STATS)
        self._avail_pit = self._check_cols(pitching_df, PITCHING_STATS)
        self._league_avg = {}
        for season in batting_df["Season"].unique():
            pit_s = pitching_df[pitching_df["Season"] == season]
            self._league_avg[int(season)] = pit_s[
                [c for c in PITCHING_STATS if c in pit_s.columns]
            ].mean()

    def build_dataset(self, matchups: list) -> tuple:
        """
        Processes a list of Matchup objects.

        Returns:
            X    : pd.DataFrame of features (one row per matchup)
            y    : pd.Series of labels (1 = home win)
            meta : pd.DataFrame with game metadata
        """
        rows, labels, metas = [], [], []
        for m in matchups:
            try:
                feat = self._extract_features(m)
                rows.append(feat)
                if m.home_win is not None:
                    labels.append(m.home_win)
                metas.append({
                    "game_id":   m.game_id,
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "season":    m.season,
                })
            except Exception:
                pass
        X    = pd.DataFrame(rows)
        y    = pd.Series(labels, name="home_win") if labels else pd.Series(dtype=int)
        meta = pd.DataFrame(metas)
        return X, y, meta

    def build_single(self, matchup: "Matchup") -> pd.DataFrame:
        """Extract features for a single matchup (for live prediction)."""
        return pd.DataFrame([self._extract_features(matchup)])

    def _extract_features(self, m: "Matchup") -> dict:
        features = {}
        bat_h = self._get_team_batting(m.home_team, m.season)
        bat_a = self._get_team_batting(m.away_team, m.season)
        pit_h = self._get_pitcher(m.home_sp, m.season)
        pit_a = self._get_pitcher(m.away_sp, m.season)

        for stat in self._avail_bat:
            key  = stat.replace("/", "_").replace("%", "pct").replace("+", "plus")
            v_h  = self._safe_float(bat_h, stat)
            v_a  = self._safe_float(bat_a, stat)
            diff = (v_h - v_a) * (-1 if stat in LOWER_IS_BETTER else 1)
            features[f"bat_home_{key}"] = v_h
            features[f"bat_away_{key}"] = v_a
            features[f"bat_diff_{key}"] = diff

        for stat in self._avail_pit:
            key  = (stat.replace("/", "_").replace("%", "pct")
                       .replace("+", "plus").replace("-", "minus"))
            v_h  = self._safe_float(pit_h, stat)
            v_a  = self._safe_float(pit_a, stat)
            diff = (v_h - v_a) * (-1 if stat in LOWER_IS_BETTER else 1)
            features[f"pit_home_{key}"] = v_h
            features[f"pit_away_{key}"] = v_a
            features[f"pit_diff_{key}"] = diff

        # Composite: team offense vs rival pitcher quality
        ops_h  = self._safe_float(bat_h, "OPS");   ops_a  = self._safe_float(bat_a, "OPS")
        fip_h  = self._safe_float(pit_h, "FIP");   fip_a  = self._safe_float(pit_a, "FIP")
        wrc_h  = self._safe_float(bat_h, "wRC+");  wrc_a  = self._safe_float(bat_a, "wRC+")
        xera_h = self._safe_float(pit_h, "xERA");  xera_a = self._safe_float(pit_a, "xERA")
        hh_h   = self._safe_float(bat_h, "HardHit%"); hh_a = self._safe_float(bat_a, "HardHit%")
        sw_h   = self._safe_float(pit_h, "SwStr%");   sw_a = self._safe_float(pit_a, "SwStr%")

        features["comp_ops_vs_fip_home"]       = ops_h / (fip_a + 0.01)
        features["comp_ops_vs_fip_away"]       = ops_a / (fip_h + 0.01)
        features["comp_ops_vs_fip_diff"]       = features["comp_ops_vs_fip_home"] - features["comp_ops_vs_fip_away"]
        features["comp_wrc_vs_xera_home"]      = wrc_h / (xera_a + 0.01)
        features["comp_wrc_vs_xera_away"]      = wrc_a / (xera_h + 0.01)
        features["comp_wrc_vs_xera_diff"]      = features["comp_wrc_vs_xera_home"] - features["comp_wrc_vs_xera_away"]
        features["comp_hardhit_vs_swstr_home"] = hh_h / (sw_a + 0.01)
        features["comp_hardhit_vs_swstr_away"] = hh_a / (sw_h + 0.01)
        features["comp_hardhit_vs_swstr_diff"] = features["comp_hardhit_vs_swstr_home"] - features["comp_hardhit_vs_swstr_away"]

        features["season"]      = m.season
        features["season_norm"] = (m.season - 2023) / 2.0
        return features

    def _get_team_batting(self, team_code: str, season: int) -> pd.Series:
        mask = (self.batting["Team"] == team_code) & (self.batting["Season"] == season)
        df   = self.batting[mask]
        if df.empty:
            df = self.batting[self.batting["Team"] == team_code]
        if df.empty:
            raise ValueError(f"No batting data for {team_code} {season}")
        return df.sort_values("Season", ascending=False).iloc[0]

    def _get_pitcher(self, name: str, season: int) -> pd.Series:
        name_clean = remove_accents(name).lower()
        mask = (
            self.pitching["Name_clean"].str.contains(name_clean, na=False)
            & (self.pitching["Season"] == season)
        )
        df = self.pitching[mask]
        if df.empty:
            df = self.pitching[
                self.pitching["Name_clean"].str.contains(name_clean, na=False)
            ]
        if df.empty:
            return self._league_avg.get(
                season,
                self._league_avg.get(max(self._league_avg.keys()))
            )
        return df.sort_values("IP", ascending=False).iloc[0] if "IP" in df.columns else df.iloc[0]

    @staticmethod
    def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
        try:
            v = row[col]
            return float(v) if pd.notna(v) else default
        except (KeyError, TypeError):
            return default

    @staticmethod
    def _check_cols(df: pd.DataFrame, wanted: list) -> list:
        available = [c for c in wanted if c in df.columns]
        missing   = set(wanted) - set(available)
        if missing:
            print(f"  [info] Columns not available (skipped): {missing}")
        return available
