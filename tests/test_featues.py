"""
test_features.py
----------------
Unit tests for feature engineering pipeline.

Run:
    pytest tests/test_features.py -v
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineer import FeatureEngineer, Matchup, remove_accents


# ── Fixtures ──────────────────────────────────────────────────────────

def make_batting_row(team, season, ops=0.750, obp=0.330, slg=0.420,
                     avg=0.250, wrc=110, woba=0.330, hh=40.0,
                     barrel=8.0, ev=88.0, hard=35.0, iso=0.170,
                     babip=0.300, xwoba=0.330, bb=9.0, k=22.0):
    return {
        "Team": team, "Season": season,
        "OPS": ops, "OBP": obp, "SLG": slg, "AVG": avg,
        "wRC+": wrc, "wOBA": woba, "HardHit%": hh,
        "Barrel%": barrel, "EV": ev, "Hard%": hard,
        "ISO": iso, "BABIP": babip, "xwOBA": xwoba,
        "BB%": bb, "K%": k,
    }

def make_pitching_row(name, team, season, era=4.00, whip=1.25, fip=3.90,
                      xfip=3.85, xera=3.90, siera=3.80, k9=9.5, bb9=2.8,
                      hr9=1.1, kbb=16.0, swstr=11.5, stuff=105.0, ip=150.0):
    return {
        "Name": name, "Team": team, "Season": season, "Name_clean": name.lower(),
        "ERA": era, "WHIP": whip, "FIP": fip, "xFIP": xfip, "xERA": xera,
        "SIERA": siera, "K/9": k9, "BB/9": bb9, "HR/9": hr9,
        "K-BB%": kbb, "SwStr%": swstr, "Stuff+": stuff, "IP": ip,
    }

@pytest.fixture
def sample_data():
    bat = pd.DataFrame([
        make_batting_row("NYY", 2024, ops=0.780, wrc=120),
        make_batting_row("BOS", 2024, ops=0.740, wrc=105),
    ])
    pit = pd.DataFrame([
        make_pitching_row("Gerrit Cole", "NYY", 2024, era=3.41, fip=3.69),
        make_pitching_row("Brayan Bello", "BOS", 2024, era=4.12, fip=4.33),
    ])
    return bat, pit

@pytest.fixture
def fe(sample_data):
    bat, pit = sample_data
    return FeatureEngineer(bat, pit)


# ── Tests ─────────────────────────────────────────────────────────────

def test_remove_accents():
    assert remove_accents("Néstor Cortés") == "Nestor Cortes"
    assert remove_accents("Pablo López")   == "Pablo Lopez"
    assert remove_accents("Gerrit Cole")   == "Gerrit Cole"

def test_build_dataset_returns_correct_shapes(fe):
    matchups = [
        Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1),
        Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=0),
    ]
    X, y, meta = fe.build_dataset(matchups)
    assert X.shape[0] == 2
    assert len(y) == 2
    assert len(meta) == 2

def test_features_are_not_nan(fe):
    matchups = [Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1)]
    X, y, meta = fe.build_dataset(matchups)
    assert X.isna().sum().sum() == 0

def test_batting_diff_direction(fe):
    """NYY has higher OPS than BOS — bat_diff_OPS should be positive."""
    matchups = [Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1)]
    X, _, _ = fe.build_dataset(matchups)
    assert X["bat_diff_OPS"].iloc[0] > 0

def test_pitching_diff_direction(fe):
    """Cole has lower ERA than Bello — pit_diff_ERA should be positive (inverted)."""
    matchups = [Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1)]
    X, _, _ = fe.build_dataset(matchups)
    assert X["pit_diff_ERA"].iloc[0] > 0

def test_composite_features_exist(fe):
    matchups = [Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1)]
    X, _, _ = fe.build_dataset(matchups)
    assert "comp_ops_vs_fip_diff"       in X.columns
    assert "comp_wrc_vs_xera_diff"      in X.columns
    assert "comp_hardhit_vs_swstr_diff" in X.columns

def test_build_single(fe):
    m = Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024)
    X = fe.build_single(m)
    assert X.shape[0] == 1
    assert X.isna().sum().sum() == 0

def test_unknown_team_raises(fe):
    m = Matchup("FAKE", "BOS", "Gerrit Cole", "Brayan Bello", 2024)
    X, y, _ = fe.build_dataset([m])
    assert len(X) == 0  # skipped silently

def test_unknown_pitcher_uses_league_average(fe):
    """Unknown pitcher should fall back to league average, not raise."""
    m = Matchup("NYY", "BOS", "Unknown Pitcher XYZ", "Brayan Bello", 2024, home_win=1)
    X, y, _ = fe.build_dataset([m])
    assert len(X) == 1  # not skipped

def test_season_norm(fe):
    matchups = [Matchup("NYY", "BOS", "Gerrit Cole", "Brayan Bello", 2024, home_win=1)]
    X, _, _ = fe.build_dataset(matchups)
    assert X["season_norm"].iloc[0] == pytest.approx(0.5)
