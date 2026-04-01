"""
predict.py
----------
Predicts the outcome of an MLB game using trained model weights.

Usage:
    python src/predict.py

    Or import:
    from src.predict import predict_game
    predict_game("Yankees", "Red Sox", "Gerrit Cole", "Brayan Bello", 2026)
"""

import warnings
import pandas as pd
warnings.filterwarnings("ignore")

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

BATTING_W = {
    "full":     {"AVG": 0.5, "OBP": 3.0, "SLG": 1.5, "OPS": 4.5, "wRC+": 2.0},
    "5innings": {"AVG": 0.3, "OBP": 3.5, "SLG": 1.0, "OPS": 4.0, "wRC+": 2.5},
}
PITCHING_W = {
    "full":     {"ERA": 4.0, "WHIP": 3.0, "FIP": 2.0, "K/9": 1.0, "BB/9": 1.5, "HR/9": 1.0},
    "5innings": {"ERA": 5.0, "WHIP": 3.5, "FIP": 2.5, "K/9": 0.7, "BB/9": 2.0, "HR/9": 0.8},
}
LOWER_IS_BETTER = {"ERA", "WHIP", "FIP", "xFIP", "BB/9", "HR/9"}


def _score_mode(b1, b2, p1, p2, t1, t2, mode):
    s1, s2 = 0.0, 0.0
    label  = "Full game" if mode == "full" else "First 5 innings"
    print(f"\n  [{label}]")
    print(f"  {'Stat':<10} {t1:>15} {t2:>15}  Edge")
    print(f"  {'-'*55}")

    for stat, weight in BATTING_W[mode].items():
        if stat not in b1.index or stat not in b2.index:
            continue
        v1, v2 = float(b1[stat]), float(b2[stat])
        winner = t1 if v1 > v2 else t2
        if winner == t1: s1 += weight
        else:            s2 += weight
        print(f"  {stat:<10} {v1:>15.3f} {v2:>15.3f}  {winner} (+{weight})")

    print()
    for stat, weight in PITCHING_W[mode].items():
        if stat not in p1.index or stat not in p2.index:
            continue
        v1, v2 = float(p1[stat]), float(p2[stat])
        winner = t1 if (v1 < v2 if stat in LOWER_IS_BETTER else v1 > v2) else t2
        if winner == t1: s1 += weight
        else:            s2 += weight
        print(f"  {stat:<10} {v1:>15.2f} {v2:>15.2f}  {winner} (+{weight})")

    print(f"\n  Score -> {t1}: {s1:.1f}  |  {t2}: {s2:.1f}")
    if s1 == s2:
        e1  = float(p1["ERA"]) if "ERA" in p1.index else 4.5
        e2  = float(p2["ERA"]) if "ERA" in p2.index else 4.5
        win = t1 if e1 < e2 else t2
        print(f"  Tie -> ERA tiebreak -> {win}")
    else:
        win = t1 if s1 > s2 else t2
        print(f"  Prediction -> {win}")
    return s1, s2, win


def predict_game(
    team1: str,
    team2: str,
    sp1: str,
    sp2: str,
    year: int = 2026,
    bat: pd.DataFrame = None,
    pit: pd.DataFrame = None,
):
    """
    Predicts the outcome of an MLB game.

    Args:
        team1 : home team name (e.g. 'Yankees')
        team2 : away team name
        sp1   : home starting pitcher name
        sp2   : away starting pitcher name
        year  : season year for stats lookup
        bat   : optional batting DataFrame (loaded from cache if None)
        pit   : optional pitching DataFrame (loaded from cache if None)
    """
    if team1 not in TEAM_NAME_TO_CODE or team2 not in TEAM_NAME_TO_CODE:
        print(f"Error: unrecognized team.")
        print(f"Valid teams: {sorted(TEAM_NAME_TO_CODE.keys())}")
        return

    c1 = TEAM_NAME_TO_CODE[team1]
    c2 = TEAM_NAME_TO_CODE[team2]

    if bat is None or pit is None:
        from src.data_collector import DataCollector
        dc  = DataCollector(years=[year, year - 1])
        bat = dc.get_batting()
        pit = dc.get_pitching()

    b1 = bat[(bat["Team"] == c1) & (bat["Season"] == year)]
    b2 = bat[(bat["Team"] == c2) & (bat["Season"] == year)]
    if b1.empty: b1 = bat[(bat["Team"] == c1) & (bat["Season"] == year - 1)]
    if b2.empty: b2 = bat[(bat["Team"] == c2) & (bat["Season"] == year - 1)]

    p1 = pit[pit["Name"].str.lower().str.contains(sp1.lower(), na=False) & (pit["Season"] == year)]
    p2 = pit[pit["Name"].str.lower().str.contains(sp2.lower(), na=False) & (pit["Season"] == year)]
    if p1.empty: p1 = pit[pit["Name"].str.lower().str.contains(sp1.lower(), na=False)]
    if p2.empty: p2 = pit[pit["Name"].str.lower().str.contains(sp2.lower(), na=False)]

    if b1.empty or b2.empty or p1.empty or p2.empty:
        print("Not enough data found for this matchup.")
        return

    b1r = b1.iloc[0]
    b2r = b2.iloc[0]
    p1r = p1.sort_values("IP", ascending=False).iloc[0] if "IP" in p1.columns else p1.iloc[0]
    p2r = p2.sort_values("IP", ascending=False).iloc[0] if "IP" in p2.columns else p2.iloc[0]

    print(f"\n{'='*62}")
    print(f"  {team1}  vs  {team2}  |  {year}")
    print(f"  {sp1}  vs  {sp2}")
    print(f"{'='*62}")

    scores, winners = {}, {}
    for mode in ["full", "5innings"]:
        s1, s2, win   = _score_mode(b1r, b2r, p1r, p2r, team1, team2, mode)
        scores[mode]  = (s1, s2)
        winners[mode] = win

    d_full = abs(scores["full"][0]     - scores["full"][1])
    d_5inn = abs(scores["5innings"][0] - scores["5innings"][1])
    avg    = (d_full + d_5inn) / 2

    print(f"\n{'='*62}")
    print(f"  FINAL ASSESSMENT")
    print(f"{'='*62}")
    if winners["full"] != winners["5innings"]:
        print("\n  Mixed signals — close game, enjoy it")
    else:
        print(f"\n  Average margin: {avg:.2f} pts")
        if   avg >= 7.5: print("  STRONG SIGNAL")
        elif avg >= 3.0: print("  SOLID SIGNAL")
        else:            print("  WEAK SIGNAL — proceed with caution")

    return {
        "winner_full": winners["full"],
        "winner_5inn": winners["5innings"],
        "avg_diff": avg,
    }


if __name__ == "__main__":
    print("MLB Game Predictor")
    print("-" * 40)
    team1 = input("Home team    : ").strip().title()
    team2 = input("Away team    : ").strip().title()
    sp1   = input(f"Starter ({team1}): ").strip()
    sp2   = input(f"Starter ({team2}): ").strip()
    predict_game(team1, team2, sp1, sp2, year=2026)
