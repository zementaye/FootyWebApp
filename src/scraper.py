"""
Scraper: pulls upcoming fixtures (with odds, when bookmakers have posted them)
from football-data.co.uk's live fixtures feed.

football-data.co.uk publishes a single rolling fixtures.csv covering ALL their
leagues (not just EPL) for matches not yet played, with the same columns as
their historical season files (Div, Date, HomeTeam, AwayTeam, B365H, B365D,
B365A, etc.) — odds columns are populated once bookmakers have priced the match,
which is usually a few days out, and empty further in advance.

This module is deliberately network-isolated from model logic: it only ever
returns fixture rows, never touches training data, so a scrape failure can
never corrupt the model.
"""

import requests
import pandas as pd
import io
import logging

logger = logging.getLogger("scraper")

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# football-data.co.uk division codes -> friendly names
LEAGUE_CODES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

ODDS_HOME_CANDIDATES = ["B365H", "PSH", "WHH", "AvgH"]
ODDS_DRAW_CANDIDATES = ["B365D", "PSD", "WHD", "AvgD"]
ODDS_AWAY_CANDIDATES = ["B365A", "PSA", "WHA", "AvgA"]


def _first_available(row, candidates):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def fetch_upcoming_fixtures(leagues=None, timeout=15):
    """
    Returns a list of dicts: [{league, date, home_team, away_team,
                                odds_home, odds_draw, odds_away}, ...]

    leagues: optional list of football-data.co.uk div codes to filter to
             (e.g. ["E0"] for Premier League only). Defaults to all 5 major leagues.
    """
    leagues = leagues or list(LEAGUE_CODES.keys())

    resp = requests.get(FIXTURES_URL, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.BytesIO(resp.content), encoding="latin1", on_bad_lines="skip")

    df = df[df["Div"].isin(leagues)].copy()

    fixtures = []
    for _, row in df.iterrows():
        oh = _first_available(row, ODDS_HOME_CANDIDATES)
        od = _first_available(row, ODDS_DRAW_CANDIDATES)
        oa = _first_available(row, ODDS_AWAY_CANDIDATES)

        fixtures.append({
            "league": LEAGUE_CODES.get(row["Div"], row["Div"]),
            "date": row.get("Date"),
            "time": row.get("Time", None),
            "home_team": row["HomeTeam"],
            "away_team": row["AwayTeam"],
            "odds_home": float(oh) if oh is not None else None,
            "odds_draw": float(od) if od is not None else None,
            "odds_away": float(oa) if oa is not None else None,
        })

    logger.info(f"Scraped {len(fixtures)} upcoming fixtures across {len(leagues)} leagues")
    return fixtures


if __name__ == "__main__":
    # quick manual test — run from your own machine, not the sandbox
    logging.basicConfig(level=logging.INFO)
    fx = fetch_upcoming_fixtures(leagues=["E0"])
    for f in fx[:10]:
        print(f)
