"""
xG (expected goals) data source: Understat, via the `understatapi` package
(not in requirements.txt by default — it's an optional dependency, only
needed if you actually run download_xg_data.py).

WHY THIS IS UNTESTED END-TO-END:
This sandbox has no network access, so fetch_understat_xg() below has never
actually been run against the live site — same situation as the existing
football-data.co.uk scraper (see README_DEPLOY.md, which flags the same
limitation for src/scraper.py). It's written against Understat's documented
per-match JSON schema (m["h"]["title"], m["xG"]["h"], etc.). Run it yourself
first via download_xg_data.py before relying on it.

TEAM NAME MISMATCH:
Understat's club names don't match football-data.co.uk's naming for a
chunk of EPL clubs (e.g. Understat "Manchester United" vs football-data.co.uk
"Man United"). TEAM_NAME_MAP translates Understat's name -> football-data.co.uk's
name. It was built from publicly documented naming conventions for both
sources, NOT verified against a live pull — download_xg_data.py logs any
Understat row it couldn't confidently match to a match_id, which will mostly
be name-mapping misses. Check that log after the first run and fix entries
here rather than assuming the table is complete; identity entries (name maps
to itself) are listed explicitly so you can see what's been checked vs.
never looked at, rather than silently falling through a .get() default.

COVERAGE:
Understat's EPL data starts with the 2014-15 season. Matches before that will
simply have no Understat match — the rest of the pipeline (see
add_rolling_xg_form in features.py) is built to degrade gracefully when xG is
missing, not to require it, so this does not shrink the usable 1993-2026
goals-based training history.
"""

import logging

import pandas as pd

logger = logging.getLogger("xg_source")

# Understat name -> football-data.co.uk name. Identity entries are listed
# explicitly (checked, confirmed same) rather than omitted (never checked).
# Verify/extend this against the unmatched-team log from your first real run.
TEAM_NAME_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "Sheffield United": "Sheffield United",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Brighton": "Brighton",
    "Norwich": "Norwich",
    "Cardiff": "Cardiff",
    "Huddersfield": "Huddersfield",
    "Bournemouth": "Bournemouth",
    "Swansea": "Swansea",
    "Stoke": "Stoke",
    "Hull": "Hull",
    "Watford": "Watford",
    "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Southampton": "Southampton",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Luton": "Luton",
    "Ipswich": "Ipswich",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa",
}


def fetch_understat_xg(seasons, league="EPL"):
    """
    seasons: list of season-start years, e.g. [2014, 2015, ..., 2025]
             (Understat convention: season "2023" = the 2023-24 season)

    Returns a DataFrame [date, home_team, away_team, home_xg, away_xg,
    home_team_understat, away_team_understat] — the last two kept so you can
    audit TEAM_NAME_MAP against what Understat actually returned.
    """
    from understatapi import UnderstatClient  # optional dep — import kept local

    rows = []
    with UnderstatClient() as client:
        for season in seasons:
            logger.info(f"Fetching Understat {league} {season}...")
            matches = client.league(league=league).get_match_data(season=str(season))
            for m in matches:
                if not m.get("isResult"):
                    continue
                rows.append({
                    "date": m["datetime"][:10],
                    "home_team_understat": m["h"]["title"],
                    "away_team_understat": m["a"]["title"],
                    "home_xg": float(m["xG"]["h"]),
                    "away_xg": float(m["xG"]["a"]),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team_understat"].map(lambda t: TEAM_NAME_MAP.get(t, t))
    df["away_team"] = df["away_team_understat"].map(lambda t: TEAM_NAME_MAP.get(t, t))
    return df[["date", "home_team", "away_team", "home_xg", "away_xg",
               "home_team_understat", "away_team_understat"]]


def match_to_results(xg_df, results_df, date_tolerance_days=1):
    """
    Attach this project's match_id (from results_df, the football-data.co.uk
    results table) onto each Understat row, via team-name match + a small
    date tolerance (the two sources occasionally record a fixture under
    different calendar dates around late kickoffs / timezones).

    Rows that don't find a confident match_id are dropped, not guessed at,
    and the count is logged — check that count isn't suspiciously high
    (it usually means a TEAM_NAME_MAP entry is missing).

    Returns [match_id, home_xg, away_xg].
    """
    if xg_df.empty:
        return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])

    results_key = results_df[["match_id", "date", "home_team", "away_team"]].copy()
    results_key["date"] = pd.to_datetime(results_key["date"])

    merged = xg_df.merge(results_key, on=["home_team", "away_team"], suffixes=("_xg", "_res"))
    merged["date_diff"] = (merged["date_xg"] - merged["date_res"]).abs().dt.days
    merged = merged[merged["date_diff"] <= date_tolerance_days]
    # If the same fixture-pair-in-a-window matches more than one real match_id
    # (rare — a team pair playing twice in a short span), keep the closest date.
    merged = merged.sort_values("date_diff").drop_duplicates(subset=["match_id"], keep="first")

    n_unmatched = len(xg_df) - len(merged)
    if n_unmatched > 0:
        logger.warning(
            f"{n_unmatched}/{len(xg_df)} Understat rows had no confident match_id — "
            f"check TEAM_NAME_MAP entries and date_tolerance_days before trusting xG coverage."
        )

    return merged[["match_id", "home_xg", "away_xg"]].reset_index(drop=True)
