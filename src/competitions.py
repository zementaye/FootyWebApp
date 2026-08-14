"""
Competition registry.

Everything that used to be hardcoded "EPL" throughout app.py / model_service.py
now reads from here, so adding a new competition is a config change, not a
code change (aside from actually having a results CSV for it).

Each entry:
  label               human-readable name (shown in API responses)
  results_path        CSV with the same schema as data/epl_results.csv:
                       match_id, season, season_code, date, home_team,
                       away_team, fthg, ftag, ftr, hthg, htag, htr, referee
                       (+ optional stage, neutral — see below)
  odds_path           historical closing-odds CSV (bet365_1x2_home, etc.),
                       or None if not available for this competition — odds
                       are optional; the model trains fine without them
  xg_path             Understat-derived xG CSV, or None if not available
  lookback_seasons    if set, training only uses the most recent N distinct
                       `season` values in results_path instead of full
                       history — see the UCL entry below
  home_advantage      Elo home-advantage constant (points). UCL uses a
                       slightly lower value than a domestic league: a good
                       chunk of UCL fixtures are the away leg of a two-legged
                       tie or a match at a redrawn/neutral venue, so the
                       "home team" label carries a bit less signal than it
                       does in a normal domestic season.
  fixtures_league_codes  football-data.co.uk Div codes used by src/scraper.py
                       for the LIVE upcoming-fixtures feed. Empty list means
                       "no live-fixtures source wired up for this competition
                       yet" — see the note on the UCL entry.

Optional columns on a results CSV, used if present (both default to
False/no-op if absent, so this stays backward compatible with the original
epl_results.csv which has neither):
  stage     round name (e.g. "Q1", "Playoff", "League phase", "Round of 16",
            "Final") — informational only right now, not fed to the model,
            but kept so it's available for future stage-aware features
            (e.g. "is this a one-off knockout leg") without re-fetching data.
  neutral   1/0 (or True/False) — match played at a neutral venue (UCL final).
            elo.py zeroes the home-advantage term for these rows. The
            Dixon-Coles layer (poisson_model.py) does NOT yet do the same —
            it fits a single global home-advantage constant — so lambdas for
            a neutral-venue final are still mildly home-skewed. Worth fixing
            if finals prediction accuracy specifically matters to you; left
            as a known limitation for now rather than a half-tested change.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_HERE), "data")

COMPETITIONS = {
    "epl": {
        "label": "Premier League",
        "results_path": os.path.join(DATA_DIR, "epl_results.csv"),
        "odds_path": os.path.join(DATA_DIR, "epl_odds.csv"),
        "xg_path": os.path.join(DATA_DIR, "epl_xg.csv"),
        "lookback_seasons": None,
        "home_advantage": 60,
        "fixtures_league_codes": ["E0"],
    },
    "ucl": {
        "label": "UEFA Champions League (incl. qualifying)",
        "results_path": os.path.join(DATA_DIR, "ucl_results.csv"),
        "odds_path": None,
        "xg_path": None,
        # Per request: train only on the last 5 completed seasons, not full
        # history. This isn't just "less data is fine" — it's the more
        # correct choice here: the competition format changed for 2024-25
        # (36-team single "league phase" replacing the old 32-team group
        # stage), qualifying-round participants and structure reshuffle most
        # years as UEFA's association coefficients move, and pre-2021 squads
        # for most clubs look very different from today's. Older seasons
        # would mostly add noise, not signal.
        "lookback_seasons": 5,
        "home_advantage": 50,
        # football-data.co.uk (src/scraper.py's live-fixtures source) only
        # covers domestic top flights — it has no UCL feed. Live UCL fixture
        # scraping + odds isn't wired up; /api/fixtures?competition=ucl will
        # report that honestly rather than silently return nothing. Historical
        # training/prediction (/api/predict?competition=ucl) work regardless.
        "fixtures_league_codes": [],
    },
}

DEFAULT_COMPETITION = "epl"


def get_competition(key):
    key = (key or DEFAULT_COMPETITION).lower()
    if key not in COMPETITIONS:
        raise KeyError(
            f"Unknown competition '{key}'. Available: {', '.join(COMPETITIONS)}"
        )
    return key, COMPETITIONS[key]
