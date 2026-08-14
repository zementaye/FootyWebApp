"""
⚠️ TESTED AND REJECTED — see data/investigation_findings.md, step 2.
Swapping this in for src/elo.py (paired with ml_model_UPDATED.py) made the
walk-forward backtest measurably WORSE: hit rate 32.8%→25.8%, ROI -6.0%→-9.6%,
Kelly bankroll $0.015→$0.00002. Likely cause: splitting each team's rating
into separate home/away ratings halves the effective sample per estimate,
and combined with ml_model_UPDATED's recency weighting, injected noise
rather than signal (Away bet volume roughly doubled, win rate cratered).
Kept in the repo for reference / in case a future fix addresses the root
cause, but src/backtest.py and src/model_service.py both intentionally
import the plain src/elo.py, not this file. Don't re-enable without
re-running the backtest and confirming it actually helps.

Elo rating engine for football teams.

Standard chess-style Elo adapted for football, with two upgrades over the
vanilla version:

- Separate home/away ratings per team, instead of one rating plus a fixed
  +60 home-advantage constant. Some teams have genuinely bigger home/away
  splits than the league-average bonus captures, so this lets each team's
  home strength and away strength be estimated independently.
- Regression to the mean at season boundaries. Squads turn over 20-30%
  every summer (transfers, promotions/relegations, managerial changes), so
  carrying a rating straight from the last match of one season into the
  first match of the next overstates how much last season's form tells you
  about a team's current strength. Standard practice (e.g. 538's SPI) is to
  shrink each team a fraction of the way back to the league-mean rating at
  every season boundary; this does that for both the home and away ratings.

Other mechanics, unchanged:
- Result mapped to actual score: Win=1, Draw=0.5, Loss=0
- K-factor scaled by goal difference (bigger wins move ratings more)
- Ratings are computed walk-forward (only using information available before
  each match), so they can be safely used as pre-match features with no
  lookahead leakage.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class EloRatings:
    def __init__(self, k=20, initial_rating=1500, mov_multiplier=True, regress_frac=1 / 3):
        """
        k: base K-factor (rating points moved per "full" upset).
        initial_rating: rating assigned to a team with no history yet, and
            the target that regress_to_mean() pulls ratings toward when the
            league itself has too few teams rated to have a stable mean.
        mov_multiplier: whether to scale K by margin of victory (538 SPI-style).
        regress_frac: fraction of the gap back to the league-mean rating that
            each team's rating closes at every season boundary. 1/3 is the
            standard SPI-style shrinkage; 0 disables regression entirely.
        """
        self.k = k
        self.initial_rating = initial_rating
        self.mov_multiplier = mov_multiplier
        self.regress_frac = regress_frac
        # separate home-venue and away-venue ratings per team
        self.home_ratings = defaultdict(lambda: initial_rating)
        self.away_ratings = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _mov_factor(self, goal_diff, elo_diff):
        """Margin-of-victory multiplier (based on the FiveThirtyEight SPI approach)."""
        if not self.mov_multiplier:
            return 1.0
        goal_diff = max(abs(goal_diff), 1)
        return np.log(goal_diff + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update ratings after a match. Returns pre-match (home_rating, away_rating)
        for feature use — the home team's home-venue rating and the away team's
        away-venue rating, i.e. what each side brings to this specific fixture."""
        home_rating = self.home_ratings[home_team]
        away_rating = self.away_ratings[away_team]

        pre_home, pre_away = home_rating, away_rating

        # no separate home_advantage term: it's now implicit in each team
        # having its own home-venue rating vs. away-venue rating
        exp_home = self.expected_score(home_rating, away_rating)
        exp_away = 1 - exp_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = home_goals - away_goals
        elo_diff = home_rating - away_rating
        mov = self._mov_factor(goal_diff, elo_diff)

        self.home_ratings[home_team] = home_rating + self.k * mov * (actual_home - exp_home)
        self.away_ratings[away_team] = away_rating + self.k * mov * (actual_away - exp_away)

        return pre_home, pre_away

    def get_rating(self, team, venue="home"):
        """venue: 'home' returns the team's home-venue rating, 'away' its away-venue rating."""
        return self.home_ratings[team] if venue == "home" else self.away_ratings[team]

    def regress_to_mean(self, frac=None):
        """
        Shrink every team's rating a fraction of the way back to the current
        league-mean rating. Call this once at each season boundary (after the
        last match of season N, before the first match of season N+1) — not
        mid-season, or it would leak the "squads have turned over" prior into
        ratings that are still describing the season actually in progress.
        Home and away ratings are regressed toward their own separate means.
        """
        frac = self.regress_frac if frac is None else frac
        if frac <= 0:
            return
        for ratings in (self.home_ratings, self.away_ratings):
            if not ratings:
                continue
            mean_rating = np.mean(list(ratings.values()))
            for team in list(ratings.keys()):
                ratings[team] = ratings[team] + frac * (mean_rating - ratings[team])


def compute_elo_features(df, k=20, initial_rating=1500, mov_multiplier=True, regress_frac=1 / 3):
    """
    Walk forward through matches (must be sorted by date) and attach
    pre-match Elo ratings for home/away teams as new columns.
    No future information leaks into any row.

    If df has a "season" column, ratings are regressed toward the league
    mean (see EloRatings.regress_to_mean) at every season boundary, before
    that season's first match is rated.
    """
    df = df.sort_values("date").reset_index(drop=True)
    elo = EloRatings(k=k, initial_rating=initial_rating, mov_multiplier=mov_multiplier,
                      regress_frac=regress_frac)

    has_season = "season" in df.columns
    current_season = None

    home_elos, away_elos = [], []
    for _, row in df.iterrows():
        if has_season:
            season = row["season"]
            if current_season is not None and season != current_season:
                elo.regress_to_mean()
            current_season = season

        pre_home, pre_away = elo.update(
            row["home_team"], row["away_team"], row["fthg"], row["ftag"]
        )
        home_elos.append(pre_home)
        away_elos.append(pre_away)

    df["elo_home"] = home_elos
    df["elo_away"] = away_elos
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, elo
