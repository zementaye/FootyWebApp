"""
Elo rating engine for football teams.

Standard chess-style Elo adapted for football:
- Home advantage bonus added to home team's rating before computing expected score
- Result mapped to actual score: Win=1, Draw=0.5, Loss=0
- K-factor scaled by goal difference (bigger wins move ratings more)
- Ratings are computed walk-forward (only using information available before each match),
  so they can be safely used as pre-match features with no lookahead leakage.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class EloRatings:
    def __init__(self, k=20, home_advantage=60, initial_rating=1500, mov_multiplier=True):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_multiplier = mov_multiplier
        self.ratings = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _mov_factor(self, goal_diff, elo_diff):
        """Margin-of-victory multiplier (based on the FiveThirtyEight SPI approach)."""
        if not self.mov_multiplier:
            return 1.0
        goal_diff = max(abs(goal_diff), 1)
        return np.log(goal_diff + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update ratings after a match. Returns pre-match ratings (for feature use)."""
        home_rating = self.ratings[home_team]
        away_rating = self.ratings[away_team]

        pre_home, pre_away = home_rating, away_rating

        exp_home = self.expected_score(home_rating + self.home_advantage, away_rating)
        exp_away = 1 - exp_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals < away_goals:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = home_goals - away_goals
        elo_diff = (home_rating + self.home_advantage) - away_rating
        mov = self._mov_factor(goal_diff, elo_diff)

        self.ratings[home_team] = home_rating + self.k * mov * (actual_home - exp_home)
        self.ratings[away_team] = away_rating + self.k * mov * (actual_away - exp_away)

        return pre_home, pre_away

    def get_rating(self, team):
        return self.ratings[team]


def compute_elo_features(df, k=20, home_advantage=60):
    """
    Walk forward through matches (must be sorted by date) and attach
    pre-match Elo ratings for home/away teams as new columns.
    No future information leaks into any row.
    """
    df = df.sort_values("date").reset_index(drop=True)
    elo = EloRatings(k=k, home_advantage=home_advantage)

    home_elos, away_elos = [], []
    for _, row in df.iterrows():
        pre_home, pre_away = elo.update(
            row["home_team"], row["away_team"], row["fthg"], row["ftag"]
        )
        home_elos.append(pre_home)
        away_elos.append(pre_away)

    df["elo_home"] = home_elos
    df["elo_away"] = away_elos
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, elo
