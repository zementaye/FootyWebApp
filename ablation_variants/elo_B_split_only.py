"""
Elo rating engine for football teams. VARIANT B: separate home/away ratings
ONLY — no season-boundary regression. Isolates the home/away-split effect.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class EloRatings:
    def __init__(self, k=20, initial_rating=1500, mov_multiplier=True):
        self.k = k
        self.initial_rating = initial_rating
        self.mov_multiplier = mov_multiplier
        self.home_ratings = defaultdict(lambda: initial_rating)
        self.away_ratings = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _mov_factor(self, goal_diff, elo_diff):
        if not self.mov_multiplier:
            return 1.0
        goal_diff = max(abs(goal_diff), 1)
        return np.log(goal_diff + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team, away_team, home_goals, away_goals):
        home_rating = self.home_ratings[home_team]
        away_rating = self.away_ratings[away_team]
        pre_home, pre_away = home_rating, away_rating

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
        return self.home_ratings[team] if venue == "home" else self.away_ratings[team]


def compute_elo_features(df, k=20, initial_rating=1500, mov_multiplier=True):
    df = df.sort_values("date").reset_index(drop=True)
    elo = EloRatings(k=k, initial_rating=initial_rating, mov_multiplier=mov_multiplier)

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
