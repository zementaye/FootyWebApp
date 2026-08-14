"""
Elo rating engine for football teams. VARIANT A: season-boundary regression to
the mean ONLY — single pooled rating + fixed home_advantage constant (original
structure), unchanged home/away Elo. Isolates the regression-to-mean effect.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class EloRatings:
    def __init__(self, k=20, home_advantage=60, initial_rating=1500, mov_multiplier=True, regress_frac=1 / 3):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.mov_multiplier = mov_multiplier
        self.regress_frac = regress_frac
        self.ratings = defaultdict(lambda: initial_rating)

    def expected_score(self, rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _mov_factor(self, goal_diff, elo_diff):
        if not self.mov_multiplier:
            return 1.0
        goal_diff = max(abs(goal_diff), 1)
        return np.log(goal_diff + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team, away_team, home_goals, away_goals):
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

    def regress_to_mean(self, frac=None):
        frac = self.regress_frac if frac is None else frac
        if frac <= 0 or not self.ratings:
            return
        mean_rating = np.mean(list(self.ratings.values()))
        for team in list(self.ratings.keys()):
            self.ratings[team] = self.ratings[team] + frac * (mean_rating - self.ratings[team])


def compute_elo_features(df, k=20, home_advantage=60, regress_frac=1 / 3):
    df = df.sort_values("date").reset_index(drop=True)
    elo = EloRatings(k=k, home_advantage=home_advantage, regress_frac=regress_frac)

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
