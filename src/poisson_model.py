"""
Dixon-Coles model (Dixon & Coles, 1997) — the classic statistical approach
to football score prediction.

Idea: home goals ~ Poisson(lambda_home), away goals ~ Poisson(lambda_away), where
    lambda_home = exp(attack_home + defense_away + home_advantage)
    lambda_away = exp(attack_away + defense_home)

Each team gets an attack and defense rating fit by maximum likelihood on
historical results. A low-score correlation adjustment (rho) corrects for the
fact that 0-0, 1-0, 0-1, 1-1 scorelines are more/less common than plain
independent Poisson predicts.

Ratings are fit with an optional time-decay so recent matches count more —
this matters a lot for football since team strength genuinely drifts season
to season (transfers, managers, injuries).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def dc_adjustment(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Low-score correlation adjustment (tau function from the Dixon-Coles paper)."""
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    elif home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    elif home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0


class DixonColesModel:
    def __init__(self, xi=0.0018):
        """
        xi: time-decay rate. Higher = faster-forgetting of old matches.
            0.0018 roughly halves a match's weight after ~1 season (per Dixon-Coles).
        """
        self.xi = xi
        self.teams = None
        self.params = None
        self.rho = 0.0
        self.home_adv = 0.0

    def _weight(self, days_ago):
        return np.exp(-self.xi * days_ago)

    def fit(self, df, date_col="date", home_col="home_team", away_col="away_team",
            hg_col="fthg", ag_col="ftag"):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        max_date = df[date_col].max()
        df["days_ago"] = (max_date - df[date_col]).dt.days
        df["weight"] = self._weight(df["days_ago"])

        self.teams = sorted(set(df[home_col]) | set(df[away_col]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        # param vector: [attack_1..attack_n, defense_1..defense_n, home_adv, rho]
        # constraint: mean(attack) = 0 for identifiability
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.0]])

        home_idx = df[home_col].map(team_idx).values
        away_idx = df[away_col].map(team_idx).values
        hg = df[hg_col].values
        ag = df[ag_col].values
        w = df["weight"].values

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] + defense[home_idx])

            log_lik = poisson.logpmf(hg, lam_home) + poisson.logpmf(ag, lam_away)

            # low-score correlation adjustment (Dixon-Coles tau), fully vectorized:
            # only the 4 corner scorelines (0-0, 0-1, 1-0, 1-1) get adjusted, else factor = 1
            adj = np.ones_like(lam_home)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            adj[m00] = 1 - (lam_home[m00] * lam_away[m00] * rho)
            adj[m01] = 1 + (lam_home[m01] * rho)
            adj[m10] = 1 + (lam_away[m10] * rho)
            adj[m11] = 1 - rho
            adj = np.clip(adj, 1e-6, None)  # avoid log(negative/0) from extreme rho
            log_lik += np.log(adj)

            return -np.sum(w * log_lik)

        constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-2, 2)] + [(-0.3, 0.3)]

        result = minimize(
            neg_log_likelihood, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 80, "ftol": 1e-4},
        )

        self.params = result.x
        self.attack = dict(zip(self.teams, self.params[:n]))
        self.defense = dict(zip(self.teams, self.params[n:2 * n]))
        self.home_adv = self.params[2 * n]
        self.rho = self.params[2 * n + 1]
        self.fit_success = result.success
        return self

    def team_ratings(self):
        return pd.DataFrame({
            "team": self.teams,
            "attack": [self.attack[t] for t in self.teams],
            "defense": [self.defense[t] for t in self.teams],
        }).sort_values("attack", ascending=False)

    def predict_lambdas(self, home_team, away_team):
        """Returns (lambda_home, lambda_away) — expected goals for each side."""
        # unseen teams (e.g. promoted club with no history) fall back to league-average rating
        a_home = self.attack.get(home_team, 0.0)
        d_home = self.defense.get(home_team, 0.0)
        a_away = self.attack.get(away_team, 0.0)
        d_away = self.defense.get(away_team, 0.0)

        lam_home = np.exp(a_home + d_away + self.home_adv)
        lam_away = np.exp(a_away + d_home)
        return lam_home, lam_away

    def score_matrix(self, home_team, away_team, max_goals=10):
        lam_home, lam_away = self.predict_lambdas(home_team, away_team)
        hg = np.arange(0, max_goals + 1)
        ag = np.arange(0, max_goals + 1)
        ph = poisson.pmf(hg, lam_home)
        pa = poisson.pmf(ag, lam_away)
        matrix = np.outer(ph, pa)

        # apply Dixon-Coles adjustment to the low-scoring cells
        for i in range(2):
            for j in range(2):
                matrix[i, j] *= dc_adjustment(i, j, lam_home, lam_away, self.rho)

        matrix /= matrix.sum()  # renormalize after adjustment
        return matrix

    def match_probabilities(self, home_team, away_team, max_goals=10):
        """Returns (P(home win), P(draw), P(away win))."""
        matrix = self.score_matrix(home_team, away_team, max_goals)
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        return p_home, p_draw, p_away

    def over_under_probability(self, home_team, away_team, line=2.5, max_goals=10):
        matrix = self.score_matrix(home_team, away_team, max_goals)
        total_goals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
        p_over = matrix[total_goals > line].sum()
        p_under = matrix[total_goals < line].sum()
        return p_over, p_under
