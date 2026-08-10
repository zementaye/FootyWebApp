"""
Feature engineering.

Builds two families of features, both computed walk-forward (only using
information strictly BEFORE the match being predicted, so nothing leaks):

1. Rolling form: each team's points-per-game, goals-for, goals-against
   averaged over their last N matches.
2. Elo difference (from elo.py) and Dixon-Coles match probabilities
   (from poisson_model.py), refit per-season on all prior history.
"""

import pandas as pd
import numpy as np


def add_rolling_form(df, windows=(5, 10)):
    """
    df must have: date, home_team, away_team, fthg, ftag (sorted by date).
    Adds home_form_pts_{w}, home_form_gf_{w}, home_form_ga_{w}, and away_ equivalents,
    for each window size w.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_row_id"] = np.arange(len(df))

    # long format: one row per team per match
    home_rows = df[["match_row_id", "date", "home_team", "fthg", "ftag"]].copy()
    home_rows.columns = ["match_row_id", "date", "team", "gf", "ga"]
    home_rows["points"] = np.select(
        [home_rows["gf"] > home_rows["ga"], home_rows["gf"] == home_rows["ga"]], [3, 1], default=0
    )

    away_rows = df[["match_row_id", "date", "away_team", "ftag", "fthg"]].copy()
    away_rows.columns = ["match_row_id", "date", "team", "gf", "ga"]
    away_rows["points"] = np.select(
        [away_rows["gf"] > away_rows["ga"], away_rows["gf"] == away_rows["ga"]], [3, 1], default=0
    )

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "match_row_id"]).reset_index(drop=True)

    for w in windows:
        grp = long_df.groupby("team")
        # shift(1) first so the current match's own result never leaks into its own features
        long_df[f"form_pts_{w}"] = grp["points"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long_df[f"form_gf_{w}"] = grp["gf"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long_df[f"form_ga_{w}"] = grp["ga"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    feature_cols = [c for c in long_df.columns if c.startswith("form_")]

    home_feats = long_df[["match_row_id", "team"] + feature_cols].copy()
    home_feats.columns = ["match_row_id", "home_team"] + [f"home_{c}" for c in feature_cols]

    away_feats = long_df[["match_row_id", "team"] + feature_cols].copy()
    away_feats.columns = ["match_row_id", "away_team"] + [f"away_{c}" for c in feature_cols]

    df = df.merge(home_feats, on=["match_row_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["match_row_id", "away_team"], how="left")

    for w in windows:
        df[f"form_pts_diff_{w}"] = df[f"home_form_pts_{w}"] - df[f"away_form_pts_{w}"]
        df[f"form_gf_diff_{w}"] = df[f"home_form_gf_{w}"] - df[f"away_form_gf_{w}"]

    return df


def add_rest_days(df, short_rest_days=5, cap_days=30):
    """
    Days of rest each team had going into this match, computed walk-forward
    (only using each team's own match history strictly before this fixture,
    via shift(1) exactly like add_rolling_form).

    CAVEAT: this dataset is EPL-only, so "rest days" here only sees gaps
    between a team's EPL fixtures. It will NOT see midweek Champions League,
    Europa League, or domestic cup matches, since those aren't in
    epl_results.csv. That means it understates fixture congestion for clubs
    playing in Europe or deep cup runs — treat it as a partial, noisy proxy
    for real fixture congestion, not the full signal. Adding a multi-competition
    fixture calendar would make this feature meaningfully stronger.

    A first-ever appearance for a team (no prior match in the data) has no
    rest-days value to compute; it's filled with the dataset's median rest
    (not e.g. 0 or a large number) so it doesn't inject a spurious signal.
    Values are capped at cap_days so summer-break gaps (~70+ days) don't
    dominate the feature's scale.
    """
    df = df.sort_values("date").reset_index(drop=True)
    if "match_row_id" not in df.columns:
        df["match_row_id"] = np.arange(len(df))

    home_rows = df[["match_row_id", "date", "home_team"]].copy()
    home_rows.columns = ["match_row_id", "date", "team"]
    away_rows = df[["match_row_id", "date", "away_team"]].copy()
    away_rows.columns = ["match_row_id", "date", "team"]

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "match_row_id"]).reset_index(drop=True)

    grp = long_df.groupby("team")
    long_df["prev_match_date"] = grp["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_match_date"]).dt.days

    median_rest = long_df["rest_days"].median()
    long_df["rest_days"] = long_df["rest_days"].fillna(median_rest).clip(upper=cap_days)
    long_df["short_rest_flag"] = (long_df["rest_days"] <= short_rest_days).astype(int)

    home_feats = long_df[["match_row_id", "team", "rest_days", "short_rest_flag"]].copy()
    home_feats.columns = ["match_row_id", "home_team", "home_rest_days", "home_short_rest_flag"]
    away_feats = long_df[["match_row_id", "team", "rest_days", "short_rest_flag"]].copy()
    away_feats.columns = ["match_row_id", "away_team", "away_rest_days", "away_short_rest_flag"]

    df = df.merge(home_feats, on=["match_row_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["match_row_id", "away_team"], how="left")
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]

    return df


def add_poisson_features(df, poisson_model, max_goals=8):
    """
    Attach Dixon-Coles match probabilities + expected goals as features.
    Vectorized across all rows at once (numpy broadcasting) instead of a
    per-row Python loop — this is the main cost center in a walk-forward
    backtest since it reruns every season, so it needs to be fast.
    """
    from scipy.stats import poisson as _poisson

    df = df.copy()
    attack = poisson_model.attack
    defense = poisson_model.defense
    home_adv = poisson_model.home_adv
    rho = poisson_model.rho
    league_avg_attack = 0.0  # attack params are constrained to mean 0
    league_avg_defense = np.mean(list(defense.values())) if defense else 0.0

    a_home = df["home_team"].map(attack).fillna(league_avg_attack).values
    d_home = df["home_team"].map(defense).fillna(league_avg_defense).values
    a_away = df["away_team"].map(attack).fillna(league_avg_attack).values
    d_away = df["away_team"].map(defense).fillna(league_avg_defense).values

    lam_home = np.exp(a_home + d_away + home_adv)
    lam_away = np.exp(a_away + d_home)

    goals = np.arange(0, max_goals + 1)
    # ph[i, g] = P(home team scores g goals) for row i
    ph = _poisson.pmf(goals[None, :], lam_home[:, None])
    pa = _poisson.pmf(goals[None, :], lam_away[:, None])

    # outer product per row -> (n_rows, max_goals+1, max_goals+1)
    matrix = ph[:, :, None] * pa[:, None, :]

    # Dixon-Coles low-score adjustment on the four corner cells, vectorized
    tau_00 = 1 - (lam_home * lam_away * rho)
    tau_01 = 1 + (lam_home * rho)
    tau_10 = 1 + (lam_away * rho)
    tau_11 = 1 - rho
    matrix[:, 0, 0] *= tau_00
    matrix[:, 0, 1] *= tau_01
    matrix[:, 1, 0] *= tau_10
    matrix[:, 1, 1] *= tau_11

    row_sums = matrix.sum(axis=(1, 2), keepdims=True)
    matrix = matrix / row_sums

    tri_mask_home = np.tril(np.ones((max_goals + 1, max_goals + 1)), -1).astype(bool)
    tri_mask_away = np.triu(np.ones((max_goals + 1, max_goals + 1)), 1).astype(bool)
    diag_mask = np.eye(max_goals + 1).astype(bool)

    p_home = matrix[:, tri_mask_home].sum(axis=1)
    p_away = matrix[:, tri_mask_away].sum(axis=1)
    p_draw = matrix[:, diag_mask].sum(axis=1)

    df["dc_p_home"] = p_home
    df["dc_p_draw"] = p_draw
    df["dc_p_away"] = p_away
    df["dc_lambda_home"] = lam_home
    df["dc_lambda_away"] = lam_away
    return df


FEATURE_COLUMNS_BASE = [
    "elo_diff",
    "dc_p_home", "dc_p_draw", "dc_p_away",
    "dc_lambda_home", "dc_lambda_away",
    "form_pts_diff_5", "form_gf_diff_5",
    "form_pts_diff_10", "form_gf_diff_10",
    "home_form_ga_5", "away_form_ga_5",
    "rest_days_diff", "home_short_rest_flag", "away_short_rest_flag",
]


def add_rolling_xg_form(df, xg_by_match_id, windows=(5, 10)):
    """
    Rolling xG-for / xG-against form per team (walk-forward, same shift(1)-
    before-rolling pattern as add_rolling_form), plus a "finishing
    overperformance" feature: actual goals scored minus xG for, averaged
    over the last N matches. Persistent overperformance is a well-known
    proxy for finishing quality that tends to partially mean-revert — worth
    giving the model as its own signal rather than assuming raw goals fully
    capture attacking strength (which is the whole motivation for adding xG
    at all).

    xg_by_match_id: DataFrame [match_id, home_xg, away_xg] — the output of
    src.xg_source.match_to_results(). Understat's EPL coverage starts
    2014-15, so most of this project's pre-2014 history (and any unmatched
    rows) will have no xG here.

    Matches with no xG get NEUTRAL FILL (0.0) for the diff/overperformance
    columns and has_xg_data=0, rather than being dropped — dropping them
    would silently discard ~20 seasons of pre-xG training history the first
    time these columns are added to feature_cols. has_xg_data lets the model
    learn to weight the xG features only on rows where they're real.
    """
    df = df.merge(xg_by_match_id, on="match_id", how="left")
    if "match_row_id" not in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        df["match_row_id"] = np.arange(len(df))

    home_rows = df[["match_row_id", "date", "home_team", "fthg", "home_xg", "away_xg"]].copy()
    home_rows.columns = ["match_row_id", "date", "team", "goals_for", "xg_for", "xg_against"]
    away_rows = df[["match_row_id", "date", "away_team", "ftag", "away_xg", "home_xg"]].copy()
    away_rows.columns = ["match_row_id", "date", "team", "goals_for", "xg_for", "xg_against"]

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "match_row_id"]).reset_index(drop=True)
    long_df["overperf"] = long_df["goals_for"] - long_df["xg_for"]

    grp = long_df.groupby("team")
    for w in windows:
        long_df[f"xg_for_{w}"] = grp["xg_for"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long_df[f"xg_overperf_{w}"] = grp["overperf"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    feat_cols = [c for c in long_df.columns if c.startswith("xg_")]
    home_feats = long_df[["match_row_id", "team"] + feat_cols].copy()
    home_feats.columns = ["match_row_id", "home_team"] + [f"home_{c}" for c in feat_cols]
    away_feats = long_df[["match_row_id", "team"] + feat_cols].copy()
    away_feats.columns = ["match_row_id", "away_team"] + [f"away_{c}" for c in feat_cols]

    df = df.merge(home_feats, on=["match_row_id", "home_team"], how="left")
    df = df.merge(away_feats, on=["match_row_id", "away_team"], how="left")

    for w in windows:
        df[f"xg_form_diff_{w}"] = df[f"home_xg_for_{w}"] - df[f"away_xg_for_{w}"]

    df["has_xg_data"] = df["home_xg"].notna().astype(int)

    fill_cols = ([f"xg_form_diff_{w}" for w in windows]
                 + [f"home_xg_overperf_{w}" for w in windows]
                 + [f"away_xg_overperf_{w}" for w in windows])
    df[fill_cols] = df[fill_cols].fillna(0.0)

    return df


FEATURE_COLUMNS_XG = [
    "xg_form_diff_5", "xg_form_diff_10",
    "home_xg_overperf_5", "away_xg_overperf_5",
    "has_xg_data",
]
