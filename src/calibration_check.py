"""
Calibration diagnostic — run this AFTER reverting backtest.py to the plain
(non-UPDATED) elo/ml_model imports.

Three backtest variants (goals-only, +xG, +updated Elo/ML) all show
pct_bets_beating_close stuck around ~22%, regardless of what changed in the
model or features. That pattern points away from "the model is bad" and
toward something structural in what happens AFTER the model produces
probabilities — specifically the value-bet selection step in value_bets.py,
which picks the SINGLE BEST edge across 3 outcomes (H/D/A) for every match.

Even a reasonably calibrated model, when you pick the argmax of
(model_prob - market_prob) across 3 outcomes x thousands of matches, will
systematically select the cases where the model's ERROR happened to point
most favorably — a "winner's curse" / selection bias, well known in
quant/betting contexts. This script tests that directly by comparing:

  (A) calibration across ALL predictions (every match, every outcome)
  (B) calibration restricted to ONLY the bets find_value_bets actually flagged

If (A) looks reasonably calibrated (predicted prob ~= actual frequency per
bucket) but (B) is badly overconfident (predicted prob >> actual frequency),
that CONFIRMS the selection-bias hypothesis: the model itself is fine, but
"take the best edge across 3 outcomes" is structurally biased toward
overconfident picks. That would explain why swapping models/features never
moved pct_bets_beating_close.

If (A) is ALSO badly calibrated, the problem is more basic (the model's
probabilities are wrong even before selection), and we should look at
CalibratedClassifierCV / the base XGBoost fit instead.

Usage:
    python -m src.calibration_check
"""

import os
import numpy as np
import pandas as pd

from src.elo import compute_elo_features
from src.poisson_model import DixonColesModel
from src.features import (add_rolling_form, add_rest_days, add_rolling_xg_form,
                           add_poisson_features, FEATURE_COLUMNS_BASE, FEATURE_COLUMNS_XG)
from src.ml_model import train_model, predict_probabilities
from src.value_bets import find_value_bets, implied_probabilities, remove_overround
from src.backtest import load_backtest_odds


def brier_score(probs, outcomes):
    """Multi-class Brier score: mean squared error between predicted prob
    vector and one-hot actual outcome, averaged over classes and rows.
    Lower is better; a model predicting the base rate everywhere gets a
    'trivial' Brier score to compare against."""
    return np.mean(np.sum((probs - outcomes) ** 2, axis=1))


def calibration_table(pred_probs, actual_hits, n_bins=10, label=""):
    """
    pred_probs: array of predicted probabilities for the outcome being checked
    actual_hits: array of 0/1, whether that outcome actually happened
    Returns a DataFrame with one row per probability bucket: mean predicted
    prob, actual hit rate, and count, so miscalibration is visible directly.
    """
    df = pd.DataFrame({"pred": pred_probs, "hit": actual_hits})
    df = df.dropna()
    df["bucket"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")
    table = df.groupby("bucket", observed=True).agg(
        n=("hit", "size"),
        mean_predicted=("pred", "mean"),
        actual_hit_rate=("hit", "mean"),
    ).reset_index(drop=True)
    table["gap"] = table["mean_predicted"] - table["actual_hit_rate"]
    print(f"\n--- Calibration: {label} (n={len(df)}) ---")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    overall_gap = (df["pred"].mean() - df["hit"].mean())
    print(f"Overall: mean predicted={df['pred'].mean():.4f}  "
          f"actual rate={df['hit'].mean():.4f}  gap={overall_gap:+.4f}")
    return table


def run(min_train_seasons=8, max_train_seasons=12, edge_threshold=0.03,
        min_odds=1.3, max_odds=8.0, exclude_draws=True):
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = pd.read_csv(os.path.join(HERE, "data", "epl_results.csv"), parse_dates=["date"])
    odds_df = load_backtest_odds(os.path.join(HERE, "data", "epl_odds.csv"))

    xg_path = os.path.join(HERE, "data", "epl_xg.csv")
    xg_by_match_id = pd.read_csv(xg_path) if os.path.exists(xg_path) else None
    use_xg = xg_by_match_id is not None and len(xg_by_match_id) > 0

    df = results.sort_values("date").reset_index(drop=True)
    df, _ = compute_elo_features(df)
    df = add_rolling_form(df)
    df = add_rest_days(df)
    if use_xg:
        df = add_rolling_xg_form(df, xg_by_match_id)
    feature_cols = FEATURE_COLUMNS_BASE + (FEATURE_COLUMNS_XG if use_xg else [])

    odds_cols = ["match_id", "odds_home", "odds_draw", "odds_away"]
    df = df.merge(odds_df[odds_cols], on="match_id", how="left")

    seasons = sorted(df["season"].unique())
    test_seasons = seasons[min_train_seasons:]

    all_preds = []   # every match, every outcome -> for the "all predictions" calibration check
    all_flagged = [] # only matches where find_value_bets flagged a bet -> "selected bets" check

    for si, season in enumerate(test_seasons):
        print(f"[{si+1}/{len(test_seasons)}] processing season {season}...", flush=True)
        season_idx = seasons.index(season)
        train_seasons_window = seasons[max(0, season_idx - max_train_seasons):season_idx]
        train_mask = df["season"].isin(train_seasons_window)
        test_mask = df["season"] == season
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if train_df["fthg"].isna().any():
            train_df = train_df.dropna(subset=["fthg", "ftag"])

        dc_model = DixonColesModel(xi=0.0018)
        dc_model.fit(train_df)
        train_df = add_poisson_features(train_df, dc_model)
        test_df = add_poisson_features(test_df, dc_model)

        train_df = train_df.dropna(subset=feature_cols + ["ftr"])
        test_df_valid = test_df.dropna(subset=feature_cols).copy()
        if len(train_df) < 200 or len(test_df_valid) == 0:
            continue

        model = train_model(train_df, feature_cols)
        probs = predict_probabilities(model, test_df_valid, feature_cols)
        test_df_valid = pd.concat([test_df_valid.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

        # --- (A) all predictions, all 3 outcomes, no selection ---
        valid_odds = test_df_valid.dropna(subset=["odds_home", "odds_draw", "odds_away"])
        for _, row in valid_odds.iterrows():
            all_preds.append({"p_home": row["p_home"], "p_draw": row["p_draw"], "p_away": row["p_away"],
                               "ftr": row["ftr"]})

        # --- (B) only the matches find_value_bets actually flags ---
        flagged = find_value_bets(
            test_df_valid, model_prob_cols=("p_home", "p_draw", "p_away"),
            odds_cols=("odds_home", "odds_draw", "odds_away"),
            edge_threshold=edge_threshold, min_odds=min_odds, max_odds=max_odds,
        )
        if exclude_draws:
            flagged.loc[flagged["bet_selection"] == "D", "bet_selection"] = None
        flagged = flagged.dropna(subset=["bet_selection"])
        for _, row in flagged.iterrows():
            won = int(row["bet_selection"] == row["ftr"])
            all_flagged.append({"model_prob": row["model_prob"], "won": won,
                                 "bet_selection": row["bet_selection"]})

    # ---- (A) calibration on ALL predictions, per outcome class ----
    preds_df = pd.DataFrame(all_preds)
    for outcome, col, label in [("H", "p_home", "Home win prob"),
                                 ("D", "p_draw", "Draw prob"),
                                 ("A", "p_away", "Away win prob")]:
        hit = (preds_df["ftr"] == outcome).astype(int)
        calibration_table(preds_df[col].values, hit.values, label=f"ALL matches — {label}")

    probs_matrix = preds_df[["p_home", "p_draw", "p_away"]].values
    outcomes_matrix = pd.get_dummies(preds_df["ftr"])[["H", "D", "A"]].values
    print(f"\nBrier score, ALL predictions (lower=better): {brier_score(probs_matrix, outcomes_matrix):.4f}")

    # ---- (B) calibration on ONLY the flagged/selected bets ----
    flagged_df = pd.DataFrame(all_flagged)
    calibration_table(flagged_df["model_prob"].values, flagged_df["won"].values,
                       label="SELECTED bets only (find_value_bets picks)")

    # ---- (B) split further by selection to see if it's specific to Away ----
    for sel in ["H", "A"]:
        sub = flagged_df[flagged_df["bet_selection"] == sel]
        if len(sub) > 0:
            calibration_table(sub["model_prob"].values, sub["won"].values,
                               n_bins=6, label=f"SELECTED bets only — {sel} selection")

    print("\n=== Read this as: ===")
    print("If ALL-matches calibration looks close (small gaps) but SELECTED-bets")
    print("calibration shows predicted >> actual (large positive gap), that confirms")
    print("the value-bet selection step itself is the problem (winner's-curse bias),")
    print("not the underlying model.")


if __name__ == "__main__":
    run()
