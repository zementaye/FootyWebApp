"""
Value-bet detection.

This is the module that actually matters for profitability. A model that
predicts football well is not the same as a model that beats the market —
what you need is cases where YOUR probability estimate is meaningfully
higher than what the odds imply.

Bookmaker odds always contain a margin ("overround" / "vig"), typically
5-8% for major leagues on 1X2 markets. We strip that out to get the
market's true implied probability before comparing to our model.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def implied_probabilities(odds_home, odds_draw, odds_away):
    """Raw implied probabilities from odds (will sum to > 1 due to bookmaker margin)."""
    return 1 / odds_home, 1 / odds_draw, 1 / odds_away


def remove_overround(p_home, p_draw, p_away):
    """Normalize implied probabilities to sum to 1 (removes the bookmaker's margin)."""
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def find_value_bets(df, model_prob_cols=("p_home", "p_draw", "p_away"),
                     odds_cols=("odds_home", "odds_draw", "odds_away"),
                     edge_threshold=0.03, min_odds=1.3, max_odds=10.0,
                     market_odds_cols=None):
    """
    For each match, compare model probabilities to market-implied probabilities
    (overround removed) across H/D/A. Flags a bet where:
      - model_prob - market_implied_prob > edge_threshold, AND
      - expected value using the ACTUAL (un-normalized) odds is positive, AND
      - odds are within a sane range (extreme odds are noisy / high variance)

    odds_cols: the EXECUTION price — what you'd actually be paid if the bet wins.
        Always used for ev, odds_taken, and the min_odds/max_odds sanity check.
    market_odds_cols: the REFERENCE price used to compute market-implied
        probability for the edge/threshold decision. Defaults to odds_cols
        (original behavior: single book used for both). Pass a broader
        consensus (e.g. market-average odds) here to test whether the model
        has real insight relative to the wider market, while still executing
        at whatever book's price odds_cols points to. This directly tests
        for single-book noise vs. genuine edge — see calibration_check.py /
        clv_baseline_check.py for the diagnostics that motivated this split.

    Returns df with added columns: bet_selection, edge, ev, model_prob, odds_taken.
    Rows with no qualifying bet get bet_selection = None.
    """
    df = df.copy()
    outcomes = ["home", "draw", "away"]
    labels = {"home": "H", "draw": "D", "away": "A"}
    market_odds_cols = market_odds_cols or odds_cols

    results = {"bet_selection": [], "edge": [], "ev": [], "model_prob": [], "odds_taken": []}

    for _, row in df.iterrows():
        odds = [row[c] for c in odds_cols]
        market_odds = [row[c] for c in market_odds_cols]
        probs = [row[c] for c in model_prob_cols]

        if any(pd.isna(odds)) or any(pd.isna(market_odds)) or any(pd.isna(probs)):
            results["bet_selection"].append(None)
            results["edge"].append(np.nan)
            results["ev"].append(np.nan)
            results["model_prob"].append(np.nan)
            results["odds_taken"].append(np.nan)
            continue

        # edge is measured against market_odds (the reference/consensus price);
        # ev, odds_taken, and the sanity range always use odds (the execution price)
        raw_implied = implied_probabilities(*market_odds)
        norm_implied = remove_overround(*raw_implied)

        best_edge, best_i = -np.inf, None
        for i in range(3):
            edge = probs[i] - norm_implied[i]
            if edge > best_edge:
                best_edge, best_i = edge, i

        o = odds[best_i]
        p = probs[best_i]
        ev = p * o - 1  # expected value per unit staked, using ACTUAL odds

        qualifies = (
            best_edge > edge_threshold
            and ev > 0
            and min_odds <= o <= max_odds
        )

        if qualifies:
            results["bet_selection"].append(labels[outcomes[best_i]])
            results["edge"].append(best_edge)
            results["ev"].append(ev)
            results["model_prob"].append(p)
            results["odds_taken"].append(o)
        else:
            results["bet_selection"].append(None)
            results["edge"].append(best_edge)
            results["ev"].append(ev)
            results["model_prob"].append(p)
            results["odds_taken"].append(o)

    for k, v in results.items():
        df[k] = v

    return df


def fit_selection_calibrator(model_probs, won, min_samples=300):
    """
    Fits an isotonic regression mapping a FLAGGED bet's raw model_prob to its
    true empirical win rate, using only bets that already cleared the edge
    threshold (not all predictions).

    Why this is needed: calibration_check.py confirmed that this model's raw
    probabilities are well-calibrated on ALL predictions (gap ~0), but badly
    overconfident specifically on the subset find_value_bets flags (gap
    ~+8pp, growing with predicted probability). Picking the single best edge
    across 3 outcomes for thousands of matches is a winner's-curse selection:
    among matches where the model's error happened to point favorably, some
    get flagged as "value" even though the true edge was smaller or absent.
    This calibrator corrects for that selection effect specifically, rather
    than re-tuning the underlying model (which calibration_check.py showed
    is not where the miscalibration lives).

    model_probs, won: arrays of raw model_prob and 0/1 actual outcome, from
    HISTORICAL flagged bets only (prior seasons in a walk-forward context —
    never include the season being predicted).

    Returns None if there isn't enough history yet to fit reliably (e.g. the
    first few seasons of a walk-forward backtest) — caller should skip
    recalibration in that case rather than force a fit on too little data.
    """
    model_probs = np.asarray(model_probs)
    won = np.asarray(won)
    if len(model_probs) < min_samples:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model_probs, won)
    return iso


def apply_selection_calibration(model_probs, calibrator):
    """Maps raw model_prob -> recalibrated probability via a fitted
    selection-bet calibrator. calibrator=None (not enough history yet)
    returns the input unchanged."""
    model_probs = np.asarray(model_probs, dtype=float)
    if calibrator is None:
        return model_probs
    return calibrator.predict(model_probs)


def kelly_fraction(prob, odds, fraction=1.0):
    """
    Kelly criterion stake as a fraction of bankroll.
    fraction < 1.0 applies 'fractional Kelly' (e.g. 0.25 = quarter-Kelly),
    which is what most disciplined bettors use in practice since full Kelly
    is extremely volatile under any model uncertainty.
    """
    b = odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    return max(0.0, f) * fraction
