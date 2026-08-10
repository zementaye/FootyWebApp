"""
Baseline check: no model involved at all. Just answers "if you naively bet
Home on every match (or Away on every match) at bet365's opening price, what
fraction of the time would you have beaten Pinnacle's closing price?"

Why this matters: pct_bets_beating_close dropped from 21.8% to 15.6% after
adding selection-bias correction (src/value_bets.py's isotonic calibrator),
and the bet mix shifted toward more Home favorites, fewer Away longshots.
CLV compares two DIFFERENT bookmakers (bet365 open vs Pinnacle close), which
have different margin structures — that gap could plausibly differ by odds
range for purely structural reasons, unrelated to any model's skill. This
script measures that structural baseline directly, with zero modeling, so we
know whether our flagged bets' CLV numbers reflect real skill or just which
odds range (favorites vs longshots) they happen to sit in.

Usage:
    python -m src.clv_baseline_check
"""

import os
import numpy as np
import pandas as pd

from src.backtest import load_backtest_odds


def naive_clv_report(odds_df, take_col, close_col, label):
    sub = odds_df.dropna(subset=[take_col, close_col]).copy()
    sub = sub[(sub[take_col] > 1) & (sub[close_col] > 1)]
    clv_pct = sub[take_col] / sub[close_col] - 1
    beat = (clv_pct > 0).mean()
    n = len(sub)
    se = np.sqrt(beat * (1 - beat) / n)
    lo, hi = beat - 1.96 * se, beat + 1.96 * se
    print(f"{label}: n={n}  avg_clv_pct={clv_pct.mean():+.4f}  "
          f"pct_beating_close={beat:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  "
          f"avg_odds={sub[take_col].mean():.2f}")
    return beat, n


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    odds = load_backtest_odds(os.path.join(HERE, "data", "epl_odds.csv"))

    print("Naive, model-free baseline: always betting this selection at bet365's")
    print("price, checked against Pinnacle's closing price. No predictions involved.\n")

    naive_clv_report(odds, "odds_home", "odds_home_close", "Always bet HOME  ")
    naive_clv_report(odds, "odds_draw", "odds_draw_close", "Always bet DRAW  ")
    naive_clv_report(odds, "odds_away", "odds_away_close", "Always bet AWAY  ")

    print("\nCompare these to the flagged-bet rates from backtest.py's output.")
    print("If a selection's naive baseline here is already close to what the")
    print("flagged bets showed, that selection's CLV number isn't telling you")
    print("much about model skill — it's mostly reflecting bet365-vs-Pinnacle")
    print("structural differences for that odds range.")
