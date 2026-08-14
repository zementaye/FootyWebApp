# EPL Betting Model — Investigation Findings

## Starting point
Walk-forward backtest across 25 seasons (2001-02 to 2025-26), goals-only features:
- 4,934 flagged bets, 33.1% hit rate, **-5.73% flat-stake ROI** (95% CI entirely negative)
- Kelly-staked bankroll: $1,000 → **$0.04** (near-total ruin)
- 21.7% of flagged bets beat the closing line (`pct_bets_beating_close`)

## Steps taken, in order

**1. Added Understat xG features** (`download_xg_data.py`, matched 4,522/4,560 rows).
Result: no meaningful change (ROI -5.99%, CLV-beating 21.8%). Ruled out "missing features" as the cause.

**2. Tried pre-written `elo_UPDATED.py` + `ml_model_UPDATED.py`** (home/away-split Elo ratings + season-boundary regression to mean; early-stopping + recency-weighted XGBoost — found unused in the repo).
Result: **made things worse** — hit rate dropped to 25.8%, ROI to -9.62%, Kelly bankroll to $0.00002. Likely cause: splitting Elo into separate home/away ratings halves the effective sample per rating, and combined with aggressive recency weighting, injected more noise than signal (Away bet volume roughly doubled, with win rate cratering to 20.4%). **Reverted.**

**3. Built a calibration diagnostic** (`src/calibration_check.py`) comparing model calibration on ALL predictions vs. only the bets `find_value_bets` flags.
Finding: the model is well-calibrated overall (gap ≈ 0 across H/D/A on all predictions), but **flagged bets are overconfident by ~+7.9 percentage points on average**, growing with predicted probability (up to +12pp in the highest-confidence bucket). This is a **winner's-curse selection bias**: picking the single best edge across 3 outcomes × thousands of matches systematically selects cases where the model's *error*, not its *signal*, pointed favorably — independent of which model or features are used underneath.

**4. Built a post-hoc correction** (`fit_selection_calibrator` / `apply_selection_calibration` in `value_bets.py`): an isotonic regression, fit walk-forward on each selection type's (H/D/A) prior-season flagged-bet history, mapping raw model probability → true historical win rate among flagged bets. Bets are re-checked against `edge_threshold` after correction.
Result:
- n_bets dropped from 4,963 to 2,104 (correctly filtering out noise-driven "value")
- hit rate improved to 34.7%
- **Kelly bankroll survived**: $1,000 → $36 instead of near-zero — the single biggest practical fix
- ROI confidence interval now includes zero (`[-12.3%, +0.4%]`) instead of being entirely negative
- But `pct_bets_beating_close` **dropped** to 15.6% (was 21.8%) — statistically significant, not noise (non-overlapping 95% CIs)

**5. Built a naive baseline check** (`src/clv_baseline_check.py`, no model involved): what fraction of the time does blindly betting every Home/Draw/Away selection at bet365's opening price beat Pinnacle's closing line?
Finding: **naive baseline is 37-39% for every selection type.** The model's flagged bets beat the close at 15.6-21.8% — **less than half the naive rate**, in both the corrected and uncorrected versions.

## Current read

This is the most important finding of the investigation: the model's high-"edge" picks aren't just failing to show a proven edge — they're **moving against the closing line significantly worse than doing nothing (random selection) would**. The most likely explanation is that the model's probability estimates are partly re-discovering public/recreational betting bias (e.g. favoring recognizable strong or home teams) rather than finding real mispricing, and that's exactly the kind of money sharp closing lines are known to correct against.

## Not yet tested
- Whether *fading* the model's picks (betting the opposite of what it flags) shows positive CLV — would help confirm or rule out the "rediscovering public bias" hypothesis
- Whether the bias is concentrated in specific team/market segments (e.g. big-club favorites specifically)
- Whether `CalibratedClassifierCV`'s random (non-time-respecting) 3-fold split is itself introducing subtle leakage or instability within a training window

## Bottom line
No evidence so far of a real, exploitable edge against these bookmakers. Real progress was made on statistical rigor and eliminating a bankroll-ruin bug, but this should be treated as a research/educational project at this stage, not a basis for real-money betting.
