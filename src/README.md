# EPL Value-Betting Backtest

A walk-forward research system testing whether a football-prediction model
(Elo + Dixon-Coles Poisson + XGBoost ensemble) can find profitable value
bets against real EPL bookmaker odds, 2001-02 through 2025-26.

**Read `data/investigation_findings.md` first.** Short version: after seven
separate diagnostic tests, no configuration showed a statistically
defensible edge, for the model's picks or their fade, against any
bookmaker. This is a research/educational project — nothing here supports
betting real money.

## Running it

```
python -m src.backtest              # main walk-forward backtest + fade test
python -m src.calibration_check     # is the model calibrated, or is
                                     # "value bet" selection itself biased?
python -m src.clv_baseline_check    # naive no-model CLV baseline, for context
```

All three need `data/epl_results.csv` and `data/epl_odds.csv`. Run
`python download_xg_data.py` first (needs network + `pip install
understatapi`) to also include xG features — optional, tested and found not
to change the outcome (see findings doc, step 1).

## Layout

- `src/elo.py`, `src/poisson_model.py`, `src/ml_model.py` — the three
  prediction layers actually in use (imported by both `backtest.py` and
  `model_service.py`).
- `src/elo_UPDATED.py`, `src/ml_model_UPDATED.py` — an alternative
  architecture that was tested and made results worse. Kept for reference;
  not imported anywhere. See the warning at the top of each file.
- `src/features.py` — walk-forward-safe feature engineering (rolling form,
  rest days, Dixon-Coles outputs, optional xG form).
- `src/value_bets.py` — turns model probabilities into flagged bets;
  includes the selection-bias calibrator (`fit_selection_calibrator`) and
  the consensus-vs-single-book edge comparison (`market_odds_cols`).
- `src/backtest.py` — the walk-forward simulation, flat + Kelly staking,
  bootstrapped ROI confidence intervals, and the fade test.
- `src/calibration_check.py`, `src/clv_baseline_check.py` — standalone
  diagnostics, not part of the normal run.
- `src/model_service.py` — presumably the live/production prediction path
  (not touched during this investigation).
- `data/backtest_runs_log.jsonl` — append-only log of every backtest run's
  config + headline results, so a good-looking number can always be checked
  against the full run history rather than memory.
