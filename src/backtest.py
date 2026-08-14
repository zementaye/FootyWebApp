"""
Walk-forward backtest.

This is the honesty check for the whole system. It simulates what would
have actually happened if you'd used this model to place bets, season by
season, using ONLY information available before each season started:

  For each test season S:
    1. Fit Dixon-Coles Poisson model on all matches strictly before S
    2. Compute Elo ratings walk-forward up through end of season S-1
    3. Train the XGBoost layer on all fully-completed seasons before S
    4. Generate predictions + value bets for every match in season S
    5. Simulate staking (flat stake and fractional-Kelly) and track bankroll

No information from season S or later ever touches training in that iteration.
This is the only way a backtest number means anything.
"""

import numpy as np
import pandas as pd

from src.elo import compute_elo_features
from src.poisson_model import DixonColesModel
from src.features import (add_rolling_form, add_rest_days, add_rolling_xg_form,
                           add_poisson_features, FEATURE_COLUMNS_BASE, FEATURE_COLUMNS_XG)
from src.ml_model import train_model, predict_probabilities
from src.value_bets import (find_value_bets, kelly_fraction,
                             fit_selection_calibrator, apply_selection_calibration)


def run_walk_forward_backtest(df, odds_df, xg_by_match_id=None, min_train_seasons=8, max_train_seasons=12,
                                edge_threshold=0.03, stake_flat=1.0, min_odds=1.3, max_odds=8.0,
                                kelly_frac=0.25, starting_bankroll=1000.0,
                                exclude_draws=True, max_kelly_stake_pct=0.05,
                                use_selection_calibration=True, calibration_min_samples=300):
    """
    df: results with columns [date, season, home_team, away_team, fthg, ftag, ftr]
    odds_df: same matches with [match_id, odds_home, odds_draw, odds_away] (e.g. bet365 closing odds)
    xg_by_match_id: optional [match_id, home_xg, away_xg] (src.xg_source.match_to_results output).
        When given, adds FEATURE_COLUMNS_XG on top of FEATURE_COLUMNS_BASE. When omitted (default),
        behaves exactly as before — this is opt-in, not a silent behavior change.
    exclude_draws: skip staking on draw selections. Defaults to True so this backtest actually
        simulates what model_service.py does live (its value_bet_flag hard-excludes draws — see
        the comment there: "draws excluded per backtest findings"). Draws are consistently this
        model's worst-calibrated class; without this, the backtest was staking on bets the live
        system would never place, so its ROI didn't represent a real deployment.
    max_kelly_stake_pct: hard cap on any single Kelly bet as a fraction of current bankroll,
        applied on top of kelly_frac. Kelly sizing assumes the model's probabilities are correct;
        when they're optimistic (as draws/long-shot-away bets have shown themselves to be here),
        proportional staking compounds that error across thousands of bets and can walk the
        bankroll to near-zero even though each individual bet looked +EV under the model's own
        beliefs. This cap is a standard practical safety net against exactly that failure mode —
        it does not fix miscalibration, it just limits how much any one bad estimate can cost.

    use_selection_calibration: if True (default), corrects for the winner's-curse
        selection bias confirmed by src/calibration_check.py — flagged "value" bets
        are systematically overconfident (~+8pp) even though the model's raw
        probabilities are well-calibrated on ALL predictions. Before staking each
        season, fits an isotonic regression (walk-forward: only prior seasons'
        flagged bets, per selection type H/D/A) mapping raw model_prob -> true
        historical win rate among flagged bets, and re-checks edge_threshold using
        the corrected probability. Bets that no longer clear the threshold after
        correction are dropped; kept bets use the corrected probability for Kelly
        sizing (flat-stake PnL is unaffected by the correction itself, only by
        which bets survive re-filtering). Has no effect until calibration_min_samples
        historical flagged bets of that selection type have accumulated.
    calibration_min_samples: minimum prior flagged bets (per selection type) needed
        before a calibrator is fit; earlier seasons stake on raw model_prob unchanged.

    Returns (bet_log_df, summary_dict)
    """
    df = df.sort_values("date").reset_index(drop=True)
    df, _ = compute_elo_features(df)  # Elo is inherently walk-forward, safe to compute once globally
    df = add_rolling_form(df)
    df = add_rest_days(df)

    use_xg = xg_by_match_id is not None and len(xg_by_match_id) > 0
    if use_xg:
        df = add_rolling_xg_form(df, xg_by_match_id)
    feature_cols = FEATURE_COLUMNS_BASE + (FEATURE_COLUMNS_XG if use_xg else [])

    odds_cols = ["match_id", "odds_home", "odds_draw", "odds_away"]
    consensus_cols = [c for c in ("odds_home_consensus", "odds_draw_consensus", "odds_away_consensus")
                       if c in odds_df.columns]
    closing_cols = [c for c in ("odds_home_close", "odds_draw_close", "odds_away_close") if c in odds_df.columns]
    df = df.merge(odds_df[odds_cols + consensus_cols + closing_cols], on="match_id", how="left")
    use_consensus_edge = len(consensus_cols) == 3

    seasons = sorted(df["season"].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError("Not enough seasons of history for a walk-forward backtest.")

    test_seasons = seasons[min_train_seasons:]
    all_bets = []
    flat_bankroll = starting_bankroll
    kelly_bankroll = starting_bankroll
    bankroll_curve = []
    # walk-forward history of flagged (pre-correction) bets, per selection type,
    # used to fit each season's selection-bias calibrator on PRIOR seasons only
    selection_history = {"H": {"prob": [], "won": []},
                          "D": {"prob": [], "won": []},
                          "A": {"prob": [], "won": []}}

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

        # 1. Fit Dixon-Coles on all history strictly before this season
        dc_model = DixonColesModel(xi=0.0018)
        dc_model.fit(train_df)

        train_df = add_poisson_features(train_df, dc_model)
        test_df = add_poisson_features(test_df, dc_model)

        train_df = train_df.dropna(subset=feature_cols + ["ftr"])
        test_df_valid = test_df.dropna(subset=feature_cols).copy()

        if len(train_df) < 200 or len(test_df_valid) == 0:
            continue

        # 2. Train ML ensemble on everything before this season
        model = train_model(train_df, feature_cols)

        # 3. Predict this season
        probs = predict_probabilities(model, test_df_valid, feature_cols)
        test_df_valid = pd.concat([test_df_valid.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

        # 4. Find value bets vs bookmaker odds. edge_threshold is checked against
        # market_odds_cols (consensus, when available) — see load_backtest_odds
        # docstring and data/investigation_findings.md for why bet365-alone was
        # dropped as the edge reference. odds_cols (bet365) is still the EXECUTION
        # price used for ev/odds_taken/staking.
        test_df_valid = find_value_bets(
            test_df_valid,
            model_prob_cols=("p_home", "p_draw", "p_away"),
            odds_cols=("odds_home", "odds_draw", "odds_away"),
            market_odds_cols=(("odds_home_consensus", "odds_draw_consensus", "odds_away_consensus")
                               if use_consensus_edge else None),
            edge_threshold=edge_threshold,
            min_odds=min_odds, max_odds=max_odds,
        )

        if exclude_draws:
            # match model_service.py's live value_bet_flag, which never stakes on draws
            test_df_valid.loc[test_df_valid["bet_selection"] == "D", "bet_selection"] = None

        # 5. Selection-bias correction (see use_selection_calibration docstring above).
        # Fit calibrators from PRIOR seasons' flagged bets only (selection_history),
        # then apply to THIS season's flagged bets before staking. This season's own
        # flagged bets are recorded into selection_history AFTER staking, so nothing
        # here leaks into its own correction.
        calibrators = {}
        if use_selection_calibration:
            for sel in ("H", "D", "A"):
                calibrators[sel] = fit_selection_calibrator(
                    selection_history[sel]["prob"], selection_history[sel]["won"],
                    min_samples=calibration_min_samples,
                )

        flagged_mask = test_df_valid["bet_selection"].notna()
        # record this season's ORIGINAL (pre-correction) flagged bets for future seasons'
        # calibrators — must happen before we drop any rows for failing corrected_edge
        for _, frow in test_df_valid.loc[flagged_mask].iterrows():
            sel = frow["bet_selection"]
            selection_history[sel]["prob"].append(frow["model_prob"])
            selection_history[sel]["won"].append(int(frow["bet_selection"] == frow["ftr"]))

        if use_selection_calibration:
            corrected_prob = test_df_valid["model_prob"].copy()
            corrected_edge = test_df_valid["edge"].copy()
            for sel in ("H", "D", "A"):
                if calibrators[sel] is None:
                    continue
                sel_mask = flagged_mask & (test_df_valid["bet_selection"] == sel)
                if not sel_mask.any():
                    continue
                raw_prob = test_df_valid.loc[sel_mask, "model_prob"]
                implied_prob_norm = raw_prob - test_df_valid.loc[sel_mask, "edge"]  # edge = model_prob - implied
                cp = apply_selection_calibration(raw_prob.values, calibrators[sel])
                corrected_prob.loc[sel_mask] = cp
                corrected_edge.loc[sel_mask] = cp - implied_prob_norm.values
            test_df_valid["model_prob"] = corrected_prob
            test_df_valid["edge"] = corrected_edge
            # drop bets that no longer clear edge_threshold once probability is corrected
            fails_now = flagged_mask & (test_df_valid["edge"] <= edge_threshold)
            test_df_valid.loc[fails_now, "bet_selection"] = None

        for _, row in test_df_valid.iterrows():
            # NOTE: row["bet_selection"] can come back as float NaN (not Python None)
            # after round-tripping through a pandas object column via iterrows() —
            # "is None" silently fails to catch that and would let non-bets leak in
            # as automatic losses. pd.isna() catches both None and NaN correctly.
            if pd.isna(row["bet_selection"]):
                continue

            won = (row["bet_selection"] == row["ftr"])
            odds_taken = row["odds_taken"]

            flat_pnl = stake_flat * (odds_taken - 1) if won else -stake_flat
            flat_bankroll += flat_pnl

            k = kelly_fraction(row["model_prob"], odds_taken, fraction=kelly_frac)
            k = min(k, max_kelly_stake_pct)  # hard cap — see max_kelly_stake_pct docstring above
            kelly_stake = k * kelly_bankroll
            kelly_pnl = kelly_stake * (odds_taken - 1) if won else -kelly_stake
            kelly_bankroll += kelly_pnl

            # Closing-line value (CLV): did we take a better price than the market
            # eventually settled on? This is a lower-variance signal than raw ROI —
            # a model can show positive ROI on a small/lucky sample while consistently
            # getting worse prices than the close (a red flag), or vice versa.
            # Positive clv_pct means our odds beat the closing price.
            close_col = {"H": "odds_home_close", "D": "odds_draw_close", "A": "odds_away_close"}[row["bet_selection"]]
            closing_odds = row[close_col] if close_col in row.index else np.nan
            clv_pct = (odds_taken / closing_odds - 1) if pd.notna(closing_odds) and closing_odds > 0 else np.nan

            # Fade test: what if we'd bet the OPPOSITE side of this same match instead?
            # Only defined for H<->A (fading a Draw pick isn't well-defined in a 3-way
            # market with no lay/exchange data). Uses the same bet365 open / Pinnacle
            # close convention as the real bet, just for the other selection.
            fade_selection = fade_pnl = fade_won = fade_odds_taken = fade_closing_odds = fade_clv_pct = None
            if row["bet_selection"] in ("H", "A"):
                fade_selection = "A" if row["bet_selection"] == "H" else "H"
                fade_odds_col = "odds_home" if fade_selection == "H" else "odds_away"
                fade_close_col = "odds_home_close" if fade_selection == "H" else "odds_away_close"
                fade_odds_taken = row[fade_odds_col] if fade_odds_col in row.index else np.nan
                fade_closing_odds = row[fade_close_col] if fade_close_col in row.index else np.nan
                if pd.notna(fade_odds_taken):
                    fade_won = (fade_selection == row["ftr"])
                    fade_pnl = stake_flat * (fade_odds_taken - 1) if fade_won else -stake_flat
                    fade_clv_pct = ((fade_odds_taken / fade_closing_odds - 1)
                                     if pd.notna(fade_closing_odds) and fade_closing_odds > 0 else np.nan)

            all_bets.append({
                "season": season, "date": row["date"],
                "home_team": row["home_team"], "away_team": row["away_team"],
                "bet_selection": row["bet_selection"], "actual_result": row["ftr"],
                "won": won, "odds_taken": odds_taken, "closing_odds": closing_odds, "clv_pct": clv_pct,
                "model_prob": row["model_prob"],
                "edge": row["edge"], "flat_pnl": flat_pnl, "kelly_stake": kelly_stake,
                "kelly_pnl": kelly_pnl, "flat_bankroll": flat_bankroll, "kelly_bankroll": kelly_bankroll,
                "fade_selection": fade_selection, "fade_won": fade_won, "fade_odds_taken": fade_odds_taken,
                "fade_closing_odds": fade_closing_odds, "fade_clv_pct": fade_clv_pct, "fade_pnl": fade_pnl,
            })

        bankroll_curve.append({"season": season, "flat_bankroll": flat_bankroll, "kelly_bankroll": kelly_bankroll})

    bet_log = pd.DataFrame(all_bets)
    bankroll_by_season = pd.DataFrame(bankroll_curve)

    if len(bet_log) == 0:
        return bet_log, {"n_bets": 0, "note": "No qualifying value bets found — try lowering edge_threshold."}

    n_bets = len(bet_log)
    n_wins = bet_log["won"].sum()
    total_staked_flat = n_bets * stake_flat
    total_pnl_flat = bet_log["flat_pnl"].sum()
    roi_flat = total_pnl_flat / total_staked_flat

    roi_ci_lo, roi_ci_hi = bootstrap_roi_ci(bet_log, stake_flat=stake_flat)

    summary = {
        "n_bets": n_bets,
        "hit_rate": n_wins / n_bets,
        "flat_stake_roi": roi_flat,
        "flat_stake_roi_ci95": [roi_ci_lo, roi_ci_hi],
        "flat_stake_total_pnl": total_pnl_flat,
        "flat_ending_bankroll": flat_bankroll,
        "kelly_ending_bankroll": kelly_bankroll,
        "starting_bankroll": starting_bankroll,
        "avg_edge": bet_log["edge"].mean(),
        "avg_odds_taken": bet_log["odds_taken"].mean(),
        "avg_clv_pct": bet_log["clv_pct"].mean(skipna=True) if bet_log["clv_pct"].notna().any() else None,
        "pct_bets_beating_close": float((bet_log["clv_pct"] > 0).mean()) if bet_log["clv_pct"].notna().any() else None,
        "seasons_tested": f"{test_seasons[0]}–{test_seasons[-1]}",
    }

    # Fade test summary (H<->A swap only — see fade_selection comment above)
    fade_log = bet_log.dropna(subset=["fade_pnl"])
    if len(fade_log) > 0:
        fade_staked = len(fade_log) * stake_flat
        fade_roi_lo, fade_roi_hi = bootstrap_roi_ci(fade_log, stake_flat=stake_flat, pnl_col="fade_pnl")
        summary["fade_test"] = {
            "n_fade_bets": len(fade_log),
            "fade_hit_rate": float(fade_log["fade_won"].mean()),
            "fade_flat_roi": float(fade_log["fade_pnl"].sum() / fade_staked),
            "fade_flat_roi_ci95": [fade_roi_lo, fade_roi_hi],
            "fade_avg_clv_pct": float(fade_log["fade_clv_pct"].mean(skipna=True)) if fade_log["fade_clv_pct"].notna().any() else None,
            "pct_fade_beating_close": float((fade_log["fade_clv_pct"] > 0).mean()) if fade_log["fade_clv_pct"].notna().any() else None,
        }

    summary["by_selection"] = []
    for sel, grp in bet_log.groupby("bet_selection"):
        sel_lo, sel_hi = bootstrap_roi_ci(grp, stake_flat=stake_flat)
        summary["by_selection"].append({
            "bet_selection": sel,
            "n": len(grp),
            "win_rate": grp["won"].mean(),
            "avg_odds": grp["odds_taken"].mean(),
            "pnl": grp["flat_pnl"].sum(),
            "roi": grp["flat_pnl"].sum() / (len(grp) * stake_flat),
            "roi_ci95": [sel_lo, sel_hi],
        })

    return bet_log, summary, bankroll_by_season


def bootstrap_roi_ci(bet_log, stake_flat=1.0, n_boot=2000, ci=0.95, seed=42, block_by="season",
                      pnl_col="flat_pnl"):
    """
    Block-bootstrap confidence interval on flat-stake ROI.

    Resampling individual bets (rather than whole seasons) understates the
    true uncertainty, because bets within the same season share the same
    fitted model and are not independent draws — a bad/lucky model-fit for
    one season moves many bets in the same direction at once. Resampling
    whole seasons with replacement respects that dependency structure, at
    the cost of a wider (more honest) interval.

    Returns (ci_low, ci_high) for ROI. With few distinct seasons in bet_log
    (e.g. testing a single season), falls back to resampling individual bets
    and the interval should be read as an understatement of true uncertainty.
    """
    if len(bet_log) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    blocks = bet_log[block_by].unique() if block_by in bet_log.columns else None
    rois = np.empty(n_boot)

    for b in range(n_boot):
        if blocks is not None and len(blocks) >= 5:
            sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
            resampled = pd.concat([bet_log[bet_log[block_by] == blk] for blk in sampled_blocks], ignore_index=True)
        else:
            idx = rng.integers(0, len(bet_log), size=len(bet_log))
            resampled = bet_log.iloc[idx]

        pnl = resampled[pnl_col].sum()
        staked = len(resampled) * stake_flat
        rois[b] = pnl / staked if staked > 0 else np.nan

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return (float(np.nanpercentile(rois, lo_pct)), float(np.nanpercentile(rois, hi_pct)))


def log_backtest_run(summary, config, log_path="data/backtest_runs_log.jsonl"):
    """
    Append this run's config + headline results to a persistent log — every
    run, not just the ones that look good. This exists specifically to guard
    against unconsciously reporting only the best-looking configuration out of
    several tried (a real multiple-testing risk once you start tuning edge
    thresholds, feature sets, or staking schemes). Before trusting a new
    result, diff it against this log rather than your memory of past runs.
    """
    import json
    import os
    from datetime import datetime, timezone

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "n_bets": summary.get("n_bets"),
        "hit_rate": summary.get("hit_rate"),
        "flat_stake_roi": summary.get("flat_stake_roi"),
        "flat_stake_roi_ci95": summary.get("flat_stake_roi_ci95"),
        "avg_clv_pct": summary.get("avg_clv_pct"),
        "pct_bets_beating_close": summary.get("pct_bets_beating_close"),
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_backtest_odds(odds_path):
    """
    Build the odds_df run_walk_forward_backtest expects, from the raw
    epl_odds.csv bundled in data/.

    Bet-taken price: bet365 (fallback William Hill) — matches what
    model_service.py uses for live predictions, so backtest ROI reflects
    the same book a real deployment would actually be pricing against.

    Consensus reference (opening): market-average odds across all books in
    the dataset (fallback: Pinnacle's opening line, the conventional "sharpest
    single book" proxy). Used as an alternative to bet365 for the value-bet
    edge/threshold decision, to test whether the model's edge vs. bet365
    specifically reflects real mispricing or just single-book noise — see
    market_odds_cols in value_bets.find_value_bets and
    data/investigation_findings.md.

    Closing reference for CLV: Pinnacle's closing line (fallback: the
    cross-book closing average). Pinnacle is used deliberately — it's the
    market most often treated as the sharpest, lowest-margin reference price
    in the sports-betting literature, so beating *its* close is a more
    meaningful claim than beating a soft retail book's close.
    """
    raw = pd.read_csv(odds_path)

    def pick(row_df, candidates):
        out = pd.Series(np.nan, index=row_df.index)
        for c in candidates:
            if c in row_df.columns:
                out = out.fillna(row_df[c])
        return out

    odds = pd.DataFrame({"match_id": raw["match_id"]})
    odds["odds_home"] = pick(raw, ["bet365_1x2_home", "williamhill_1x2_home"])
    odds["odds_draw"] = pick(raw, ["bet365_1x2_draw", "williamhill_1x2_draw"])
    odds["odds_away"] = pick(raw, ["bet365_1x2_away", "williamhill_1x2_away"])
    odds["odds_home_consensus"] = pick(raw, ["market_avg_1x2_home", "pinnacle_1x2_home"])
    odds["odds_draw_consensus"] = pick(raw, ["market_avg_1x2_draw", "pinnacle_1x2_draw"])
    odds["odds_away_consensus"] = pick(raw, ["market_avg_1x2_away", "pinnacle_1x2_away"])
    odds["odds_home_close"] = pick(raw, ["pinnacle_1x2_home_close", "market_avg_1x2_home_close"])
    odds["odds_draw_close"] = pick(raw, ["pinnacle_1x2_draw_close", "market_avg_1x2_draw_close"])
    odds["odds_away_close"] = pick(raw, ["pinnacle_1x2_away_close", "market_avg_1x2_away_close"])
    return odds


if __name__ == "__main__":
    import json
    import os

    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = pd.read_csv(os.path.join(HERE, "data", "epl_results.csv"), parse_dates=["date"])
    odds = load_backtest_odds(os.path.join(HERE, "data", "epl_odds.csv"))

    xg_path = os.path.join(HERE, "data", "epl_xg.csv")
    xg_by_match_id = None
    if os.path.exists(xg_path):
        xg_by_match_id = pd.read_csv(xg_path)
        print(f"Loaded {len(xg_by_match_id)} xG-matched rows from data/epl_xg.csv — xG features enabled.")
    else:
        print("No data/epl_xg.csv found — running with the goals-only feature set. "
              "Run `python3 download_xg_data.py` first to include xG features.")

    config = dict(min_train_seasons=8, max_train_seasons=12, edge_threshold=0.03,
                  stake_flat=1.0, min_odds=1.3, max_odds=8.0, kelly_frac=0.25,
                  starting_bankroll=1000.0, exclude_draws=True, max_kelly_stake_pct=0.05,
                  use_selection_calibration=True, calibration_min_samples=300)
    config["used_xg_features"] = xg_by_match_id is not None

    bet_log, summary, bankroll_by_season = run_walk_forward_backtest(
        results, odds, xg_by_match_id=xg_by_match_id, **{k: v for k, v in config.items() if k != "used_xg_features"}
    )

    print(json.dumps(summary, indent=2, default=str))

    out_path = os.path.join(HERE, "data", "backtest_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log_backtest_run(summary, config, log_path=os.path.join(HERE, "data", "backtest_runs_log.jsonl"))
    print(f"\nWrote {out_path} and appended a row to data/backtest_runs_log.jsonl")
