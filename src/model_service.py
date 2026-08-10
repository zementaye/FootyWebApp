"""
Model service: owns the trained model in memory and provides prediction methods
for the API layer. Trains once on startup (takes ~15-30s), then serves from
memory — no retraining per-request.

Also holds a rolling table of each team's most recent form (points/goals over
their last 5-10 matches) so that live fixture predictions get real form
features, not the neutral defaults used in main.py's standalone predict command.
"""

import pandas as pd
import numpy as np
import logging
import os
import threading

from src.elo import compute_elo_features
from src.poisson_model import DixonColesModel
import datetime as _dt

from src.features import (add_rolling_form, add_rest_days, add_rolling_xg_form,
                           add_poisson_features, FEATURE_COLUMNS_BASE, FEATURE_COLUMNS_XG)
from src.ml_model import train_model, predict_probabilities
from src.value_bets import find_value_bets, remove_overround, implied_probabilities

logger = logging.getLogger("model_service")


class ModelService:
    def __init__(self, results_path, odds_path, xg_path=None):
        self.results_path = results_path
        self.odds_path = odds_path
        # Auto-discovers data/epl_xg.csv next to results_path if not given explicitly —
        # keeps the "no required config" behavior described in README_DEPLOY.md: xG
        # features turn on automatically once you've run download_xg_data.py, and this
        # falls back cleanly to the goals-only feature set if that file isn't there.
        self.xg_path = xg_path or os.path.join(os.path.dirname(results_path), "epl_xg.csv")
        self.lock = threading.Lock()
        self.ready = False
        self._train()

    def _load_data(self):
        results = pd.read_csv(self.results_path, parse_dates=["date"])
        odds_raw = pd.read_csv(self.odds_path)
        odds = odds_raw[["match_id", "bet365_1x2_home", "bet365_1x2_draw", "bet365_1x2_away",
                          "williamhill_1x2_home", "williamhill_1x2_draw", "williamhill_1x2_away"]].copy()
        odds["odds_home"] = odds["bet365_1x2_home"].fillna(odds["williamhill_1x2_home"])
        odds["odds_draw"] = odds["bet365_1x2_draw"].fillna(odds["williamhill_1x2_draw"])
        odds["odds_away"] = odds["bet365_1x2_away"].fillna(odds["williamhill_1x2_away"])
        return results, odds[["match_id", "odds_home", "odds_draw", "odds_away"]]

    def _train(self):
        logger.info("Training model on startup...")
        results, odds = self._load_data()
        results = results.sort_values("date").reset_index(drop=True)

        results, elo_engine = compute_elo_features(results)
        results = add_rolling_form(results)
        results = add_rest_days(results)

        use_xg = os.path.exists(self.xg_path)
        if use_xg:
            xg_by_match_id = pd.read_csv(self.xg_path)
            results = add_rolling_xg_form(results, xg_by_match_id)
            logger.info(f"Loaded {len(xg_by_match_id)} xG-matched rows from {self.xg_path} — xG features enabled.")
        else:
            logger.info(f"No xG file at {self.xg_path} — serving with the goals-only feature set. "
                        f"Run download_xg_data.py to enable xG features.")
        self.use_xg = use_xg

        dc_model = DixonColesModel(xi=0.0018)
        dc_model.fit(results.dropna(subset=["fthg", "ftag"]))
        results = add_poisson_features(results, dc_model)

        feature_cols = FEATURE_COLUMNS_BASE + (FEATURE_COLUMNS_XG if use_xg else [])
        train_df = results.dropna(subset=feature_cols + ["ftr"])
        ml_model = train_model(train_df, feature_cols)

        # keep each team's most recent form snapshot for live fixture predictions
        latest_form = {}
        latest_match_date = {}
        for team in set(results["home_team"]) | set(results["away_team"]):
            home_rows = results[results["home_team"] == team].tail(1)
            away_rows = results[results["away_team"] == team].tail(1)
            candidates = []
            if len(home_rows):
                snap = {
                    "form_pts_5": home_rows["home_form_pts_5"].iloc[0],
                    "form_gf_5": home_rows["home_form_gf_5"].iloc[0],
                    "form_ga_5": home_rows["home_form_ga_5"].iloc[0],
                    "form_pts_10": home_rows["home_form_pts_10"].iloc[0],
                    "form_gf_10": home_rows["home_form_gf_10"].iloc[0],
                }
                if use_xg:
                    snap["xg_for_5"] = home_rows["home_xg_for_5"].iloc[0]
                    snap["xg_for_10"] = home_rows["home_xg_for_10"].iloc[0]
                    snap["xg_overperf_5"] = home_rows["home_xg_overperf_5"].iloc[0]
                candidates.append((home_rows["date"].iloc[0], snap))
            if len(away_rows):
                snap = {
                    "form_pts_5": away_rows["away_form_pts_5"].iloc[0],
                    "form_gf_5": away_rows["away_form_gf_5"].iloc[0],
                    "form_ga_5": away_rows["away_form_ga_5"].iloc[0],
                    "form_pts_10": away_rows["away_form_pts_10"].iloc[0],
                    "form_gf_10": away_rows["away_form_gf_10"].iloc[0],
                }
                if use_xg:
                    snap["xg_for_5"] = away_rows["away_xg_for_5"].iloc[0]
                    snap["xg_for_10"] = away_rows["away_xg_for_10"].iloc[0]
                    snap["xg_overperf_5"] = away_rows["away_xg_overperf_5"].iloc[0]
                candidates.append((away_rows["date"].iloc[0], snap))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                latest_form[team] = candidates[-1][1]
                latest_match_date[team] = candidates[-1][0]

        median_rest_days = pd.concat([results["home_rest_days"], results["away_rest_days"]]).median()

        with self.lock:
            self.elo_engine = elo_engine
            self.dc_model = dc_model
            self.ml_model = ml_model
            self.feature_cols = feature_cols
            self.latest_form = latest_form
            self.latest_match_date = latest_match_date
            self.median_rest_days = float(median_rest_days)
            self.league_avg_form_ga = float(results["home_form_ga_5"].mean())
            self.ready = True

        logger.info(f"Model trained on {len(train_df)} matches. Ready.")

    def retrain(self):
        """Call this periodically (e.g. weekly) as new results come in, if you extend
        the pipeline to append newly completed matches to results_path."""
        self._train()

    def _build_features(self, home_team, away_team, match_date=None):
        # elo.py's EloRatings uses a single rating per team (plus a fixed home-advantage
        # constant baked into update()), not separate home/away ratings — the venue= kwarg
        # belongs to the rejected elo_UPDATED.py variant (see that file's header) and isn't
        # part of this class's API.
        elo_diff = (self.elo_engine.get_rating(home_team)
                    - self.elo_engine.get_rating(away_team))
        ph, pd_, pa = self.dc_model.match_probabilities(home_team, away_team)
        lh, la = self.dc_model.predict_lambdas(home_team, away_team)

        hf = self.latest_form.get(home_team, {})
        af = self.latest_form.get(away_team, {})
        default_ga = self.league_avg_form_ga

        match_date = match_date or pd.Timestamp(_dt.date.today())
        cap_days = 30
        short_rest_days = 5

        def _rest_days(team):
            last_date = self.latest_match_date.get(team)
            if last_date is None:
                return self.median_rest_days
            return min((match_date - last_date).days, cap_days)

        home_rest = _rest_days(home_team)
        away_rest = _rest_days(away_team)

        feat = {
            "elo_diff": elo_diff,
            "dc_p_home": ph, "dc_p_draw": pd_, "dc_p_away": pa,
            "dc_lambda_home": lh, "dc_lambda_away": la,
            "form_pts_diff_5": hf.get("form_pts_5", 1.3) - af.get("form_pts_5", 1.3),
            "form_gf_diff_5": hf.get("form_gf_5", 1.3) - af.get("form_gf_5", 1.3),
            "form_pts_diff_10": hf.get("form_pts_10", 1.3) - af.get("form_pts_10", 1.3),
            "form_gf_diff_10": hf.get("form_gf_10", 1.3) - af.get("form_gf_10", 1.3),
            "home_form_ga_5": hf.get("form_ga_5", default_ga),
            "away_form_ga_5": af.get("form_ga_5", default_ga),
            "rest_days_diff": home_rest - away_rest,
            "home_short_rest_flag": int(home_rest <= short_rest_days),
            "away_short_rest_flag": int(away_rest <= short_rest_days),
        }

        if self.use_xg:
            has_xg = ("xg_for_5" in hf) and ("xg_for_5" in af)
            feat["xg_form_diff_5"] = hf.get("xg_for_5", 0.0) - af.get("xg_for_5", 0.0)
            feat["xg_form_diff_10"] = hf.get("xg_for_10", 0.0) - af.get("xg_for_10", 0.0)
            feat["home_xg_overperf_5"] = hf.get("xg_overperf_5", 0.0)
            feat["away_xg_overperf_5"] = af.get("xg_overperf_5", 0.0)
            feat["has_xg_data"] = int(has_xg)

        return feat

    def predict_match(self, home_team, away_team, odds_home=None, odds_draw=None, odds_away=None,
                       edge_threshold=0.05, match_date=None):
        if isinstance(match_date, str):
            match_date = pd.to_datetime(match_date, dayfirst=True, errors="coerce")
        feat = self._build_features(home_team, away_team, match_date=match_date)
        feat_df = pd.DataFrame([feat])[self.feature_cols]
        probs = predict_probabilities(self.ml_model, feat_df, self.feature_cols).iloc[0]

        result = {
            "home_team": home_team,
            "away_team": away_team,
            "p_home": round(float(probs["p_home"]), 4),
            "p_draw": round(float(probs["p_draw"]), 4),
            "p_away": round(float(probs["p_away"]), 4),
            "expected_goals_home": round(float(feat["dc_lambda_home"]), 2),
            "expected_goals_away": round(float(feat["dc_lambda_away"]), 2),
        }

        if odds_home and odds_draw and odds_away:
            raw = implied_probabilities(odds_home, odds_draw, odds_away)
            norm = remove_overround(*raw)
            probs_arr = [probs["p_home"], probs["p_draw"], probs["p_away"]]
            odds_arr = [odds_home, odds_draw, odds_away]
            labels = ["H", "D", "A"]

            best_edge, best_i = -np.inf, None
            for i in range(3):
                edge = probs_arr[i] - norm[i]
                if edge > best_edge:
                    best_edge, best_i = edge, i

            ev = probs_arr[best_i] * odds_arr[best_i] - 1
            result["odds_home"] = odds_home
            result["odds_draw"] = odds_draw
            result["odds_away"] = odds_away
            result["market_implied_home"] = round(norm[0], 4)
            result["market_implied_draw"] = round(norm[1], 4)
            result["market_implied_away"] = round(norm[2], 4)
            result["best_edge"] = round(float(best_edge), 4)
            result["best_edge_selection"] = labels[best_i]
            result["expected_value"] = round(float(ev), 4)
            result["value_bet_flag"] = bool(best_edge > edge_threshold and ev > 0
                                             and 1.3 <= odds_arr[best_i] <= 8.0
                                             and labels[best_i] != "D")  # draws excluded per backtest findings

        return result
