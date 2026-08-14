"""
⚠️ TESTED AND REJECTED — see data/investigation_findings.md, step 2.
Swapping this in for src/ml_model.py (paired with elo_UPDATED.py) made the
walk-forward backtest measurably WORSE. Kept for reference; not imported by
src/backtest.py or src/model_service.py. Don't re-enable without re-running
the backtest and confirming it actually helps.

Gradient-boosted ensemble layer.

The Dixon-Coles model and Elo ratings are each decent on their own, but an
XGBoost classifier trained on BOTH (plus rolling form) as input features
usually edges out either alone, because it can learn non-linear interactions
(e.g. "big Elo favourite but out of form" behaves differently than either
signal suggests alone).

Two training-time upgrades over a flat fit:

- Early stopping instead of a fixed n_estimators. The most recent season
  inside the training window is held out as a validation split; a probe
  model is fit with early_stopping_rounds against it to pick how many trees
  are appropriate for THAT window, instead of using one fixed tree count
  across 25 seasons of walk-forward refits (early eras have far less
  training data than recent ones, so the "right" number of trees isn't
  constant). The final model is then refit with that chosen tree count on
  the FULL window (including the validation season), so no training data is
  thrown away in the deployed/backtested model itself.
- Recency-weighted training rows. DixonColesModel already time-decays match
  weight via its `xi` parameter; XGBoost previously saw a flat sample weight
  over the whole window. Using the same exponential decay here means both
  halves of the ensemble agree that a match from three years ago should
  count for less than one from three months ago.

Outputs are probability-calibrated (isotonic/sigmoid), which matters a lot
for betting: a classifier can be great at *ranking* outcomes while still
being badly miscalibrated in absolute probability terms, and miscalibrated
probabilities are exactly what leads to bad Kelly stakes and false "value".
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
INV_OUTCOME_MAP = {v: k for k, v in OUTCOME_MAP.items()}

DEFAULT_N_ESTIMATORS = 150  # fallback when early stopping can't be used (e.g. no season/date info)
MAX_N_ESTIMATORS_PROBE = 500  # generous cap the early-stopping probe searches up to


def _recency_weights(dates, xi=0.0018):
    """Same exponential time-decay as DixonColesModel's xi, applied to XGBoost's
    sample weights so both layers of the ensemble discount old matches the same way."""
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    return np.exp(-xi * days_ago).values.astype(np.float32)


def _pick_n_estimators(train_df, feature_cols, target_col, early_stopping_rounds, sample_weight_col):
    """Hold out the most recent season in train_df as a validation split and use
    early stopping to choose a tree count. Returns DEFAULT_N_ESTIMATORS if there's
    no season column, too few seasons, or too little data to hold anything out."""
    if "season" not in train_df.columns:
        return DEFAULT_N_ESTIMATORS

    seasons = sorted(train_df["season"].unique())
    if len(seasons) < 2:
        return DEFAULT_N_ESTIMATORS

    val_season = seasons[-1]
    fit_df = train_df[train_df["season"] != val_season]
    val_df = train_df[train_df["season"] == val_season]

    # need enough rows on both sides for early stopping to mean anything
    if len(fit_df) < 200 or len(val_df) < 50:
        return DEFAULT_N_ESTIMATORS

    X_fit = fit_df[feature_cols].values
    y_fit = fit_df[target_col].map(OUTCOME_MAP).values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].map(OUTCOME_MAP).values

    probe = XGBClassifier(
        n_estimators=MAX_N_ESTIMATORS_PROBE,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=4,
        early_stopping_rounds=early_stopping_rounds,
    )

    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sample_weight_col is not None:
        fit_kwargs["sample_weight"] = sample_weight_col.loc[fit_df.index].values
        fit_kwargs["sample_weight_eval_set"] = [sample_weight_col.loc[val_df.index].values]

    probe.fit(X_fit, y_fit, **fit_kwargs)

    best_iter = getattr(probe, "best_iteration", None)
    if best_iter is None:
        return probe.n_estimators
    return int(best_iter) + 1


def train_model(train_df, feature_cols, target_col="ftr", calibrate=True,
                 use_early_stopping=True, early_stopping_rounds=20,
                 recency_weight=True, xi=0.0018):
    X = train_df[feature_cols].values
    y = train_df[target_col].map(OUTCOME_MAP).values

    sample_weight = None
    sample_weight_series = None
    if recency_weight and "date" in train_df.columns:
        sample_weight = _recency_weights(train_df["date"], xi=xi)
        sample_weight_series = pd.Series(sample_weight, index=train_df.index)

    n_estimators = DEFAULT_N_ESTIMATORS
    if use_early_stopping:
        n_estimators = _pick_n_estimators(
            train_df, feature_cols, target_col, early_stopping_rounds, sample_weight_series
        )

    base = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=4,
    )

    if calibrate and len(train_df) > 500:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        model = base

    if sample_weight is not None:
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def predict_probabilities(model, df, feature_cols):
    """Returns a DataFrame with p_home, p_draw, p_away columns aligned to df's index."""
    X = df[feature_cols].values
    probs = model.predict_proba(X)
    out = pd.DataFrame(probs, columns=[INV_OUTCOME_MAP[i] for i in range(3)], index=df.index)
    out = out.rename(columns={"H": "p_home", "D": "p_draw", "A": "p_away"})
    return out[["p_home", "p_draw", "p_away"]]
