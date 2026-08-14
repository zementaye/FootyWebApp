"""
Gradient-boosted ensemble layer. VARIANT D: early-stopping tree count ONLY —
no recency-weighted sample_weight. Isolates the early-stopping effect.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
INV_OUTCOME_MAP = {v: k for k, v in OUTCOME_MAP.items()}

DEFAULT_N_ESTIMATORS = 150
MAX_N_ESTIMATORS_PROBE = 500


def _pick_n_estimators(train_df, feature_cols, target_col, early_stopping_rounds):
    if "season" not in train_df.columns:
        return DEFAULT_N_ESTIMATORS

    seasons = sorted(train_df["season"].unique())
    if len(seasons) < 2:
        return DEFAULT_N_ESTIMATORS

    val_season = seasons[-1]
    fit_df = train_df[train_df["season"] != val_season]
    val_df = train_df[train_df["season"] == val_season]

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
    probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = getattr(probe, "best_iteration", None)
    if best_iter is None:
        return probe.n_estimators
    return int(best_iter) + 1


def train_model(train_df, feature_cols, target_col="ftr", calibrate=True,
                 use_early_stopping=True, early_stopping_rounds=20):
    X = train_df[feature_cols].values
    y = train_df[target_col].map(OUTCOME_MAP).values

    n_estimators = DEFAULT_N_ESTIMATORS
    if use_early_stopping:
        n_estimators = _pick_n_estimators(train_df, feature_cols, target_col, early_stopping_rounds)

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

    model.fit(X, y)
    return model


def predict_probabilities(model, df, feature_cols):
    X = df[feature_cols].values
    probs = model.predict_proba(X)
    out = pd.DataFrame(probs, columns=[INV_OUTCOME_MAP[i] for i in range(3)], index=df.index)
    out = out.rename(columns={"H": "p_home", "D": "p_draw", "A": "p_away"})
    return out[["p_home", "p_draw", "p_away"]]
