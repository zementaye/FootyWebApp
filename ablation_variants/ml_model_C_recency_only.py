"""
Gradient-boosted ensemble layer. VARIANT C: recency-weighted training rows
ONLY — fixed n_estimators=150 (original), no early stopping. Isolates the
recency-weighting effect.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


OUTCOME_MAP = {"H": 0, "D": 1, "A": 2}
INV_OUTCOME_MAP = {v: k for k, v in OUTCOME_MAP.items()}


def _recency_weights(dates, xi=0.0018):
    dates = pd.to_datetime(dates)
    days_ago = (dates.max() - dates).dt.days
    return np.exp(-xi * days_ago).values.astype(np.float32)


def train_model(train_df, feature_cols, target_col="ftr", calibrate=True,
                 recency_weight=True, xi=0.0018):
    X = train_df[feature_cols].values
    y = train_df[target_col].map(OUTCOME_MAP).values

    sample_weight = None
    if recency_weight and "date" in train_df.columns:
        sample_weight = _recency_weights(train_df["date"], xi=xi)

    base = XGBClassifier(
        n_estimators=150,
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
    X = df[feature_cols].values
    probs = model.predict_proba(X)
    out = pd.DataFrame(probs, columns=[INV_OUTCOME_MAP[i] for i in range(3)], index=df.index)
    out = out.rename(columns={"H": "p_home", "D": "p_draw", "A": "p_away"})
    return out[["p_home", "p_draw", "p_away"]]
