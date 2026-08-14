"""
Gradient-boosted ensemble layer.

The Dixon-Coles model and Elo ratings are each decent on their own, but an
XGBoost classifier trained on BOTH (plus rolling form) as input features
usually edges out either alone, because it can learn non-linear interactions
(e.g. "big Elo favourite but out of form" behaves differently than either
signal suggests alone).

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


def train_model(train_df, feature_cols, target_col="ftr", calibrate=True):
    X = train_df[feature_cols].values
    y = train_df[target_col].map(OUTCOME_MAP).values

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

    model.fit(X, y)
    return model


def predict_probabilities(model, df, feature_cols):
    """Returns a DataFrame with p_home, p_draw, p_away columns aligned to df's index."""
    X = df[feature_cols].values
    probs = model.predict_proba(X)
    out = pd.DataFrame(probs, columns=[INV_OUTCOME_MAP[i] for i in range(3)], index=df.index)
    out = out.rename(columns={"H": "p_home", "D": "p_draw", "A": "p_away"})
    return out[["p_home", "p_draw", "p_away"]]
