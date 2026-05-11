"""Double Machine Learning (DML) for MTA (Chernozhukov et al. 2018).

DML removes nuisance parameter bias via cross-fitting:
1. Partial out confounders from both treatment and outcome
2. Estimate treatment effect from residuals
3. Cross-fitting prevents overfitting bias

For each channel: T = channel exposure, Y = conversion,
W = segment + other channel exposures.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from part1_simulation import AttributionResult, CHANNEL_NAMES


def _build_user_level_data(journeys: pd.DataFrame) -> pd.DataFrame:
    """Build user-level data with channel exposures and segment."""
    user_level = journeys.groupby("user_id").agg(
        converted=("converted", "first"),
        segment=("segment", "first"),
        journey_length=("journey_length", "first"),
    )

    channel_presence = (
        journeys
        .groupby(["user_id", "channel"], observed=True)
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )
    for ch in CHANNEL_NAMES:
        if ch not in channel_presence.columns:
            channel_presence[ch] = 0

    result = user_level.join(channel_presence[list(CHANNEL_NAMES)])
    seg_dummies = pd.get_dummies(result["segment"], prefix="seg", drop_first=True)
    result = pd.concat([result, seg_dummies], axis=1)

    return result.reset_index()


def _estimate_dml_ate(
    Y: np.ndarray,
    T: np.ndarray,
    W: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Estimate ATE using partialling-out DML with cross-fitting.

    Steps:
    1. For each fold k:
       a. Train outcome model: E[Y | W] on training data
       b. Train treatment model: E[T | W] on training data
       c. Compute residuals on held-out data:
          Y_res = Y - E[Y|W], T_res = T - E[T|W]
    2. ATE = mean(Y_res * T_res) / mean(T_res^2)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    Y_residuals = np.zeros_like(Y, dtype=float)
    T_residuals = np.zeros_like(T, dtype=float)

    for train_idx, test_idx in kf.split(W):
        W_train, W_test = W[train_idx], W[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        # Outcome model: E[Y | W]
        outcome_model = LogisticRegression(
            max_iter=500, solver="lbfgs", random_state=42, class_weight="balanced",
        )
        outcome_model.fit(W_train, Y_train.astype(int))
        Y_hat = outcome_model.predict_proba(W_test)[:, 1]
        Y_residuals[test_idx] = Y_test - Y_hat

        # Treatment model: E[T | W]
        treatment_model = LogisticRegression(
            max_iter=500, solver="lbfgs", random_state=42,
        )
        treatment_model.fit(W_train, T_train.astype(int))
        T_hat = treatment_model.predict_proba(W_test)[:, 1]
        T_residuals[test_idx] = T_test - T_hat

    # ATE from residuals (Frisch-Waugh-Lovell)
    denominator = np.mean(T_residuals ** 2)
    if abs(denominator) < 1e-10:
        return 0.0

    ate = np.mean(Y_residuals * T_residuals) / denominator
    return float(ate)


def compute_dml_attribution(
    journeys: pd.DataFrame,
    n_folds: int = 5,
) -> AttributionResult:
    """Compute DML-based attribution: ATE per channel via cross-fitting.

    For each of 7 channels, treats it as the treatment variable,
    with segment + other channels as confounders.

    Args:
        journeys: long-format journey DataFrame.
        n_folds: number of cross-fitting folds.

    Returns:
        AttributionResult with normalized ATE-based credits.
    """
    user_data = _build_user_level_data(journeys)
    seg_cols = [c for c in user_data.columns if c.startswith("seg_")]

    Y = user_data["converted"].values.astype(float)
    ate_estimates = {}

    for channel in CHANNEL_NAMES:
        T = user_data[channel].values.astype(float)
        other_channels = [ch for ch in CHANNEL_NAMES if ch != channel]
        W = user_data[seg_cols + other_channels].values.astype(float)

        ate = _estimate_dml_ate(Y, T, W, n_folds=n_folds)
        ate_estimates[channel] = ate

    # Normalize
    clamped = {k: max(0.0, v) for k, v in ate_estimates.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    return AttributionResult(
        method="DML",
        channel_credits=normalized,
        channel_credits_raw=ate_estimates,
        metadata={"n_folds": n_folds, "estimator": "partialling_out_dml"},
    )
