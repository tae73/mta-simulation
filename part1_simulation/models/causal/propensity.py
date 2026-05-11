"""Propensity Score-based causal attribution: IPW and Doubly Robust.

Addresses selection bias: user segments determine channel exposure
(e.g., Loyal users see more Email). Without correction, Email's effect
is confounded by user loyalty.

IPW: Re-weight each observation by 1/P(treatment | confounders)
DR:  Combine outcome model + propensity model for double robustness
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from part1_simulation import AttributionResult, CHANNEL_NAMES


def _build_user_level_data(journeys: pd.DataFrame) -> pd.DataFrame:
    """Build user-level data with channel exposures and segment."""
    user_level = journeys.groupby("user_id").agg(
        converted=("converted", "first"),
        segment=("segment", "first"),
        journey_length=("journey_length", "first"),
    )

    # Binary channel presence
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
    # Segment dummies
    seg_dummies = pd.get_dummies(result["segment"], prefix="seg", drop_first=True)
    result = pd.concat([result, seg_dummies], axis=1)

    return result.reset_index()


def _estimate_propensity(
    user_data: pd.DataFrame,
    treatment_channel: str,
    confounder_cols: list,
) -> np.ndarray:
    """Estimate P(exposed to channel | confounders) via logistic regression."""
    X = user_data[confounder_cols].values
    T = user_data[treatment_channel].values

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X, T)
    propensity = model.predict_proba(X)[:, 1]

    # Clip to avoid extreme weights
    return np.clip(propensity, 0.01, 0.99)


def compute_ipw_attribution(
    journeys: pd.DataFrame,
) -> AttributionResult:
    """Inverse Propensity Weighting attribution.

    For each channel c:
    ATE_c = E[Y * T_c / e(X)] - E[Y * (1 - T_c) / (1 - e(X))]
    where e(X) = P(T_c = 1 | X)
    """
    user_data = _build_user_level_data(journeys)
    seg_cols = [c for c in user_data.columns if c.startswith("seg_")]

    ate_estimates = {}
    for channel in CHANNEL_NAMES:
        other_channels = [ch for ch in CHANNEL_NAMES if ch != channel]
        confounder_cols = seg_cols + other_channels

        propensity = _estimate_propensity(user_data, channel, confounder_cols)

        T = user_data[channel].values
        Y = user_data["converted"].values.astype(float)

        # IPW estimator: ATE = mean(Y * T / e) - mean(Y * (1-T) / (1-e))
        treated_term = np.mean(Y * T / propensity)
        control_term = np.mean(Y * (1 - T) / (1 - propensity))
        ate_estimates[channel] = treated_term - control_term

    # Normalize
    clamped = {k: max(0.0, v) for k, v in ate_estimates.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    return AttributionResult(
        method="IPW",
        channel_credits=normalized,
        channel_credits_raw=ate_estimates,
        metadata={"estimator": "inverse_propensity_weighting"},
    )


def compute_doubly_robust_attribution(
    journeys: pd.DataFrame,
) -> AttributionResult:
    """Doubly Robust attribution: combines outcome model + propensity model.

    DR estimator:
    ATE = E[m1(X) - m0(X) + T(Y - m1(X))/e(X) - (1-T)(Y - m0(X))/(1-e(X))]
    where m1, m0 are outcome models for treated/control.
    """
    user_data = _build_user_level_data(journeys)
    seg_cols = [c for c in user_data.columns if c.startswith("seg_")]

    ate_estimates = {}
    for channel in CHANNEL_NAMES:
        other_channels = [ch for ch in CHANNEL_NAMES if ch != channel]
        confounder_cols = seg_cols + other_channels

        X = user_data[confounder_cols].values
        T = user_data[channel].values
        Y = user_data["converted"].values.astype(float)

        # Propensity model
        propensity = _estimate_propensity(user_data, channel, confounder_cols)

        # Outcome models: E[Y | X, T=1] and E[Y | X, T=0]
        X_with_T = np.column_stack([X, T])

        outcome_model = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42, class_weight="balanced",
        )
        outcome_model.fit(X_with_T, Y.astype(int))

        # Predict under treatment and control
        X_treated = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        mu1 = outcome_model.predict_proba(X_treated)[:, 1]
        mu0 = outcome_model.predict_proba(X_control)[:, 1]

        # DR estimator
        dr = (
            mu1 - mu0
            + T * (Y - mu1) / propensity
            - (1 - T) * (Y - mu0) / (1 - propensity)
        )
        ate_estimates[channel] = float(np.mean(dr))

    # Normalize
    clamped = {k: max(0.0, v) for k, v in ate_estimates.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    return AttributionResult(
        method="Doubly Robust",
        channel_credits=normalized,
        channel_credits_raw=ate_estimates,
        metadata={"estimator": "doubly_robust"},
    )
