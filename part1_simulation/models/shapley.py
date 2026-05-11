"""Exact Shapley Value attribution (128 coalitions).

With 7 channels, enumerates all 2^7 = 128 coalitions for exact computation.

Two value function versions:
    Version A (conversion_rate): v(S) = conversion rate of journeys containing
        only channels in S.
    Version B (model_based): Train logistic regression, then v(S) = average
        model prediction with non-S channels masked.
"""

import itertools
import math
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from part1_simulation import AttributionResult, CHANNEL_NAMES


def _build_user_channel_matrix(journeys: pd.DataFrame) -> pd.DataFrame:
    """Build user-level binary channel presence matrix.

    Returns DataFrame: user_id, converted, + one column per channel (0/1).
    """
    user_level = journeys.groupby("user_id").agg(
        converted=("converted", "first"),
    )

    # Pivot channel presence
    channel_presence = (
        journeys
        .groupby(["user_id", "channel"], observed=True)
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)  # binary presence
    )

    # Ensure all channels present as columns
    for ch in CHANNEL_NAMES:
        if ch not in channel_presence.columns:
            channel_presence[ch] = 0

    channel_presence = channel_presence[list(CHANNEL_NAMES)]

    result = user_level.join(channel_presence)
    return result.reset_index()


# ============================================================
# Value Function A: Conversion Rate
# ============================================================

def _compute_coalition_value_conversion_rate(
    user_matrix: pd.DataFrame,
    coalition: FrozenSet[str],
) -> float:
    """v(S) = conversion rate of journeys where all channels are in S.

    Filters to users whose channel set is a subset of the coalition.
    """
    if not coalition:
        return 0.0

    # Users whose channels are all within the coalition
    non_coalition = [ch for ch in CHANNEL_NAMES if ch not in coalition]
    mask = pd.Series(True, index=user_matrix.index)
    for ch in non_coalition:
        mask = mask & (user_matrix[ch] == 0)

    subset = user_matrix[mask]
    if len(subset) == 0:
        return 0.0

    return float(subset["converted"].mean())


# ============================================================
# Value Function B: Model-Based
# ============================================================

def _train_logistic_model(
    user_matrix: pd.DataFrame,
) -> LogisticRegression:
    """Train logistic regression on channel presence indicators."""
    X = user_matrix[list(CHANNEL_NAMES)].values
    y = user_matrix["converted"].values.astype(int)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X, y)
    return model


def _compute_coalition_value_model_based(
    user_matrix: pd.DataFrame,
    coalition: FrozenSet[str],
    model: LogisticRegression,
) -> float:
    """v(S) = average model prediction with non-S channels masked to 0."""
    X = user_matrix[list(CHANNEL_NAMES)].values.copy()

    # Mask non-coalition channels to 0
    for i, ch in enumerate(CHANNEL_NAMES):
        if ch not in coalition:
            X[:, i] = 0

    probs = model.predict_proba(X)[:, 1]
    return float(probs.mean())


# ============================================================
# Shapley Computation
# ============================================================

def _compute_exact_shapley(
    channels: Tuple[str, ...],
    value_fn,
) -> Dict[str, float]:
    """Compute exact Shapley values by enumerating all coalitions.

    Shapley formula:
        φ_i = Σ_{S �� N\\{i}} [|S|!(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
    """
    n = len(channels)

    shapley_values = {}
    for channel in channels:
        others = [c for c in channels if c != channel]
        sv = 0.0

        for r in range(n):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                S_with = S | {channel}

                marginal = value_fn(S_with) - value_fn(S)
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                sv += weight * marginal

        shapley_values[channel] = sv

    return shapley_values


def compute_shapley_conversion_rate(
    journeys: pd.DataFrame,
) -> AttributionResult:
    """Shapley Value with conversion-rate value function (Version A).

    v(S) = conversion rate of journeys where all channels ⊆ S.
    """
    user_matrix = _build_user_channel_matrix(journeys)

    # Cache coalition values
    cache: Dict[FrozenSet[str], float] = {}

    def value_fn(coalition: FrozenSet[str]) -> float:
        if coalition not in cache:
            cache[coalition] = _compute_coalition_value_conversion_rate(
                user_matrix, coalition,
            )
        return cache[coalition]

    raw_shapley = _compute_exact_shapley(CHANNEL_NAMES, value_fn)

    # Normalize: clamp negatives to 0, then normalize to sum=1
    clamped = {k: max(0.0, v) for k, v in raw_shapley.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    return AttributionResult(
        method="Shapley (conv. rate)",
        channel_credits=normalized,
        channel_credits_raw=raw_shapley,
        metadata={"value_function": "conversion_rate", "n_coalitions": len(cache)},
    )


def compute_shapley_model_based(
    journeys: pd.DataFrame,
) -> AttributionResult:
    """Shapley Value with model-based value function (Version B).

    v(S) = average logistic regression prediction with non-S channels masked.
    """
    user_matrix = _build_user_channel_matrix(journeys)
    model = _train_logistic_model(user_matrix)

    cache: Dict[FrozenSet[str], float] = {}

    def value_fn(coalition: FrozenSet[str]) -> float:
        if coalition not in cache:
            if not coalition:
                # Empty coalition: predict with all channels zeroed
                X_zero = np.zeros((len(user_matrix), len(CHANNEL_NAMES)))
                cache[coalition] = float(model.predict_proba(X_zero)[:, 1].mean())
            else:
                cache[coalition] = _compute_coalition_value_model_based(
                    user_matrix, coalition, model,
                )
        return cache[coalition]

    raw_shapley = _compute_exact_shapley(CHANNEL_NAMES, value_fn)

    clamped = {k: max(0.0, v) for k, v in raw_shapley.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    return AttributionResult(
        method="Shapley (model-based)",
        channel_credits=normalized,
        channel_credits_raw=raw_shapley,
        metadata={
            "value_function": "model_based",
            "n_coalitions": len(cache),
            "logistic_coefs": dict(zip(CHANNEL_NAMES, model.coef_[0].tolist())),
        },
    )
