"""Incremental Shapley Value attribution (Du et al. 2019).

Two-step pipeline following the original paper:
    Step 1 (Response Modeling): Train a model on user-level features derived
        from observed journey data to predict P(conversion).
    Step 2 (Credit Allocation): Compute Shapley Values on the INCREMENTAL
        value function v(S) = E[Y|S] - E[Y|empty], allocating only the
        ad-driven portion of conversions (not base conversions).

Key difference from traditional Shapley: base conversion (P(conv | no ads))
is subtracted so channels receive credit only for incremental lift.

The response model is learned from data — no DGP oracle access.
"""

import itertools
import logging
import math
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from part1_simulation import AttributionResult, CHANNEL_NAMES

logger = logging.getLogger(__name__)

# Per-channel feature names: (present, count, recency)
_CHANNEL_FEATURE_PREFIXES = ("present", "count", "recency")
# Values representing "no exposure" for each feature type
_ABSENT_VALUES = (0.0, 0.0, 1.0)


# ============================================================
# Feature Engineering (from observed data only)
# ============================================================

def _build_user_features(
    journeys: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Build user-level feature matrix from journey data.

    Features per user (no DGP parameters):
        - present_{ch}: binary channel presence (7)
        - count_{ch}: touch count per channel (7)
        - recency_{ch}: normalized time since last touch (7)
        - segment dummies: drop-first one-hot (2)
        - journey_length: total touchpoints (1)
        - journey_duration_hours: total time span (1)

    Returns:
        (user_df with 'converted' column, channel_feature_map).
    """
    user_groups = journeys.groupby("user_id", sort=False)
    all_segments = sorted(journeys["segment"].unique())

    rows = []
    for user_id, group in user_groups:
        group_sorted = group.sort_values("touchpoint_idx")
        row: Dict[str, float] = {}

        row["converted"] = float(group_sorted["converted"].iloc[0])

        last_ts = group_sorted["timestamp"].max()
        first_ts = group_sorted["timestamp"].min()
        journey_duration = max(last_ts - first_ts, 1.0)

        for ch in CHANNEL_NAMES:
            ch_touches = group_sorted[group_sorted["channel"] == ch]
            row[f"present_{ch}"] = 1.0 if len(ch_touches) > 0 else 0.0
            row[f"count_{ch}"] = float(len(ch_touches))
            if len(ch_touches) > 0:
                row[f"recency_{ch}"] = (last_ts - ch_touches["timestamp"].max()) / journey_duration
            else:
                row[f"recency_{ch}"] = 1.0

        seg = group_sorted["segment"].iloc[0]
        for s in all_segments[1:]:
            row[f"seg_{s}"] = 1.0 if seg == s else 0.0

        row["journey_length"] = float(len(group_sorted))
        row["journey_duration_hours"] = journey_duration
        rows.append(row)

    df = pd.DataFrame(rows)
    col_names = list(df.columns)

    channel_feature_map: Dict[str, List[int]] = {}
    feature_cols = [c for c in col_names if c != "converted"]
    for ch in CHANNEL_NAMES:
        channel_feature_map[ch] = [
            feature_cols.index(f"{prefix}_{ch}")
            for prefix in _CHANNEL_FEATURE_PREFIXES
        ]

    return df, channel_feature_map


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (everything except 'converted')."""
    return [c for c in df.columns if c != "converted"]


# ============================================================
# Response Model
# ============================================================

def _train_response_model(
    df: pd.DataFrame,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train response model on enriched user-level features.

    Uses LogisticRegression with balanced class weights on scaled features.
    """
    feature_cols = _get_feature_columns(df)
    X = df[feature_cols].values
    y = df["converted"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
        C=1.0,
    )
    model.fit(X_scaled, y)

    from sklearn.metrics import roc_auc_score
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
        train_auc = roc_auc_score(y, probs)
    except ValueError:
        train_auc = 0.5
    logger.info(f"  Response model: train_AUC={train_auc:.4f}")

    return model, scaler


# ============================================================
# Coalition Prediction
# ============================================================

def _predict_coalition(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_raw: np.ndarray,
    coalition: FrozenSet[str],
    channel_feature_map: Dict[str, List[int]],
) -> float:
    """Predict mean P(conversion) with only coalition channels active.

    Non-coalition channels are set to their "absent" encoding:
        present=0, count=0, recency=1.0
    """
    X_masked = X_raw.copy()
    for ch in CHANNEL_NAMES:
        if ch not in coalition:
            for col_idx, absent_val in zip(channel_feature_map[ch], _ABSENT_VALUES):
                X_masked[:, col_idx] = absent_val

    X_scaled = scaler.transform(X_masked)
    probs = model.predict_proba(X_scaled)[:, 1]
    return float(probs.mean())


# ============================================================
# Exact Shapley Computation
# ============================================================

def _compute_exact_shapley(
    channels: Tuple[str, ...],
    value_fn,
) -> Dict[str, float]:
    """Compute exact Shapley values by enumerating all 2^n coalitions.

    phi_i = sum_{S in N\\{i}} [|S|!(n-|S|-1)!/n!] * [v(S u {i}) - v(S)]
    """
    n = len(channels)
    shapley_values = {}

    for channel in channels:
        others = [c for c in channels if c != channel]
        sv = 0.0
        for r in range(n):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                marginal = value_fn(S | {channel}) - value_fn(S)
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                sv += weight * marginal
        shapley_values[channel] = sv

    return shapley_values


# ============================================================
# Public API
# ============================================================

def compute_incremental_shapley(
    journeys: pd.DataFrame,
    sample_users: int = 5000,
    random_seed: int = 42,
) -> AttributionResult:
    """Compute Incremental Shapley with learned response model.

    Du et al. (2019) two-step pipeline:
        1. Train response model on observed data -> P(conversion | features)
        2. v(S) = model_predict(S active) - model_predict(no channels)
        3. Exact Shapley on v(S) over 7 channels (128 coalitions)

    Args:
        journeys: Long-format journey DataFrame.
        sample_users: Subsample for coalition evaluation.
        random_seed: For reproducibility.
    """
    # Train on ALL users for best model quality
    logger.info("  Building user features...")
    full_df, channel_feature_map = _build_user_features(journeys)

    logger.info("  Training response model...")
    model, scaler = _train_response_model(full_df)

    # Subsample for coalition evaluation (computational tractability)
    rng = np.random.default_rng(random_seed)
    if len(full_df) > sample_users:
        sample_idx = rng.choice(len(full_df), size=sample_users, replace=False)
        sample_df = full_df.iloc[sample_idx]
    else:
        sample_df = full_df

    feature_cols = _get_feature_columns(sample_df)
    X_raw = sample_df[feature_cols].values.astype(np.float32)

    # Pre-compute base prediction (empty coalition)
    base_pred = _predict_coalition(
        model, scaler, X_raw, frozenset(), channel_feature_map,
    )

    cache: Dict[FrozenSet[str], float] = {frozenset(): 0.0}

    def incremental_value(coalition: FrozenSet[str]) -> float:
        if coalition in cache:
            return cache[coalition]
        p_coal = _predict_coalition(
            model, scaler, X_raw, coalition, channel_feature_map,
        )
        val = max(0.0, p_coal - base_pred)
        cache[coalition] = val
        return val

    logger.info("  Computing Shapley values over 128 coalitions...")
    raw_shapley = _compute_exact_shapley(CHANNEL_NAMES, incremental_value)

    # Normalize: clamp negatives to 0, then sum to 1
    clamped = {k: max(0.0, v) for k, v in raw_shapley.items()}
    total = sum(clamped.values())
    normalized = {k: v / total for k, v in clamped.items()} if total > 0 else {
        k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES
    }

    # Full coalition prediction for reporting
    full_pred = _predict_coalition(
        model, scaler, X_raw, frozenset(CHANNEL_NAMES), channel_feature_map,
    )

    return AttributionResult(
        method="Incremental Shapley",
        channel_credits=normalized,
        channel_credits_raw=raw_shapley,
        metadata={
            "base_conversion_rate": base_pred,
            "full_coalition_rate": full_pred,
            "incremental_fraction": max(0, full_pred - base_pred) / max(full_pred, 1e-8),
            "n_coalitions": len(cache),
        },
    )
