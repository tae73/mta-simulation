"""Logistic-response DGP — alternative probability process.

Conversion model:
    P(convert | user) = sigmoid(α₀ + Σ_k w_k · count_k(user) + η_segment)

where count_k = number of channel-k touchpoints for the user.
This DGP naturally favors LR-based methods (Du Incremental Shapley with LR
response, Shapley model-based) — included to test if Survival/Poisson methods
remain competitive when the underlying generating process matches LR's
functional form.

Ground truth (channel credit):
    Marginal contribution of channel k under counterfactual ablation:
        credit_k = E_user[P(conv | full) - P(conv | count_k = 0)]
    Then normalize to sum=1.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES, JOURNEY_SCHEMA
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import (
    assign_timestamps,
    generate_channel_sequences,
)
from part1_simulation.dgp.user_segments import assign_segments


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _channel_weights() -> Dict[str, float]:
    """Channel weights (logit scale). Roughly proportional to original DGP β values
    so credit ranking is similar (Paid Search > Email > Direct > ... > Social)."""
    return {
        "Display": 0.20,
        "Social": 0.15,
        "Organic Search": 0.35,
        "Paid Search": 0.80,
        "Email": 0.55,
        "Referral": 0.30,
        "Direct": 0.45,
    }


def _segment_intercepts() -> Dict[str, float]:
    """Per-segment intercept additive to logit (η equivalent)."""
    return {"New": -0.3, "Exploratory": 0.0, "Loyal": 0.5}


def _calibrate_alpha0(
    counts_df: pd.DataFrame,
    segments: pd.Series,
    weights: Dict[str, float],
    seg_eta: Dict[str, float],
    target_rate: float = 0.025,
    n_steps: int = 30,
) -> float:
    """Binary search α₀ so empirical conversion rate ≈ target."""
    # Pre-compute logit base for each user
    logit_base = np.zeros(len(counts_df), dtype=np.float64)
    for ch, w in weights.items():
        logit_base += w * counts_df[ch].values
    logit_base += segments.astype(str).map(seg_eta).values

    lo, hi = -10.0, 5.0
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        rate = _sigmoid(mid + logit_base).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _per_user_channel_counts(journeys: pd.DataFrame) -> pd.DataFrame:
    """User × channel count matrix (users sorted by user_id)."""
    counts = (
        journeys.groupby(["user_id", "channel"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for ch in CHANNEL_NAMES:
        if ch not in counts.columns:
            counts[ch] = 0
    return counts[list(CHANNEL_NAMES)].sort_index()


def _compute_ground_truth(
    counts: pd.DataFrame,
    segments: pd.Series,
    weights: Dict[str, float],
    seg_eta: Dict[str, float],
    alpha0: float,
) -> Dict[str, float]:
    """GT credit = mean P(conv | full) − P(conv | mask channel k) over all users."""
    # Logit base
    logit_full = np.full(len(counts), alpha0, dtype=np.float64)
    for ch, w in weights.items():
        logit_full += w * counts[ch].values
    logit_full += segments.reindex(counts.index).astype(str).map(seg_eta).values
    p_full = _sigmoid(logit_full)

    raw_credit: Dict[str, float] = {}
    for ch_target in CHANNEL_NAMES:
        # Counterfactual: zero out channel target
        logit_cf = logit_full - weights[ch_target] * counts[ch_target].values
        p_cf = _sigmoid(logit_cf)
        raw_credit[ch_target] = float((p_full - p_cf).mean())

    total = sum(raw_credit.values())
    if total > 0:
        return {k: v / total for k, v in raw_credit.items()}
    return {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}


def generate_dgp_logistic(
    n_users: int = 20_000,
    seed: int = 42,
    target_conversion_rate: float = 0.025,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Generate journeys with logistic-response conversion model.

    Returns:
        journeys: long-format DataFrame (JOURNEY_SCHEMA-compatible)
        gt_credits: channel → normalized credit (sum=1)
        metadata: dict of params + calibration log
    """
    rng = np.random.default_rng(seed)

    # Reuse existing sequence/timestamp generation
    base_config = load_dgp_config(overrides=[f"n_users={n_users}", f"random_seed={seed}"])
    users_df = assign_segments(
        base_config.n_users, base_config.segments, base_config.max_touchpoints, rng,
    )
    sequences = generate_channel_sequences(users_df, base_config, rng)
    journeys = assign_timestamps(
        sequences, base_config.inter_arrival_lambda_hours, rng,
    )

    # Compute logistic conversion
    counts = _per_user_channel_counts(journeys)
    user_segments = (
        journeys.groupby("user_id", observed=True)["segment"]
        .first()
        .reindex(counts.index)
    )
    weights = _channel_weights()
    seg_eta = _segment_intercepts()

    alpha0 = _calibrate_alpha0(
        counts, user_segments, weights, seg_eta, target_conversion_rate,
    )

    # Compute logit per user
    logit = np.full(len(counts), alpha0, dtype=np.float64)
    for ch, w in weights.items():
        logit += w * counts[ch].values
    logit += user_segments.astype(str).map(seg_eta).values

    p_conv = _sigmoid(logit)
    converted_per_user = rng.random(len(counts)) < p_conv
    converted_map = dict(zip(counts.index, converted_per_user))

    journeys = journeys.assign(
        converted=journeys["user_id"].map(converted_map).astype(bool),
        conversion_intensity=journeys["user_id"].map(
            dict(zip(counts.index, p_conv))
        ).astype(np.float64),
        is_last_touchpoint=(
            journeys["touchpoint_idx"] == journeys["journey_length"] - 1
        ),
        touchpoint_cost=0.0,
    )

    # Cast to schema
    for col, dtype in JOURNEY_SCHEMA.items():
        if col in journeys.columns:
            try:
                journeys[col] = journeys[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    # Ground truth
    gt_credits = _compute_ground_truth(
        counts, user_segments, weights, seg_eta, alpha0,
    )

    n_converted = int(converted_per_user.sum())
    actual_rate = n_converted / len(counts)
    metadata = {
        "dgp_name": "logistic",
        "n_users": n_users,
        "seed": seed,
        "alpha0": float(alpha0),
        "channel_weights": weights,
        "segment_eta": seg_eta,
        "n_converted": n_converted,
        "conversion_rate": float(actual_rate),
        "target_conversion_rate": target_conversion_rate,
    }
    return journeys, gt_credits, metadata


if __name__ == "__main__":
    j, gt, meta = generate_dgp_logistic(n_users=5000, seed=42)
    print(f"shape: {j.shape}, users: {j.user_id.nunique()}")
    print(f"conv rate: {meta['conversion_rate']:.4f} (target {meta['target_conversion_rate']})")
    print(f"alpha0: {meta['alpha0']:.3f}")
    print(f"GT sum: {sum(gt.values()):.4f}")
    print("GT credits (sorted):")
    for k, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.4f}")
