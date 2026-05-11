"""Cox proportional hazards DGP with Weibull baseline — alternative DGP.

Hazard model:
    λ(t) = h₀(t) · exp(β^T x(t))
where:
    h₀(t) = (k/θ) (t/θ)^{k-1}        # Weibull baseline (non-log-linear in t)
    β^T x(t) = Σ_k β_k · I[channel k seen by t] + η_segment

This DGP intentionally violates the log-linear baseline assumption of Shender's
TEDDA. The baseline hazard is power-law (Weibull, shape=1.5 > 1 = increasing),
which makes step-function approximation imperfect.

Ground truth (channel credit):
    Hazard ratio (HR) — under proportional hazards, exp(β_k) is the multiplicative
    hazard increase from having seen channel k. Normalize {exp(β_k) - 1} across
    channels to sum=1.
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

# Channel β coefficients (log hazard ratio)
_CHANNEL_BETAS = {
    "Display":         0.20,
    "Social":          0.15,
    "Organic Search":  0.40,
    "Paid Search":     0.90,
    "Email":           0.65,
    "Referral":        0.30,
    "Direct":          0.55,
}

_SEGMENT_ETAS = {"New": -0.2, "Exploratory": 0.0, "Loyal": 0.4}

# Weibull baseline parameters
_WEIBULL_SHAPE = 1.5  # k > 1 → increasing hazard over time
_WEIBULL_SCALE_HOURS = 720.0  # θ — characteristic time scale (~30 days)


def _weibull_baseline_hazard(t: float) -> float:
    """h₀(t) = (k/θ) (t/θ)^{k-1}."""
    if t <= 0:
        return 1e-9
    k, theta = _WEIBULL_SHAPE, _WEIBULL_SCALE_HOURS
    return (k / theta) * (t / theta) ** (k - 1)


def _weibull_cumulative_hazard(t: float) -> float:
    """H₀(t) = (t/θ)^k."""
    if t <= 0:
        return 0.0
    return (t / _WEIBULL_SCALE_HOURS) ** _WEIBULL_SHAPE


def _compute_log_hazard_ratio(
    channels_seen: set,
    segment: str,
    alpha0: float,
) -> float:
    """β^T x = α₀ + Σ β_k I[ch k seen] + η_seg."""
    hr = alpha0 + _SEGMENT_ETAS.get(segment, 0.0)
    for ch in channels_seen:
        hr += _CHANNEL_BETAS.get(ch, 0.0)
    return hr


def _simulate_conversion_time(
    journey: pd.DataFrame,
    alpha0: float,
    rng: np.random.Generator,
    horizon_hours: float = 720.0,
) -> Tuple[bool, float]:
    """Simulate time-to-event under Cox PH with Weibull baseline.

    Conversion happens at smallest t where cumulative hazard exceeds U where
    U ~ Exp(1) (inverse-CDF method for survival simulation).

    Returns:
        (converted, conv_time_hours). If no conv within horizon, converted=False.
    """
    # Sort journey by timestamp
    journey = journey.sort_values("touchpoint_idx")
    ts = journey["timestamp"].values.astype(float)
    chs = journey["channel"].values
    segment = journey["segment"].iloc[0]

    # Survival: integrate hazard piecewise
    # Between touchpoints, channels_seen is fixed
    channels_seen: set = set()
    target_cumhaz = -np.log(rng.random())  # ~Exp(1)

    t_prev = 0.0
    cum_haz = 0.0

    for i, t_i in enumerate(ts):
        # In interval [t_prev, t_i), channels_seen is the channels with idx < i
        if i > 0:
            log_hr = _compute_log_hazard_ratio(channels_seen, segment, alpha0)
            hr = np.exp(np.clip(log_hr, -10, 10))
            seg_haz = (
                _weibull_cumulative_hazard(t_i)
                - _weibull_cumulative_hazard(t_prev)
            ) * hr
            if cum_haz + seg_haz >= target_cumhaz:
                # Conversion in this interval — invert
                # H(t) - H(t_prev) = (target - cum_haz) / hr
                target_h = (target_cumhaz - cum_haz) / hr
                # Solve (t/θ)^k - (t_prev/θ)^k = target_h
                t_conv_pow = (t_prev / _WEIBULL_SCALE_HOURS) ** _WEIBULL_SHAPE + target_h
                t_conv = _WEIBULL_SCALE_HOURS * t_conv_pow ** (1.0 / _WEIBULL_SHAPE)
                if t_conv <= horizon_hours:
                    return True, float(t_conv)
                else:
                    return False, float(horizon_hours)
            cum_haz += seg_haz

        channels_seen.add(chs[i])
        t_prev = t_i

    # After last touchpoint until horizon
    log_hr = _compute_log_hazard_ratio(channels_seen, segment, alpha0)
    hr = np.exp(np.clip(log_hr, -10, 10))
    seg_haz = (
        _weibull_cumulative_hazard(horizon_hours)
        - _weibull_cumulative_hazard(t_prev)
    ) * hr
    if cum_haz + seg_haz >= target_cumhaz:
        target_h = (target_cumhaz - cum_haz) / hr
        t_conv_pow = (t_prev / _WEIBULL_SCALE_HOURS) ** _WEIBULL_SHAPE + target_h
        t_conv = _WEIBULL_SCALE_HOURS * t_conv_pow ** (1.0 / _WEIBULL_SHAPE)
        return True, float(min(t_conv, horizon_hours))

    return False, float(horizon_hours)


def _calibrate_alpha0(
    journeys: pd.DataFrame,
    target_rate: float = 0.025,
    n_steps: int = 12,
) -> float:
    """Binary search α₀ for target conversion rate."""
    lo, hi = -10.0, 5.0
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        n_conv = 0
        n_users = journeys["user_id"].nunique()
        # Sample subset for fast calibration
        sample_uids = journeys["user_id"].drop_duplicates().sample(
            n=min(2000, n_users), random_state=0,
        ).values
        sub = journeys[journeys["user_id"].isin(sample_uids)]
        for _, group in sub.groupby("user_id", sort=False):
            converted, _ = _simulate_conversion_time(group, mid, rng)
            if converted:
                n_conv += 1
        rate = n_conv / len(sample_uids)
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _compute_ground_truth_hr(betas: Dict[str, float]) -> Dict[str, float]:
    """GT credit = (exp(β_k) - 1), normalized to sum=1."""
    raw = {ch: max(0.0, np.exp(b) - 1.0) for ch, b in betas.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def generate_dgp_cox(
    n_users: int = 20_000,
    seed: int = 42,
    target_conversion_rate: float = 0.025,
    horizon_hours: float = 720.0,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Generate journeys via Cox PH with Weibull baseline.

    Returns:
        journeys, gt_credits, metadata
    """
    rng = np.random.default_rng(seed)

    base_config = load_dgp_config(overrides=[f"n_users={n_users}", f"random_seed={seed}"])
    users_df = assign_segments(
        base_config.n_users, base_config.segments, base_config.max_touchpoints, rng,
    )
    sequences = generate_channel_sequences(users_df, base_config, rng)
    journeys = assign_timestamps(
        sequences, base_config.inter_arrival_lambda_hours, rng,
    )

    # Calibrate α₀
    alpha0 = _calibrate_alpha0(journeys, target_conversion_rate)

    # Simulate conversion per user
    converted_map: Dict[int, bool] = {}
    conv_time_map: Dict[int, float] = {}

    for uid, group in journeys.groupby("user_id", sort=False):
        converted, conv_time = _simulate_conversion_time(
            group, alpha0, rng, horizon_hours,
        )
        converted_map[uid] = converted
        conv_time_map[uid] = conv_time

    journeys = journeys.assign(
        converted=journeys["user_id"].map(converted_map).astype(bool),
        conversion_intensity=0.0,
        is_last_touchpoint=(
            journeys["touchpoint_idx"] == journeys["journey_length"] - 1
        ),
        touchpoint_cost=0.0,
    )

    for col, dtype in JOURNEY_SCHEMA.items():
        if col in journeys.columns:
            try:
                journeys[col] = journeys[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    gt_credits = _compute_ground_truth_hr(_CHANNEL_BETAS)

    n_converted = sum(converted_map.values())
    metadata = {
        "dgp_name": "cox_weibull",
        "n_users": n_users,
        "seed": seed,
        "alpha0": float(alpha0),
        "weibull_shape": _WEIBULL_SHAPE,
        "weibull_scale_hours": _WEIBULL_SCALE_HOURS,
        "channel_betas": dict(_CHANNEL_BETAS),
        "segment_etas": dict(_SEGMENT_ETAS),
        "n_converted": int(n_converted),
        "conversion_rate": float(n_converted / n_users),
    }
    return journeys, gt_credits, metadata


if __name__ == "__main__":
    j, gt, meta = generate_dgp_cox(n_users=3000, seed=42)
    print(f"shape: {j.shape}, users: {j.user_id.nunique()}")
    print(f"conv rate: {meta['conversion_rate']:.4f}")
    print(f"alpha0: {meta['alpha0']:.3f}")
    print(f"GT sum: {sum(gt.values()):.4f}")
    print("GT credits (sorted):")
    for k, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.4f}")
