"""Hawkes process DGP — alternative self-exciting probability process.

Conversion intensity (multivariate Hawkes):
    λ(t) = μ + Σ_{t_j < t} α_{c(t_j)} · exp(-β · (t - t_j))

where:
    μ: baseline conversion rate (no ad effect)
    α_k: excitation magnitude when channel-k touchpoint occurs
    β: decay rate of excitation
    c(t_j): channel of touchpoint j

This model has *temporal clustering* — recent touchpoints excite future
conversion intensity. Differs from Shender's TEDDA in that the excitation
decay is exponential (not step function) and conversion times are explicit
(not derived from per-interval Poisson).

Ground truth (channel credit):
    For each channel k, expected total excitation contribution to cumulative
    hazard up to conversion time, integrated over user paths:
        credit_k ∝ E[Σ_{t_j: c(t_j)=k} α_k · (1 - exp(-β·(τ - t_j))) / β]
    Normalized to sum=1.
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

# Channel excitation magnitudes (α_k)
_CHANNEL_ALPHAS = {
    "Display":         0.0008,
    "Social":          0.0006,
    "Organic Search":  0.0020,
    "Paid Search":     0.0050,
    "Email":           0.0035,
    "Referral":        0.0015,
    "Direct":          0.0028,
}

_BETA_DECAY = 1.0 / 24.0  # 1/day decay rate (exp half-life ~16h)
_BASELINE_MU = 1e-5  # very low baseline (almost no spontaneous conversion)
_HORIZON_HOURS = 720.0  # 30 days

_SEGMENT_MULTIPLIERS = {"New": 0.85, "Exploratory": 1.0, "Loyal": 1.25}


def _simulate_hawkes_user(
    journey: pd.DataFrame,
    alphas: Dict[str, float],
    beta: float,
    mu: float,
    horizon: float,
    rng: np.random.Generator,
) -> Tuple[bool, float]:
    """Simulate first conversion under Hawkes process via Ogata thinning.

    Returns:
        (converted, conv_time_hours)
    """
    journey = journey.sort_values("touchpoint_idx")
    ts = journey["timestamp"].values.astype(float)
    chs = journey["channel"].values
    segment = journey["segment"].iloc[0]
    seg_mult = _SEGMENT_MULTIPLIERS.get(str(segment), 1.0)

    # Ogata thinning: at each step, compute upper bound λ_max,
    # propose candidate from Exp(λ_max), accept with prob λ(t)/λ_max
    t = 0.0
    while t < horizon:
        # Compute current intensity λ(t)
        active_excitation = 0.0
        for j, t_j in enumerate(ts):
            if t_j > t:
                break
            active_excitation += (
                alphas.get(chs[j], 0.0) * np.exp(-beta * (t - t_j))
            )
        lam_t = (mu + active_excitation) * seg_mult

        # Upper bound at next touchpoint or +1 day, whichever sooner
        # (intensity only decreases between touchpoints — exp decay)
        lam_max = lam_t  # decreasing → current is max for the next interval
        if lam_max <= 1e-15:
            t += 1.0  # skip ahead
            continue

        dt = rng.exponential(1.0 / lam_max)
        t_candidate = t + dt
        if t_candidate >= horizon:
            return False, horizon

        # Compute λ(t_candidate)
        active_excitation_new = 0.0
        for j, t_j in enumerate(ts):
            if t_j > t_candidate:
                break
            active_excitation_new += (
                alphas.get(chs[j], 0.0) * np.exp(-beta * (t_candidate - t_j))
            )
        lam_new = (mu + active_excitation_new) * seg_mult

        # Accept/reject
        if rng.random() < lam_new / lam_max:
            return True, float(t_candidate)
        t = t_candidate

    return False, horizon


def _compute_ground_truth_excitation(
    journeys: pd.DataFrame,
    alphas: Dict[str, float],
    beta: float,
    horizon: float,
) -> Dict[str, float]:
    """GT credit_k = mean over users of Σ_{j:c=k} α_k · (1 - exp(-β·(τ-t_j))) / β.

    This is the expected cumulative excitation contribution from channel k
    integrated over [t_j, horizon].
    """
    raw: Dict[str, float] = {ch: 0.0 for ch in CHANNEL_NAMES}
    # Convert to plain str arrays to avoid categorical comparison issues
    chs_all = journeys["channel"].astype(str).values
    ts_all = journeys["timestamp"].values.astype(float)
    for t_j, ch in zip(ts_all, chs_all):
        alpha_k = alphas.get(ch, 0.0)
        # Clamp dt to non-negative (touchpoints after horizon contribute 0)
        dt = max(0.0, horizon - t_j)
        integrated = alpha_k * (1.0 - np.exp(-beta * dt)) / beta
        raw[ch] += integrated

    total = sum(raw.values())
    if total > 0:
        return {k: v / total for k, v in raw.items()}
    return {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}


def _calibrate_alpha_scale(
    journeys: pd.DataFrame,
    target_rate: float,
    horizon: float,
    n_steps: int = 8,
) -> float:
    """Binary search a global multiplier on α_k to hit target conversion rate."""
    rng = np.random.default_rng(0)
    sample_uids = (
        journeys["user_id"].drop_duplicates()
        .sample(n=min(1500, journeys["user_id"].nunique()), random_state=0)
        .values
    )
    sub = journeys[journeys["user_id"].isin(sample_uids)]

    lo, hi = 0.1, 100.0  # multiplier range
    for _ in range(n_steps):
        mid = np.sqrt(lo * hi)  # geometric midpoint
        scaled_alphas = {k: v * mid for k, v in _CHANNEL_ALPHAS.items()}
        n_conv = 0
        for _, group in sub.groupby("user_id", sort=False):
            converted, _ = _simulate_hawkes_user(
                group, scaled_alphas, _BETA_DECAY, _BASELINE_MU, horizon, rng,
            )
            if converted:
                n_conv += 1
        rate = n_conv / len(sample_uids)
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def generate_dgp_hawkes(
    n_users: int = 20_000,
    seed: int = 42,
    target_conversion_rate: float = 0.025,
    horizon_hours: float = _HORIZON_HOURS,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Generate journeys via multivariate self-exciting Hawkes process.

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

    # Calibrate alpha multiplier
    alpha_scale = _calibrate_alpha_scale(journeys, target_conversion_rate, horizon_hours)
    alphas = {k: v * alpha_scale for k, v in _CHANNEL_ALPHAS.items()}

    # Simulate conversions
    converted_map: Dict[int, bool] = {}
    for uid, group in journeys.groupby("user_id", sort=False):
        converted, _ = _simulate_hawkes_user(
            group, alphas, _BETA_DECAY, _BASELINE_MU, horizon_hours, rng,
        )
        converted_map[uid] = converted

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

    gt_credits = _compute_ground_truth_excitation(
        journeys, alphas, _BETA_DECAY, horizon_hours,
    )

    n_converted = sum(converted_map.values())
    metadata = {
        "dgp_name": "hawkes",
        "n_users": n_users,
        "seed": seed,
        "alpha_scale": alpha_scale,
        "channel_alphas": alphas,
        "beta_decay": _BETA_DECAY,
        "baseline_mu": _BASELINE_MU,
        "n_converted": int(n_converted),
        "conversion_rate": float(n_converted / n_users),
    }
    return journeys, gt_credits, metadata


if __name__ == "__main__":
    j, gt, meta = generate_dgp_hawkes(n_users=2000, seed=42)
    print(f"shape: {j.shape}, users: {j.user_id.nunique()}")
    print(f"conv rate: {meta['conversion_rate']:.4f}")
    print(f"alpha_scale: {meta['alpha_scale']:.3f}")
    print(f"GT sum: {sum(gt.values()):.4f}")
    print("GT credits (sorted):")
    for k, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.4f}")
