"""Multivariate user-feature DGP — extends canonical Poisson backbone with a
second user-level feature (`device`) on top of `segment`.

Conversion model (extension of canonical Eq):
    log(λ_i(t)) = α₀
        + Σⱼ βₖ · exp(-Δt / (half_life_k · 24))   # channel decay (canonical)
        + Σ δᵢⱼ · f_source(Δt)                    # cross-influence (canonical)
        + η_segment                               # canonical heterogeneity
        + η_device                                # NEW — multivariate feature

Purpose: probe whether multivariate Survival/Poisson recovers per-channel
attribution more accurately when *both* user features genuinely shift the
baseline. Compare:
    (a) compute_survival_attribution(j, user_features=("segment",))           # univariate
    (b) compute_survival_attribution(j, user_features=("segment", "device"))  # multivariate
    (c) compute_survival_attribution(j, user_features=("device",))            # omit segment
    (d) compute_survival_attribution(j, user_features=())                     # both omitted

Hypothesis (H_recovery): MAE(b) < MAE(a) and MAE(b) < MAE(c), MAE(b) < MAE(d).

Ground truth (channel credits): same intensity-decomposition logic as Ground
Truth A in `evaluation/ground_truth.py`, but the per-user heterogeneity
distributed to channels is the *combined* η_segment + η_device.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES, JOURNEY_SCHEMA
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.conversion_model import (
    compute_log_intensity,
    compute_temporal_decay,
    intensity_to_conversion_prob,
)
from part1_simulation.dgp.generate_data import (
    assign_timestamps,
    generate_channel_sequences,
)
from part1_simulation.dgp.user_segments import assign_segments


DEFAULT_DEVICE_ETAS: Dict[str, float] = {"desktop": 0.0, "mobile": -0.4, "tablet": 0.3}
DEFAULT_DEVICE_PROPORTIONS: Dict[str, float] = {"desktop": 0.45, "mobile": 0.40, "tablet": 0.15}


# ============================================================
# Device assignment (per-user)
# ============================================================

def assign_devices(
    user_ids: np.ndarray,
    device_proportions: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[int, str]:
    """Sample one device per user with the given multinomial proportions."""
    devices = list(device_proportions.keys())
    probs = np.array([device_proportions[d] for d in devices], dtype=float)
    probs /= probs.sum()
    sampled = rng.choice(devices, size=len(user_ids), p=probs)
    return dict(zip(user_ids.tolist(), sampled.tolist()))


# ============================================================
# Per-user log-intensity with multivariate eta
# ============================================================

def _user_log_intensity_multivariate(
    group: pd.DataFrame,
    device: str,
    config,
    segment_lookup: Dict[str, Any],
    device_etas: Dict[str, float],
) -> Tuple[float, float]:
    """Compute (log_intensity, observation_time) for one user under the
    multivariate DGP. Returns (log_lambda, t_obs)."""
    channels = group["channel"].tolist()
    timestamps = group["timestamp"].tolist()
    observation_time = float(timestamps[-1])
    seg_def = segment_lookup[group["segment"].iloc[0]]
    base_log = compute_log_intensity(
        channels, timestamps, observation_time, config, seg_def,
    )
    return base_log + device_etas.get(device, 0.0), observation_time


# ============================================================
# α₀ calibration (binary search) for multivariate model
# ============================================================

def _calibrate_alpha0(
    journeys: pd.DataFrame,
    config_template,
    user_devices: Dict[int, str],
    device_etas: Dict[str, float],
    target_rate: float,
    max_iter: int = 30,
    tol: float = 0.001,
) -> float:
    """Binary search on α₀ so that empirical conversion rate ≈ target."""
    segment_lookup = {seg.name: seg for seg in config_template.segments}
    user_groups = list(journeys.groupby("user_id", sort=False))

    def _rate(alpha0: float) -> float:
        cfg = config_template._replace(alpha_0=alpha0)
        n_conv = 0
        for uid, g in user_groups:
            log_l, _ = _user_log_intensity_multivariate(
                g, user_devices[int(uid)], cfg, segment_lookup, device_etas,
            )
            p = intensity_to_conversion_prob(log_l)
            n_conv += p  # use expected count for stable, deterministic calibration
        return n_conv / len(user_groups)

    lo, hi = -12.0, 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = _rate(mid)
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================
# Ground truth — intensity decomposition with multivariate η
# ============================================================

def _decompose_user_intensity_multivariate(
    channels: List[str],
    timestamps: List[float],
    observation_time: float,
    config,
    segment,
    device_eta: float,
) -> Dict[str, float]:
    """Per-user channel-credit decomposition that distributes (η_segment + η_device)
    proportionally to channel effects (mirrors evaluation/ground_truth.py:_decompose_user_intensity).
    """
    channel_lookup = {ch.name: ch for ch in config.channels}

    contributions: Dict[str, float] = {name: 0.0 for name in CHANNEL_NAMES}

    # Channel β · decay
    for ch_name, t_j in zip(channels, timestamps):
        ch_def = channel_lookup[ch_name]
        delta_t = max(0.0, observation_time - t_j)
        decay = compute_temporal_decay(ch_def.decay_half_life_days, delta_t)
        contributions[ch_name] += ch_def.beta * decay

    # Cross-influence split by β ratio with source decay
    channel_first: Dict[str, Tuple[int, float]] = {}
    for idx, (ch, ts) in enumerate(zip(channels, timestamps)):
        if ch not in channel_first:
            channel_first[ch] = (idx, ts)
    for ci in config.cross_influences:
        src_info = channel_first.get(ci.source)
        tgt_info = channel_first.get(ci.target)
        if src_info is not None and tgt_info is not None and src_info[0] < tgt_info[0]:
            source_decay = compute_temporal_decay(
                channel_lookup[ci.source].decay_half_life_days,
                max(0.0, observation_time - src_info[1]),
            )
            decayed_delta = ci.delta * source_decay
            src_beta = channel_lookup[ci.source].beta
            tgt_beta = channel_lookup[ci.target].beta
            total_beta = src_beta + tgt_beta
            contributions[ci.source] += decayed_delta * (src_beta / total_beta)
            contributions[ci.target] += decayed_delta * (tgt_beta / total_beta)

    # Combined heterogeneity (segment + device) distributed proportionally
    combined_eta = segment.eta + device_eta
    total_channel_effect = sum(max(0.0, v) for v in contributions.values())
    if total_channel_effect > 0 and combined_eta != 0.0:
        eta_abs = abs(combined_eta)
        for ch_name in contributions:
            weight = max(0.0, contributions[ch_name]) / total_channel_effect
            contributions[ch_name] += eta_abs * weight * np.sign(combined_eta)

    return {k: max(0.0, v) for k, v in contributions.items()}


def _compute_ground_truth(
    journeys: pd.DataFrame,
    config,
    user_devices: Dict[int, str],
    device_etas: Dict[str, float],
) -> Dict[str, float]:
    """Aggregate per-user channel decomposition over converted users → normalize."""
    segment_lookup = {seg.name: seg for seg in config.segments}
    converted = journeys.loc[journeys["converted"]].groupby("user_id", sort=False)
    totals: Dict[str, float] = {name: 0.0 for name in CHANNEL_NAMES}
    for uid, g in converted:
        channels = g["channel"].tolist()
        timestamps = g["timestamp"].tolist()
        seg = segment_lookup[g["segment"].iloc[0]]
        device = user_devices[int(uid)]
        contrib = _decompose_user_intensity_multivariate(
            channels, timestamps, float(timestamps[-1]), config, seg,
            device_etas.get(device, 0.0),
        )
        for ch, v in contrib.items():
            totals[ch] += v
    total = sum(totals.values())
    if total > 0:
        return {k: v / total for k, v in totals.items()}
    return {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}


# ============================================================
# Public API
# ============================================================

def generate_dgp_multivariate(
    n_users: int = 20_000,
    seed: int = 42,
    target_conversion_rate: float = 0.025,
    device_etas: Optional[Dict[str, float]] = None,
    device_proportions: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Generate journeys under the canonical Poisson DGP plus a per-user device
    feature with its own η_device intercept shift.

    Args:
        n_users: number of users.
        seed: RNG seed.
        target_conversion_rate: target empirical conversion rate (α₀ calibrated).
        device_etas: dict mapping device level → η intercept shift (log scale).
            Default: desktop 0.0 (reference), mobile -0.4, tablet 0.3.
        device_proportions: dict mapping device level → user proportion (sums to 1).
            Default: desktop 0.45, mobile 0.40, tablet 0.15.

    Returns:
        journeys: long-format DataFrame with all canonical columns + `device`.
        gt_credits: channel → normalized credit (sum=1).
        metadata: dict of params + calibration log.
    """
    if device_etas is None:
        device_etas = dict(DEFAULT_DEVICE_ETAS)
    if device_proportions is None:
        device_proportions = dict(DEFAULT_DEVICE_PROPORTIONS)

    rng = np.random.default_rng(seed)

    # Step 1-3: canonical pipeline (segments → sequences → timestamps)
    base_config = load_dgp_config(overrides=[f"n_users={n_users}", f"random_seed={seed}"])
    users_df = assign_segments(
        base_config.n_users, base_config.segments, base_config.max_touchpoints, rng,
    )
    sequences = generate_channel_sequences(users_df, base_config, rng)
    journeys = assign_timestamps(
        sequences, base_config.inter_arrival_lambda_hours, rng,
    )

    # Step 4: device assignment (user-level)
    user_ids = users_df["user_id"].astype(np.int64).to_numpy()
    user_devices = assign_devices(user_ids, device_proportions, rng)
    journeys = journeys.assign(
        device=journeys["user_id"].map(user_devices).astype("category"),
    )

    # Step 5: α₀ calibration (binary search) under multivariate intensity
    alpha0 = _calibrate_alpha0(
        journeys, base_config, user_devices, device_etas,
        target_rate=target_conversion_rate,
    )
    config = base_config._replace(alpha_0=alpha0)

    # Step 6: per-user conversion decision (Bernoulli draw)
    segment_lookup = {seg.name: seg for seg in config.segments}
    converted_map: Dict[int, bool] = {}
    intensity_map: Dict[int, float] = {}
    for uid, g in journeys.groupby("user_id", sort=False):
        log_l, _ = _user_log_intensity_multivariate(
            g, user_devices[int(uid)], config, segment_lookup, device_etas,
        )
        p = intensity_to_conversion_prob(log_l)
        converted_map[int(uid)] = bool(rng.random() < p)
        intensity_map[int(uid)] = float(log_l)

    journeys = journeys.assign(
        converted=journeys["user_id"].map(converted_map).astype(bool),
        conversion_intensity=journeys["user_id"].map(intensity_map).astype(np.float64),
        is_last_touchpoint=(
            journeys["touchpoint_idx"] == journeys["journey_length"] - 1
        ),
        touchpoint_cost=0.0,
    )

    # Cast to schema (best-effort — `device` not in JOURNEY_SCHEMA, kept as category)
    for col, dtype in JOURNEY_SCHEMA.items():
        if col in journeys.columns:
            try:
                journeys[col] = journeys[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    # Ground truth: intensity decomposition with combined η
    gt_credits = _compute_ground_truth(journeys, config, user_devices, device_etas)

    n_converted = int(sum(converted_map.values()))
    actual_rate = n_converted / len(converted_map)
    device_counts = (
        pd.Series(user_devices).value_counts().to_dict()
    )
    metadata = {
        "dgp_name": "multivariate",
        "n_users": int(n_users),
        "seed": int(seed),
        "alpha0": float(alpha0),
        "device_etas": dict(device_etas),
        "device_proportions": dict(device_proportions),
        "device_counts": {str(k): int(v) for k, v in device_counts.items()},
        "n_converted": n_converted,
        "conversion_rate": float(actual_rate),
        "target_conversion_rate": float(target_conversion_rate),
    }
    return journeys, gt_credits, metadata


if __name__ == "__main__":
    j, gt, meta = generate_dgp_multivariate(n_users=5000, seed=42)
    print(f"shape: {j.shape}, users: {j.user_id.nunique()}")
    print(f"conv rate: {meta['conversion_rate']:.4f} (target {meta['target_conversion_rate']})")
    print(f"alpha0: {meta['alpha0']:.3f}")
    print(f"device dist: {meta['device_counts']}")
    print(f"GT sum: {sum(gt.values()):.4f}")
    print("GT credits (sorted):")
    for k, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.4f}")
