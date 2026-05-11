"""Ground truth computation from DGP parameters.

Two ground truth definitions:

Ground Truth A (Intensity Decomposition) — Primary benchmark:
    Decompose each converted user's log-intensity into per-touchpoint contributions
    (beta * decay + cross-influence share + heterogeneity share), aggregate by channel,
    normalize to sum=1.0.

Ground Truth B (Counterfactual Shapley):
    For each coalition S of channels, recompute conversion rates from the DGP
    using only touchpoints in S. Compute exact Shapley values from 128 coalition values.
    Used specifically to validate Shapley and Incremental Shapley methods.
"""

import itertools
import json
import math
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES, DGPConfig, SegmentDef
from part1_simulation.dgp.conversion_model import (
    compute_cross_influence_bonus,
    compute_temporal_decay,
    intensity_to_conversion_prob,
)


# ============================================================
# Ground Truth A: Intensity Decomposition
# ============================================================

def _decompose_user_intensity(
    channels: List[str],
    timestamps: List[float],
    observation_time: float,
    config: DGPConfig,
    segment: SegmentDef,
) -> Dict[str, float]:
    """Decompose a single user's log-intensity into per-channel contributions.

    Returns dict of channel_name → contribution (non-negative).
    Cross-influence bonuses are split proportionally to β between source and target.
    Cross-influence decays with the source channel's temporal decay.
    Heterogeneity (eta) is distributed proportionally to channel effects.
    """
    channel_lookup = {ch.name: ch for ch in config.channels}

    # 1. Channel-specific beta * decay contributions
    channel_contributions: Dict[str, float] = {name: 0.0 for name in CHANNEL_NAMES}

    for ch_name, t_j in zip(channels, timestamps):
        ch_def = channel_lookup[ch_name]
        delta_t = max(0.0, observation_time - t_j)
        decay = compute_temporal_decay(ch_def.decay_half_life_days, delta_t)
        channel_contributions[ch_name] += ch_def.beta * decay

    # 2. Cross-influence: split by β ratio, with source temporal decay
    channel_first: Dict[str, Tuple[int, float]] = {}
    for idx, (ch, ts) in enumerate(zip(channels, timestamps)):
        if ch not in channel_first:
            channel_first[ch] = (idx, ts)

    for ci in config.cross_influences:
        src_info = channel_first.get(ci.source)
        tgt_info = channel_first.get(ci.target)
        if src_info is not None and tgt_info is not None and src_info[0] < tgt_info[0]:
            # Source channel decay applied to synergy
            source_decay = compute_temporal_decay(
                channel_lookup[ci.source].decay_half_life_days,
                max(0.0, observation_time - src_info[1]),
            )
            decayed_delta = ci.delta * source_decay
            # Split by β ratio (higher β channel gets more credit)
            src_beta = channel_lookup[ci.source].beta
            tgt_beta = channel_lookup[ci.target].beta
            total_beta = src_beta + tgt_beta
            channel_contributions[ci.source] += decayed_delta * (src_beta / total_beta)
            channel_contributions[ci.target] += decayed_delta * (tgt_beta / total_beta)

    # 3. Heterogeneity (eta): distribute proportionally to channel effects
    total_channel_effect = sum(max(0.0, v) for v in channel_contributions.values())
    if total_channel_effect > 0 and segment.eta != 0:
        eta_abs = abs(segment.eta)
        for ch_name in channel_contributions:
            weight = max(0.0, channel_contributions[ch_name]) / total_channel_effect
            channel_contributions[ch_name] += eta_abs * weight * np.sign(segment.eta)

    # Clamp negative contributions to zero (can arise from negative eta)
    return {k: max(0.0, v) for k, v in channel_contributions.items()}


def compute_ground_truth_intensity(
    journeys: pd.DataFrame,
    config: DGPConfig,
) -> Dict[str, float]:
    """Ground Truth A: intensity decomposition across all converted users.

    For each converted journey, decompose log-intensity into per-channel contributions.
    Sum across all converted users, then normalize to sum=1.0.

    Args:
        journeys: long-format journey DataFrame.
        config: DGP configuration.

    Returns:
        Dict of channel_name → normalized credit (sum = 1.0).
    """
    segment_lookup = {seg.name: seg for seg in config.segments}

    # Filter to converted users only
    converted_users = journeys.loc[journeys["converted"]].groupby("user_id", sort=False)

    total_contributions: Dict[str, float] = {name: 0.0 for name in CHANNEL_NAMES}

    for user_id, group in converted_users:
        channels = group["channel"].tolist()
        timestamps = group["timestamp"].tolist()
        observation_time = timestamps[-1]
        segment = segment_lookup[group["segment"].iloc[0]]

        user_contrib = _decompose_user_intensity(
            channels, timestamps, observation_time, config, segment,
        )
        for ch, val in user_contrib.items():
            total_contributions[ch] += val

    # Normalize
    total = sum(total_contributions.values())
    if total > 0:
        return {k: v / total for k, v in total_contributions.items()}
    return total_contributions


# ============================================================
# Ground Truth B: Counterfactual Shapley
# ============================================================

def _compute_coalition_conversion_rate(
    journeys: pd.DataFrame,
    coalition: FrozenSet[str],
    config: DGPConfig,
) -> float:
    """Compute conversion rate when only channels in the coalition are active.

    For each user, keep only touchpoints matching the coalition channels,
    recompute log-intensity, and get conversion probability.
    Returns average conversion probability across all users.
    """
    if not coalition:
        # Empty coalition: only baseline + heterogeneity
        segment_lookup = {seg.name: seg for seg in config.segments}
        user_segments = journeys.groupby("user_id")["segment"].first()
        probs = []
        for _, seg_name in user_segments.items():
            seg = segment_lookup[seg_name]
            log_i = config.alpha_0 + seg.eta
            probs.append(intensity_to_conversion_prob(log_i))
        return float(np.mean(probs))

    segment_lookup = {seg.name: seg for seg in config.segments}
    channel_lookup = {ch.name: ch for ch in config.channels}

    # Filter cross-influences to only those within the coalition
    active_ci = tuple(
        ci for ci in config.cross_influences
        if ci.source in coalition and ci.target in coalition
    )
    config_filtered = config._replace(cross_influences=active_ci)

    user_groups = journeys.groupby("user_id", sort=False)
    probs = []

    for user_id, group in user_groups:
        # Filter to coalition channels only
        mask = group["channel"].isin(coalition)
        filtered = group[mask]

        if filtered.empty:
            # User has no touchpoints in this coalition
            seg = segment_lookup[group["segment"].iloc[0]]
            log_i = config.alpha_0 + seg.eta
            probs.append(intensity_to_conversion_prob(log_i))
            continue

        channels = filtered["channel"].tolist()
        timestamps = filtered["timestamp"].tolist()
        observation_time = timestamps[-1]
        segment = segment_lookup[filtered["segment"].iloc[0]]

        from part1_simulation.dgp.conversion_model import compute_log_intensity
        log_i = compute_log_intensity(
            channels, timestamps, observation_time, config_filtered, segment,
        )
        probs.append(intensity_to_conversion_prob(log_i))

    return float(np.mean(probs))


def compute_ground_truth_shapley(
    journeys: pd.DataFrame,
    config: DGPConfig,
    channels: Tuple[str, ...] = CHANNEL_NAMES,
    sample_users: int = 5000,
) -> Dict[str, float]:
    """Ground Truth B: Exact Shapley values from counterfactual coalition values.

    With 7 channels, enumerates all 2^7 = 128 coalitions.
    Uses a subsample of users for computational tractability.

    Args:
        journeys: long-format journey DataFrame.
        config: DGP configuration.
        channels: tuple of channel names to include.
        sample_users: number of users to subsample (for speed).

    Returns:
        Dict of channel_name → Shapley value (sum = total conversion lift).
    """
    n_channels = len(channels)

    # Subsample users for speed
    all_user_ids = journeys["user_id"].unique()
    rng = np.random.default_rng(config.random_seed)
    if len(all_user_ids) > sample_users:
        sampled_ids = rng.choice(all_user_ids, size=sample_users, replace=False)
        journeys_sample = journeys[journeys["user_id"].isin(sampled_ids)]
    else:
        journeys_sample = journeys

    # Compute coalition values: v(S) = avg conversion probability with only channels in S
    coalition_values: Dict[FrozenSet[str], float] = {}

    all_coalitions = []
    for r in range(n_channels + 1):
        for combo in itertools.combinations(channels, r):
            all_coalitions.append(frozenset(combo))

    print(f"  Computing {len(all_coalitions)} coalition values "
          f"(sampled {len(journeys_sample['user_id'].unique())} users)...")

    for i, coalition in enumerate(all_coalitions):
        coalition_values[coalition] = _compute_coalition_conversion_rate(
            journeys_sample, coalition, config,
        )
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(all_coalitions)} coalitions computed")

    # Compute Shapley values
    shapley_values: Dict[str, float] = {}
    grand_coalition = frozenset(channels)

    for channel in channels:
        sv = 0.0
        others = [c for c in channels if c != channel]

        for r in range(n_channels):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                S_with_channel = S | {channel}

                # Marginal contribution
                marginal = coalition_values[S_with_channel] - coalition_values[S]

                # Shapley weight: |S|! * (n - |S| - 1)! / n!
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n_channels - len(S) - 1)
                    / math.factorial(n_channels)
                )
                sv += weight * marginal

        shapley_values[channel] = sv

    # Normalize to sum = 1.0
    total = sum(abs(v) for v in shapley_values.values())
    if total > 0:
        shapley_normalized = {k: max(0.0, v) / total for k, v in shapley_values.items()}
        # Re-normalize after clamping
        norm = sum(shapley_normalized.values())
        if norm > 0:
            shapley_normalized = {k: v / norm for k, v in shapley_normalized.items()}
        return shapley_normalized

    return {k: 1.0 / n_channels for k in channels}


# ============================================================
# Combined Ground Truth + Save
# ============================================================

def compute_all_ground_truths(
    journeys: pd.DataFrame,
    config: DGPConfig,
    sample_users_shapley: int = 5000,
) -> dict:
    """Compute both ground truth definitions and package into a dict.

    Args:
        journeys: long-format journey DataFrame.
        config: DGP configuration.
        sample_users_shapley: sample size for Shapley coalition computation.

    Returns:
        Dict containing both ground truths, DGP parameters, and data statistics.
    """
    print("Computing Ground Truth A (intensity decomposition)...")
    gt_a = compute_ground_truth_intensity(journeys, config)

    print("Computing Ground Truth B (counterfactual Shapley)...")
    gt_b = compute_ground_truth_shapley(
        journeys, config, sample_users=sample_users_shapley,
    )

    # Data statistics
    user_level = journeys.groupby("user_id").agg(
        converted=("converted", "first"),
    )

    result = {
        "ground_truth_A": {
            "method": "intensity_decomposition",
            "channel_credits": gt_a,
            "channel_ranking": sorted(gt_a, key=gt_a.get, reverse=True),
        },
        "ground_truth_B": {
            "method": "counterfactual_shapley",
            "channel_credits": gt_b,
            "channel_ranking": sorted(gt_b, key=gt_b.get, reverse=True),
        },
        "dgp_parameters": {
            "alpha_0": config.alpha_0,
            "betas": {ch.name: ch.beta for ch in config.channels},
            "decay_half_lives_days": {
                ch.name: ch.decay_half_life_days for ch in config.channels
            },
            "cross_influences": [
                [ci.source, ci.target, ci.delta] for ci in config.cross_influences
            ],
            "segment_etas": {seg.name: seg.eta for seg in config.segments},
        },
        "data_statistics": {
            "n_users": int(len(user_level)),
            "conversion_rate": float(user_level["converted"].mean()),
            "n_converters": int(user_level["converted"].sum()),
        },
    }

    return result


def save_ground_truth(ground_truth: dict, output_dir: str) -> None:
    """Save ground truth to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Ground truth saved to {output_path / 'ground_truth.json'}")
