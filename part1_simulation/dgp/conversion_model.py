"""Log-linear conversion intensity model (ground truth core).

Implements the integrated conversion model from three frameworks:
- Du et al. (2019): user heterogeneity (d_i · η)
- Shender et al. (2023): log-linear intensity with temporal decay f_channel(Δt)
- CDA (2025): cross-channel influence (δ_ij)

Conversion model:
    log(λ_i(t)) = α₀
        + Σⱼ βₖ · exp(-Δt / (half_life_k × 24))   # channel effects with decay
        + Σ δᵢⱼ · f_source(Δt) · I[source before target]  # cross-influence with decay
        + η_segment                                   # user heterogeneity

    P(conversion) = 1 - exp(-exp(log_intensity))     # Poisson process realization
"""

from typing import Dict, List, Tuple

import numpy as np

from part1_simulation import ChannelDef, CrossInfluence, DGPConfig, SegmentDef


def compute_temporal_decay(
    decay_half_life_days: float,
    delta_t_hours: float,
) -> float:
    """Compute temporal decay factor for a channel touchpoint.

    f(Δt) = exp(-Δt / (half_life_days × 24))

    Args:
        decay_half_life_days: channel-specific half-life in days.
        delta_t_hours: time elapsed since touchpoint (in hours).

    Returns:
        Decay factor in (0, 1].
    """
    half_life_hours = decay_half_life_days * 24.0
    return np.exp(-delta_t_hours / half_life_hours)


def compute_cross_influence_bonus(
    journey_channels: List[str],
    journey_timestamps: List[float],
    observation_time: float,
    cross_influences: Tuple[CrossInfluence, ...],
    channel_defs: Tuple[ChannelDef, ...],
) -> float:
    """Sum cross-influence deltas with source-channel temporal decay.

    Activated when source appears before target in the journey sequence.
    The delta is modulated by the source channel's temporal decay at
    observation time, ensuring synergy fades as the source effect fades.

    δ_effective = δ_ij × f_source(t_obs - t_source)

    Args:
        journey_channels: ordered list of channel names in the journey.
        journey_timestamps: corresponding timestamps (hours).
        observation_time: time at which to evaluate (typically last touchpoint).
        cross_influences: tuple of CrossInfluence definitions.
        channel_defs: tuple of ChannelDef (for decay half-lives).

    Returns:
        Total cross-influence bonus to add to log-intensity.
    """
    if not cross_influences:
        return 0.0

    channel_half_lives = {ch.name: ch.decay_half_life_days for ch in channel_defs}

    # Build first-occurrence index and timestamp per channel
    channel_first: Dict[str, Tuple[int, float]] = {}
    for idx, (ch, ts) in enumerate(zip(journey_channels, journey_timestamps)):
        if ch not in channel_first:
            channel_first[ch] = (idx, ts)

    bonus = 0.0
    for ci in cross_influences:
        src_info = channel_first.get(ci.source)
        tgt_info = channel_first.get(ci.target)
        if src_info is not None and tgt_info is not None and src_info[0] < tgt_info[0]:
            # Apply source channel's temporal decay to the synergy delta
            source_timestamp = src_info[1]
            delta_t = max(0.0, observation_time - source_timestamp)
            source_half_life = channel_half_lives.get(ci.source, 7.0)
            decay = compute_temporal_decay(source_half_life, delta_t)
            bonus += ci.delta * decay

    return bonus


def compute_log_intensity(
    touchpoint_channels: List[str],
    touchpoint_timestamps: List[float],
    observation_time: float,
    config: DGPConfig,
    segment: SegmentDef,
) -> float:
    """Compute the full log-intensity log(λ_i(t)) for a user's journey.

    Args:
        touchpoint_channels: ordered channel names for each touchpoint.
        touchpoint_timestamps: corresponding timestamps (hours from journey start).
        observation_time: time at which to evaluate intensity (typically last touchpoint).
        config: DGP configuration with channel betas, decay rates, etc.
        segment: user's segment definition (provides eta for heterogeneity).

    Returns:
        log(λ_i(t)) scalar value.
    """
    # Build channel name → ChannelDef lookup
    channel_lookup: Dict[str, ChannelDef] = {ch.name: ch for ch in config.channels}

    # Base intensity
    log_intensity = config.alpha_0

    # Channel effects with temporal decay: Σⱼ βₖ · f_channel(t - tⱼ)
    for ch_name, t_j in zip(touchpoint_channels, touchpoint_timestamps):
        ch_def = channel_lookup[ch_name]
        delta_t = observation_time - t_j
        decay = compute_temporal_decay(ch_def.decay_half_life_days, max(0.0, delta_t))
        log_intensity += ch_def.beta * decay

    # Cross-channel influence: Σ δ_ij · f_source(Δt) · I[source before target]
    log_intensity += compute_cross_influence_bonus(
        touchpoint_channels, touchpoint_timestamps, observation_time,
        config.cross_influences, config.channels,
    )

    # User heterogeneity: η_segment
    log_intensity += segment.eta

    return log_intensity


def intensity_to_conversion_prob(log_intensity: float) -> float:
    """Convert log-intensity to conversion probability via Poisson process.

    P(conversion) = 1 - exp(-λ) = 1 - exp(-exp(log_intensity))

    Numerically stable: clamp log_intensity to avoid overflow.
    """
    clamped = min(log_intensity, 10.0)  # exp(10) ≈ 22026, safe
    lam = np.exp(clamped)
    return 1.0 - np.exp(-lam)


def decide_conversion(log_intensity: float, rng: np.random.Generator) -> bool:
    """Make a Bernoulli conversion decision based on log-intensity.

    Args:
        log_intensity: log(λ_i(t)) from compute_log_intensity.
        rng: numpy random generator.

    Returns:
        True if user converts, False otherwise.
    """
    prob = intensity_to_conversion_prob(log_intensity)
    return bool(rng.random() < prob)


def compute_log_intensity_vectorized(
    touchpoint_channels_list: List[List[str]],
    touchpoint_timestamps_list: List[List[float]],
    observation_times: List[float],
    config: DGPConfig,
    segments: List[SegmentDef],
) -> np.ndarray:
    """Vectorized version for batch computation across multiple users.

    Args:
        touchpoint_channels_list: list of channel sequences (one per user).
        touchpoint_timestamps_list: list of timestamp sequences.
        observation_times: evaluation time per user.
        config: DGP configuration.
        segments: segment definition per user.

    Returns:
        Array of log-intensity values, one per user.
    """
    return np.array([
        compute_log_intensity(channels, timestamps, obs_time, config, seg)
        for channels, timestamps, obs_time, seg
        in zip(
            touchpoint_channels_list,
            touchpoint_timestamps_list,
            observation_times,
            segments,
        )
    ])
