"""Ground truth optimal budget allocation from known DGP parameters.

Approach A (Linear Response):
    efficiency_k = β_k × E[f_k(Δt)] / c_k
    Optimal allocation: proportional to efficiency (paid channels only).

The ground truth is derived entirely from DGP parameters (β, decay half-lives)
and cost parameters (base_cost_per_touchpoint), with no learned components.
"""

from typing import Dict

import numpy as np
import pandas as pd

from part1_simulation import BudgetConfig, CHANNEL_NAMES, DGPConfig
from part1_simulation.dgp.conversion_model import compute_temporal_decay


def compute_channel_marginal_effect(
    config: DGPConfig,
    journeys: pd.DataFrame,
) -> Dict[str, float]:
    """Compute marginal conversion effect per touchpoint for each channel.

    marginal_effect_k = β_k × E[f_k(Δt)]

    where E[f_k(Δt)] is the average temporal decay factor for channel k,
    computed from actual journey data.

    Args:
        config: DGP configuration with known β and decay parameters.
        journeys: long-format journey DataFrame with timestamps.

    Returns:
        Dict of channel_name → marginal effect (non-negative).
    """
    channel_lookup = {ch.name: ch for ch in config.channels}

    # Compute average decay per channel from observed data
    # For each touchpoint, Δt = observation_time - touchpoint_time
    last_ts = journeys.groupby("user_id", sort=False)["timestamp"].last()
    journeys_with_obs = journeys.assign(
        observation_time=journeys["user_id"].map(last_ts),
    )
    journeys_with_obs = journeys_with_obs.assign(
        delta_t=lambda df: np.maximum(0.0, df["observation_time"] - df["timestamp"]),
    )

    marginal_effects: Dict[str, float] = {}

    for ch_name in CHANNEL_NAMES:
        ch_def = channel_lookup[ch_name]
        ch_mask = journeys_with_obs["channel"].astype(str) == ch_name
        ch_deltas = journeys_with_obs.loc[ch_mask, "delta_t"]

        if len(ch_deltas) == 0:
            marginal_effects[ch_name] = 0.0
            continue

        # Average decay factor across all touchpoints of this channel
        avg_decay = float(np.mean(
            list(map(
                lambda dt: compute_temporal_decay(ch_def.decay_half_life_days, dt),
                ch_deltas.values,
            ))
        ))
        marginal_effects[ch_name] = ch_def.beta * avg_decay

    return marginal_effects


def compute_channel_efficiency(
    marginal_effects: Dict[str, float],
    budget_config: BudgetConfig,
) -> Dict[str, float]:
    """Compute efficiency = marginal_effect / cost for paid channels.

    Zero-cost channels are excluded (infinite efficiency, always active).

    Args:
        marginal_effects: per-channel marginal effect from DGP.
        budget_config: cost configuration.

    Returns:
        Dict of paid channel_name → efficiency (effect per dollar).
    """
    cost_lookup = {cd.channel_name: cd for cd in budget_config.cost_defs}
    efficiency: Dict[str, float] = {}

    for ch_name, effect in marginal_effects.items():
        cd = cost_lookup.get(ch_name)
        if cd is None or cd.cost_type == "zero" or cd.base_cost_per_touchpoint <= 0:
            continue
        # Use base cost (no segment multiplier) for ground truth efficiency
        efficiency[ch_name] = effect / cd.base_cost_per_touchpoint

    return efficiency


def compute_optimal_allocation(
    config: DGPConfig,
    budget_config: BudgetConfig,
    journeys: pd.DataFrame,
) -> Dict:
    """Compute ground truth optimal budget allocation (Approach A: Linear).

    With linear response, the optimal allocation is proportional to efficiency
    (marginal_effect / cost) for paid channels.

    Args:
        config: DGP configuration.
        budget_config: cost and budget configuration.
        journeys: journey DataFrame for computing average decay.

    Returns:
        Dict with optimal allocation details for ground_truth.json.
    """
    marginal_effects = compute_channel_marginal_effect(config, journeys)
    efficiency = compute_channel_efficiency(marginal_effects, budget_config)

    # Normalize to allocation fractions (sum = 1.0 across paid channels)
    total_eff = sum(efficiency.values())
    if total_eff > 0:
        allocation_fractions = {ch: eff / total_eff for ch, eff in efficiency.items()}
    else:
        n_paid = len(efficiency)
        allocation_fractions = {ch: 1.0 / n_paid for ch in efficiency}

    # Dollar allocation
    allocation_dollars = {
        ch: frac * budget_config.total_budget
        for ch, frac in allocation_fractions.items()
    }

    # Efficiency ranking
    efficiency_ranking = sorted(efficiency, key=efficiency.get, reverse=True)

    return {
        "method": "linear_response_efficiency",
        "marginal_effects": marginal_effects,
        "channel_efficiency": efficiency,
        "optimal_allocation_fraction": allocation_fractions,
        "optimal_allocation_dollars": allocation_dollars,
        "efficiency_ranking": efficiency_ranking,
        "total_budget": budget_config.total_budget,
        "revenue_per_conversion": budget_config.revenue_per_conversion,
    }
