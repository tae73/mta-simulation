"""Cost assignment layer for the DGP pipeline.

Observation-only layer: attaches per-touchpoint costs AFTER conversion decisions.
Does NOT affect conversion probabilities or any DGP mechanics.

Cost formula per touchpoint:
    actual_cost = base_cost × segment_multiplier × exp(ε),  ε ~ N(0, σ²)
    (zero-cost channels always produce 0.0, no noise applied)
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from part1_simulation import BudgetConfig, CostDef


def _build_cost_lookup(
    budget_config: BudgetConfig,
) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """Build (channel, segment) → (effective_base_cost, cost_type) lookup.

    Returns:
        Dict mapping (channel_name, segment_name) to (base × multiplier, cost_type).
    """
    lookup: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for cd in budget_config.cost_defs:
        for seg_name, mult in cd.segment_multipliers.items():
            lookup[(cd.channel_name, seg_name)] = (
                cd.base_cost_per_touchpoint * mult,
                cd.cost_type,
            )
    return lookup


def assign_touchpoint_costs(
    journeys: pd.DataFrame,
    budget_config: BudgetConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign per-touchpoint costs to journey DataFrame.

    For each row, looks up (channel, segment) to get effective base cost,
    then applies log-normal noise for non-zero channels.

    Args:
        journeys: long-format journey DataFrame (must have 'channel', 'segment').
        budget_config: cost configuration.
        rng: numpy random Generator for reproducibility.

    Returns:
        DataFrame with 'touchpoint_cost' column added.
    """
    lookup = _build_cost_lookup(budget_config)
    n_rows = len(journeys)

    # Vectorized: map (channel, segment) pairs to base costs and cost types
    channels = journeys["channel"].astype(str).values
    segments = journeys["segment"].astype(str).values

    base_costs = np.zeros(n_rows, dtype=np.float64)
    is_paid = np.zeros(n_rows, dtype=bool)

    for i in range(n_rows):
        key = (channels[i], segments[i])
        if key in lookup:
            cost, cost_type = lookup[key]
            base_costs[i] = cost
            is_paid[i] = cost_type != "zero"

    # Apply log-normal noise only to paid channels
    noise = np.ones(n_rows, dtype=np.float64)
    n_paid = is_paid.sum()
    if n_paid > 0 and budget_config.cost_noise_sigma > 0:
        noise[is_paid] = np.exp(
            rng.normal(0.0, budget_config.cost_noise_sigma, size=n_paid)
        )

    costs = base_costs * noise
    # Ensure zero-cost channels remain exactly 0.0
    costs[~is_paid] = 0.0

    return journeys.assign(touchpoint_cost=costs)


def compute_cost_summary(journeys: pd.DataFrame) -> Dict:
    """Compute aggregate cost statistics from journeys with touchpoint_cost.

    Args:
        journeys: DataFrame with 'touchpoint_cost', 'channel', 'converted' columns.

    Returns:
        Dict with channel-level and overall cost statistics.
    """
    channel_stats = (
        journeys
        .groupby("channel", observed=True)
        .agg(
            total_cost=("touchpoint_cost", "sum"),
            touchpoint_count=("touchpoint_cost", "count"),
            avg_cost_per_touchpoint=("touchpoint_cost", "mean"),
        )
        .to_dict("index")
    )

    total_spend = journeys["touchpoint_cost"].sum()

    # Cost per conversion (total spend / number of converters)
    n_converters = journeys.loc[journeys["converted"]]["user_id"].nunique()
    cost_per_conversion = total_spend / n_converters if n_converters > 0 else 0.0

    # Channel total costs as flat dict
    channel_total_cost = {
        ch: stats["total_cost"] for ch, stats in channel_stats.items()
    }
    channel_avg_cost = {
        ch: stats["avg_cost_per_touchpoint"] for ch, stats in channel_stats.items()
    }

    return {
        "channel_total_cost": channel_total_cost,
        "channel_avg_cost_per_touchpoint": channel_avg_cost,
        "total_spend": float(total_spend),
        "cost_per_conversion": float(cost_per_conversion),
        "n_converters": int(n_converters),
    }
