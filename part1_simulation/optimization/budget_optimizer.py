"""Budget optimization from attribution results (Approach A: Linear Response).

Given attribution credits and per-channel costs, computes ROI-based budget
allocation. This represents what a practitioner would do with attribution output.

Linear response: allocate budget proportional to estimated efficiency
(channel_credit / cost_per_touchpoint) across paid channels.
"""

from typing import Dict

from part1_simulation import AttributionResult, BudgetConfig


def compute_channel_costs(budget_config: BudgetConfig) -> Dict[str, float]:
    """Extract base cost per touchpoint for each channel.

    Returns:
        Dict of channel_name → base_cost_per_touchpoint.
    """
    return {
        cd.channel_name: cd.base_cost_per_touchpoint
        for cd in budget_config.cost_defs
    }


def optimize_budget(
    attribution: AttributionResult,
    budget_config: BudgetConfig,
    total_conversions: int,
) -> Dict:
    """Derive budget allocation from attribution credits (Linear Response).

    For each paid channel:
        estimated_efficiency_k = channel_credit_k / cost_per_touchpoint_k

    Allocation: B_k = B × (efficiency_k / Σ efficiency_j)

    Args:
        attribution: result from any attribution method (normalized credits).
        budget_config: cost and budget configuration.
        total_conversions: observed number of conversions.

    Returns:
        Dict with allocation fractions, dollars, ROAS, and CPA per channel.
    """
    channel_costs = compute_channel_costs(budget_config)
    credits = attribution.channel_credits
    revenue = budget_config.revenue_per_conversion

    # Compute estimated efficiency for paid channels only
    efficiency: Dict[str, float] = {}
    for ch_name, credit in credits.items():
        cost = channel_costs.get(ch_name, 0.0)
        if cost <= 0.0:
            continue  # skip zero-cost channels
        efficiency[ch_name] = credit / cost

    # Normalize to allocation fractions
    total_eff = sum(efficiency.values())
    if total_eff > 0:
        allocation_fractions = {ch: eff / total_eff for ch, eff in efficiency.items()}
    else:
        n_paid = len(efficiency) or 1
        allocation_fractions = {ch: 1.0 / n_paid for ch in efficiency}

    allocation_dollars = {
        ch: frac * budget_config.total_budget
        for ch, frac in allocation_fractions.items()
    }

    # ROAS and CPA per paid channel
    channel_roas: Dict[str, float] = {}
    channel_cpa: Dict[str, float] = {}
    for ch_name in efficiency:
        credit = credits.get(ch_name, 0.0)
        attributed_conversions = credit * total_conversions
        attributed_revenue = attributed_conversions * revenue
        actual_cost = channel_costs[ch_name]  # per-TP cost as proxy

        # ROAS = attributed revenue / cost (using allocated budget as cost basis)
        alloc = allocation_dollars.get(ch_name, 0.0)
        channel_roas[ch_name] = attributed_revenue / alloc if alloc > 0 else 0.0
        channel_cpa[ch_name] = alloc / attributed_conversions if attributed_conversions > 0 else 0.0

    return {
        "method": attribution.method,
        "allocation_fraction": allocation_fractions,
        "allocation_dollars": allocation_dollars,
        "channel_roas": channel_roas,
        "channel_cpa": channel_cpa,
        "efficiency_ranking": sorted(efficiency, key=efficiency.get, reverse=True),
    }
