"""Rule-based attribution models (5 heuristic methods).

Each method assigns credit to touchpoints based on simple rules,
then aggregates to channel-level normalized credits.

Methods:
    1. Last Click: 100% credit to last touchpoint
    2. First Click: 100% credit to first touchpoint
    3. Linear: Equal credit to all touchpoints
    4. Time Decay: Exponential decay from last touchpoint (half-life = 7 days)
    5. Position-Based: 40% first, 40% last, 20% split among middle
"""

from typing import List

import numpy as np
import pandas as pd

from part1_simulation import AttributionResult, CHANNEL_NAMES


def _normalize_credits(credits_series: pd.Series) -> dict:
    """Normalize a channel credit Series to sum=1.0 and return as dict."""
    total = credits_series.sum()
    if total > 0:
        normalized = credits_series / total
    else:
        normalized = pd.Series(1.0 / len(CHANNEL_NAMES), index=CHANNEL_NAMES)
    # Ensure all channels present
    return {ch: float(normalized.get(ch, 0.0)) for ch in CHANNEL_NAMES}


def _get_converted_journeys(journeys: pd.DataFrame) -> pd.DataFrame:
    """Filter to converted users' touchpoints only."""
    return journeys.loc[journeys["converted"]].copy()


def compute_last_click(journeys: pd.DataFrame) -> AttributionResult:
    """Last Click: 100% credit to the last touchpoint before conversion."""
    converted = _get_converted_journeys(journeys)

    credits = (
        converted
        .loc[converted["is_last_touchpoint"]]
        .groupby("channel", observed=True)
        .size()
    )
    raw = {ch: float(credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    normalized = _normalize_credits(credits)

    return AttributionResult(
        method="Last Click",
        channel_credits=normalized,
        channel_credits_raw=raw,
        metadata={},
    )


def compute_first_click(journeys: pd.DataFrame) -> AttributionResult:
    """First Click: 100% credit to the first touchpoint in the journey."""
    converted = _get_converted_journeys(journeys)

    credits = (
        converted
        .loc[converted["touchpoint_idx"] == 0]
        .groupby("channel", observed=True)
        .size()
    )
    raw = {ch: float(credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    normalized = _normalize_credits(credits)

    return AttributionResult(
        method="First Click",
        channel_credits=normalized,
        channel_credits_raw=raw,
        metadata={},
    )


def compute_linear(journeys: pd.DataFrame) -> AttributionResult:
    """Linear: Equal credit to all touchpoints in the journey."""
    converted = _get_converted_journeys(journeys)

    credits = (
        converted
        .assign(credit=lambda df: 1.0 / df["journey_length"])
        .groupby("channel", observed=True)["credit"]
        .sum()
    )
    raw = {ch: float(credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    normalized = _normalize_credits(credits)

    return AttributionResult(
        method="Linear",
        channel_credits=normalized,
        channel_credits_raw=raw,
        metadata={"description": "Equal credit to all touchpoints"},
    )


def compute_time_decay(
    journeys: pd.DataFrame,
    half_life_days: float = 7.0,
) -> AttributionResult:
    """Time Decay: Exponential decay from the last touchpoint.

    Touchpoints closer to conversion get more credit.
    weight = 2^(-time_before_last / half_life)

    Args:
        journeys: long-format journey DataFrame.
        half_life_days: half-life in days for the decay function.
    """
    converted = _get_converted_journeys(journeys)
    half_life_hours = half_life_days * 24.0

    # Compute time before last touchpoint (per user)
    last_timestamps = (
        converted
        .groupby("user_id", sort=False)["timestamp"]
        .transform("max")
    )
    time_before_last = last_timestamps - converted["timestamp"]

    # Decay weight: 2^(-t / half_life)
    weights = np.power(2.0, -time_before_last / half_life_hours)

    # Normalize weights within each user to sum=1
    user_weight_sums = (
        pd.Series(weights, index=converted.index)
        .groupby(converted["user_id"])
        .transform("sum")
    )
    normalized_weights = weights / user_weight_sums

    credits = (
        converted
        .assign(credit=normalized_weights)
        .groupby("channel", observed=True)["credit"]
        .sum()
    )
    raw = {ch: float(credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    normalized = _normalize_credits(credits)

    return AttributionResult(
        method=f"Time Decay ({half_life_days}d)",
        channel_credits=normalized,
        channel_credits_raw=raw,
        metadata={"half_life_days": half_life_days},
    )


def compute_position_based(
    journeys: pd.DataFrame,
    first_weight: float = 0.4,
    last_weight: float = 0.4,
) -> AttributionResult:
    """Position-Based: Fixed credit to first/last, remainder split among middle.

    Default: 40% first, 40% last, 20% distributed equally among middle touchpoints.
    Single-touchpoint journeys: 100% credit.
    Two-touchpoint journeys: 50%/50% first/last.

    Args:
        journeys: long-format journey DataFrame.
        first_weight: credit fraction for first touchpoint.
        last_weight: credit fraction for last touchpoint.
    """
    converted = _get_converted_journeys(journeys)
    middle_weight = 1.0 - first_weight - last_weight

    def _assign_position_credit(row: pd.Series) -> float:
        idx = row["touchpoint_idx"]
        length = row["journey_length"]

        if length == 1:
            return 1.0
        elif length == 2:
            return 0.5  # split first/last evenly
        else:
            if idx == 0:
                return first_weight
            elif idx == length - 1:
                return last_weight
            else:
                n_middle = length - 2
                return middle_weight / n_middle if n_middle > 0 else 0.0

    credits_per_tp = converted.apply(_assign_position_credit, axis=1)

    credits = (
        converted
        .assign(credit=credits_per_tp)
        .groupby("channel", observed=True)["credit"]
        .sum()
    )
    raw = {ch: float(credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    normalized = _normalize_credits(credits)

    return AttributionResult(
        method=f"Position-Based ({first_weight:.0%}/{last_weight:.0%})",
        channel_credits=normalized,
        channel_credits_raw=raw,
        metadata={"first_weight": first_weight, "last_weight": last_weight},
    )


def run_all_rule_based(journeys: pd.DataFrame) -> List[AttributionResult]:
    """Run all 5 rule-based attribution methods.

    Returns:
        List of 5 AttributionResult objects.
    """
    return [
        compute_last_click(journeys),
        compute_first_click(journeys),
        compute_linear(journeys),
        compute_time_decay(journeys),
        compute_position_based(journeys),
    ]
