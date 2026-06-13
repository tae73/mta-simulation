"""Shared test helpers — toy journey builders + faithful tiny configs.

NOT a test module (no ``test_`` prefix → pytest won't collect it). New test
files import from here to avoid duplicating the journey-construction and
config-construction boilerplate.

Conventions mirror the existing ``test_survival_attribution.py`` helpers:
  - ``journey_rows`` / ``make_journeys`` build long-format DataFrames matching
    ``JOURNEY_SCHEMA`` (one row per touchpoint per user).
  - ``default_dgp_config`` / ``default_budget_config`` reproduce the canonical
    ``configs/dgp/default.yaml`` values in-code so tests are fast, deterministic,
    and independent of YAML/Hydra loading.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd

from part1_simulation import (
    BudgetConfig,
    ChannelDef,
    CostDef,
    CrossInfluence,
    DGPConfig,
    SegmentDef,
)

# Spec tuple: (user_id, segment, channels, timestamps, converted)
JourneySpec = Tuple[int, str, List[str], List[float], bool]


# ============================================================
# Toy journey builders (mirror test_survival_attribution helpers)
# ============================================================

def journey_rows(
    user_id: int,
    segment: str,
    channels: Sequence[str],
    timestamps: Sequence[float],
    converted: bool,
) -> List[dict]:
    """Build the per-touchpoint row dicts for a single user's journey.

    Fills every ``JOURNEY_SCHEMA`` column so the result is a drop-in input for
    any DGP/model/evaluation function.
    """
    n = len(channels)
    return [
        {
            "user_id": user_id,
            "segment": segment,
            "touchpoint_idx": i,
            "channel": ch,
            "timestamp": float(ts),
            "is_last_touchpoint": (i == n - 1),
            "converted": converted,
            "journey_length": n,
            "conversion_intensity": 0.0,
            "touchpoint_cost": 0.0,
        }
        for i, (ch, ts) in enumerate(zip(channels, timestamps))
    ]


def make_journeys(specs: Sequence[JourneySpec]) -> pd.DataFrame:
    """Convert a list of ``(uid, segment, channels, timestamps, converted)`` to a DataFrame."""
    rows: List[dict] = []
    for spec in specs:
        rows.extend(journey_rows(*spec))
    return pd.DataFrame(rows)


# ============================================================
# Canonical configs (faithful to configs/dgp/default.yaml)
# ============================================================

# (name, beta, decay_half_life_days, funnel_position)
_CHANNELS: Tuple[Tuple[str, float, float, str], ...] = (
    ("Display", 0.3, 14.0, "upper"),
    ("Social", 0.4, 3.0, "upper"),
    ("Organic Search", 0.5, 7.0, "mid"),
    ("Paid Search", 1.2, 1.0, "lower"),
    ("Email", 0.8, 5.0, "mid"),
    ("Referral", 0.5, 7.0, "mid"),
    ("Direct", 0.7, 2.0, "lower"),
)

# (name, proportion, geometric_p, geometric_offset, eta, start_channels)
_SEGMENTS: Tuple[Tuple[str, float, float, int, float, Tuple[str, ...]], ...] = (
    ("New", 0.5, 0.25, 1, -0.3, ("Display", "Social")),
    ("Exploratory", 0.3, 0.2, 2, 0.0, ("Organic Search",)),
    ("Loyal", 0.2, 0.5, 1, 0.5, ("Email", "Direct")),
)

# (source, target, delta)
_CROSS: Tuple[Tuple[str, str, float], ...] = (
    ("Display", "Paid Search", 0.4),
    ("Social", "Email", 0.3),
    ("Organic Search", "Direct", 0.2),
)

# (channel_name, cost_type, base_cost_per_touchpoint, segment_multipliers)
_COSTS: Tuple[Tuple[str, str, float, dict], ...] = (
    ("Display", "cpm", 0.005, {"New": 1.2, "Exploratory": 1.0, "Loyal": 0.8}),
    ("Social", "cpm", 0.008, {"New": 1.3, "Exploratory": 1.0, "Loyal": 0.7}),
    ("Organic Search", "zero", 0.0, {"New": 1.0, "Exploratory": 1.0, "Loyal": 1.0}),
    ("Paid Search", "cpc", 2.50, {"New": 1.1, "Exploratory": 1.0, "Loyal": 0.9}),
    ("Email", "fixed", 0.003, {"New": 1.0, "Exploratory": 1.0, "Loyal": 1.0}),
    ("Referral", "zero", 0.0, {"New": 1.0, "Exploratory": 1.0, "Loyal": 1.0}),
    ("Direct", "zero", 0.0, {"New": 1.0, "Exploratory": 1.0, "Loyal": 1.0}),
)


def default_channels() -> Tuple[ChannelDef, ...]:
    return tuple(ChannelDef(*c) for c in _CHANNELS)


def default_segments() -> Tuple[SegmentDef, ...]:
    return tuple(SegmentDef(*s) for s in _SEGMENTS)


def default_cross_influences() -> Tuple[CrossInfluence, ...]:
    return tuple(CrossInfluence(*c) for c in _CROSS)


def default_dgp_config(
    n_users: int = 2_000,
    seed: int = 42,
    alpha_0: float = -5.0,
    max_touchpoints: int = 20,
) -> DGPConfig:
    """A faithful (small-by-default) copy of configs/dgp/default.yaml.

    ``alpha_0`` defaults to the YAML seed value (-5.0); raise it in tests that
    need a non-trivial conversion count from a small sample.
    """
    return DGPConfig(
        n_users=n_users,
        n_channels=7,
        target_conversion_rate=0.025,
        alpha_0=alpha_0,
        inter_arrival_lambda_hours=48.0,
        max_touchpoints=max_touchpoints,
        random_seed=seed,
        channels=default_channels(),
        segments=default_segments(),
        cross_influences=default_cross_influences(),
    )


def default_budget_config(
    total_budget: float = 200_000.0,
    revenue_per_conversion: float = 100.0,
    cost_noise_sigma: float = 0.1,
) -> BudgetConfig:
    """A faithful copy of the budget_config section of configs/dgp/default.yaml."""
    return BudgetConfig(
        total_budget=total_budget,
        revenue_per_conversion=revenue_per_conversion,
        cost_noise_sigma=cost_noise_sigma,
        cost_defs=tuple(CostDef(*c) for c in _COSTS),
    )


def segment_by_name(name: str) -> SegmentDef:
    """Look up a canonical SegmentDef by name (e.g. for compute_log_intensity)."""
    for seg in default_segments():
        if seg.name == name:
            return seg
    raise KeyError(name)
