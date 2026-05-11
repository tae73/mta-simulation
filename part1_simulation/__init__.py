"""Part 1: Simulation-based MTA — Shared type definitions.

Every module in the project depends on these NamedTuples.
DGP modules produce data conforming to JOURNEY_SCHEMA.
Model modules consume journeys and return AttributionResult.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple


# ============================================================
# Channel Definitions
# ============================================================

class ChannelDef(NamedTuple):
    """Single marketing channel definition with DGP parameters."""
    name: str
    beta: float                  # conversion effect coefficient
    decay_half_life_days: float  # temporal decay: exp(-Δt / (half_life * 24h))
    funnel_position: str         # "upper", "mid", "lower"


# ============================================================
# User Segments
# ============================================================

class SegmentDef(NamedTuple):
    """User segment definition controlling journey generation."""
    name: str
    proportion: float            # fraction of total users (sum across segments = 1.0)
    geometric_p: float           # Geometric distribution param for journey length
    geometric_offset: int        # minimum journey length (added to Geometric draw)
    eta: float                   # user heterogeneity effect on conversion intensity
    start_channels: Tuple[str, ...]  # preferred starting channels


# ============================================================
# Cross-Channel Influence
# ============================================================

class CrossInfluence(NamedTuple):
    """Directional synergy between two channels (CDA 2025 framework)."""
    source: str                  # channel that must appear first
    target: str                  # channel that benefits from source
    delta: float                 # synergy magnitude added to log-intensity


# ============================================================
# Cost & Budget (observation layer — does NOT affect conversions)
# ============================================================

class CostDef(NamedTuple):
    """Per-channel cost parameters for budget optimization layer."""
    channel_name: str
    cost_type: str                       # "cpm", "cpc", "fixed", "zero"
    base_cost_per_touchpoint: float      # dollars per touchpoint
    segment_multipliers: Dict[str, float]  # segment_name → cost multiplier

class BudgetConfig(NamedTuple):
    """Budget optimization configuration (separate from DGPConfig)."""
    total_budget: float = 200_000.0       # total available marketing budget ($)
    revenue_per_conversion: float = 100.0  # average revenue per conversion
    cost_noise_sigma: float = 0.1          # log-normal noise std on per-TP cost
    cost_defs: Tuple[CostDef, ...] = ()


# ============================================================
# DGP Configuration
# ============================================================

class DGPConfig(NamedTuple):
    """Complete DGP configuration (Hydra YAML → this NamedTuple)."""
    n_users: int = 100_000
    n_channels: int = 7
    target_conversion_rate: float = 0.025
    alpha_0: float = -5.0        # baseline log-intensity (calibrated via binary search)
    inter_arrival_lambda_hours: float = 48.0  # mean hours between touchpoints
    max_touchpoints: int = 20
    random_seed: int = 42
    channels: Tuple[ChannelDef, ...] = ()
    segments: Tuple[SegmentDef, ...] = ()
    cross_influences: Tuple[CrossInfluence, ...] = ()


# ============================================================
# Attribution Result (every model returns this)
# ============================================================

class AttributionResult(NamedTuple):
    """Standardized output from any attribution method.

    channel_credits: normalized credits (sum = 1.0)
    channel_credits_raw: before normalization
    metadata: method-specific extras (e.g., transition matrix, attention weights)
    """
    method: str
    channel_credits: Dict[str, float]
    channel_credits_raw: Dict[str, float]
    metadata: Dict[str, object]


# ============================================================
# Evaluation Result
# ============================================================

class EvaluationResult(NamedTuple):
    """Metrics comparing one attribution method against ground truth."""
    method: str
    mae: float
    kendall_tau: float
    rmse: float
    channel_bias: Dict[str, float]  # per-channel: predicted - truth


# ============================================================
# Journey DataFrame Schema
# ============================================================

CHANNEL_NAMES: Tuple[str, ...] = (
    "Display",
    "Social",
    "Organic Search",
    "Paid Search",
    "Email",
    "Referral",
    "Direct",
)

SEGMENT_NAMES: Tuple[str, ...] = ("New", "Exploratory", "Loyal")

# Long-format: one row per touchpoint per user
JOURNEY_SCHEMA = {
    "user_id": "int64",
    "segment": "category",           # New / Exploratory / Loyal
    "touchpoint_idx": "int64",       # 0-indexed position in journey
    "channel": "category",           # one of CHANNEL_NAMES
    "timestamp": "float64",          # hours from user's journey start
    "is_last_touchpoint": "bool",
    "converted": "bool",             # same for all rows of a user
    "journey_length": "int64",       # total touchpoints for this user
    "conversion_intensity": "float64",  # λ_i(t) at observation time
    "touchpoint_cost": "float64",        # per-touchpoint cost in dollars
}
