"""Config loading pipeline: Hydra Compose API → OmegaConf → NamedTuple.

All downstream modules receive NamedTuples only (no OmegaConf dependency).
Precedence: CLI overrides > YAML defaults > NamedTuple defaults.
"""

import os
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from part1_simulation import (
    BudgetConfig,
    ChannelDef,
    CostDef,
    CrossInfluence,
    DGPConfig,
    SegmentDef,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_DIR = str(_PROJECT_ROOT / "configs" / "dgp")


def _to_channel_def(raw: dict) -> ChannelDef:
    return ChannelDef(
        name=raw["name"],
        beta=float(raw["beta"]),
        decay_half_life_days=float(raw["decay_half_life_days"]),
        funnel_position=raw["funnel_position"],
    )


def _to_segment_def(raw: dict) -> SegmentDef:
    return SegmentDef(
        name=raw["name"],
        proportion=float(raw["proportion"]),
        geometric_p=float(raw["geometric_p"]),
        geometric_offset=int(raw["geometric_offset"]),
        eta=float(raw["eta"]),
        start_channels=tuple(raw["start_channels"]),
    )


def _to_cross_influence(raw: dict) -> CrossInfluence:
    return CrossInfluence(
        source=raw["source"],
        target=raw["target"],
        delta=float(raw["delta"]),
    )


def load_dgp_config(
    config_dir: Optional[str] = None,
    config_name: str = "default",
    overrides: Optional[List[str]] = None,
) -> DGPConfig:
    """Load DGP config from YAML via Hydra Compose API.

    Args:
        config_dir: Absolute path to config directory. Defaults to configs/dgp/.
        config_name: YAML filename (without .yaml extension).
        overrides: Hydra-style overrides, e.g. ["n_users=10000", "alpha_0=-4.5"].

    Returns:
        Immutable DGPConfig NamedTuple.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    overrides = overrides or []

    # Clear any existing Hydra state (allows multiple calls in same process)
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides)

    raw = OmegaConf.to_container(cfg, resolve=True)

    channels = tuple(map(_to_channel_def, raw["channels"]))
    segments = tuple(map(_to_segment_def, raw["segments"]))
    cross_influences = tuple(map(_to_cross_influence, raw["cross_influences"]))

    return DGPConfig(
        n_users=int(raw["n_users"]),
        n_channels=int(raw["n_channels"]),
        target_conversion_rate=float(raw["target_conversion_rate"]),
        alpha_0=float(raw["alpha_0"]),
        inter_arrival_lambda_hours=float(raw["inter_arrival_lambda_hours"]),
        max_touchpoints=int(raw["max_touchpoints"]),
        random_seed=int(raw["random_seed"]),
        channels=channels,
        segments=segments,
        cross_influences=cross_influences,
    )


# ============================================================
# Budget Config Loading (separate from DGPConfig)
# ============================================================

def _to_cost_def(raw: dict) -> CostDef:
    return CostDef(
        channel_name=raw["channel_name"],
        cost_type=raw["cost_type"],
        base_cost_per_touchpoint=float(raw["base_cost_per_touchpoint"]),
        segment_multipliers={k: float(v) for k, v in raw["segment_multipliers"].items()},
    )


def _to_budget_config(raw: dict) -> BudgetConfig:
    cost_defs = tuple(map(_to_cost_def, raw["cost_defs"]))
    return BudgetConfig(
        total_budget=float(raw.get("total_budget", 200_000.0)),
        revenue_per_conversion=float(raw.get("revenue_per_conversion", 100.0)),
        cost_noise_sigma=float(raw.get("cost_noise_sigma", 0.1)),
        cost_defs=cost_defs,
    )


def load_budget_config(
    config_dir: Optional[str] = None,
    config_name: str = "default",
) -> Optional[BudgetConfig]:
    """Load BudgetConfig from YAML if budget_config key exists.

    Separate from load_dgp_config() to preserve backward compatibility.
    Returns None if the YAML has no budget_config section.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR

    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg: DictConfig = compose(config_name=config_name)

    raw = OmegaConf.to_container(cfg, resolve=True)

    if "budget_config" not in raw or raw["budget_config"] is None:
        return None

    return _to_budget_config(raw["budget_config"])
