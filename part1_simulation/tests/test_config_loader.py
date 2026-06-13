"""Unit tests for config_loader.py (Hydra Compose API → OmegaConf → NamedTuple).

Verifies that load_dgp_config / load_budget_config faithfully materialize the
canonical configs/dgp/default.yaml into the immutable DGPConfig / BudgetConfig
NamedTuples. Path resolution is exercised both via the explicit ``config_dir``
argument and via the module's default (_DEFAULT_CONFIG_DIR).

Mapping of assertions to YAML facts:
    - 7 channels / 3 segments / 3 cross_influences (structure)
    - Paid Search beta==1.2, decay_half_life_days==1.0  (channel parse)
    - New segment eta==-0.3                              (segment parse)
    - Display->Paid Search delta==0.4                    (cross-influence parse)
    - 7 cost_defs, total_budget==200000.0, rev/conv==100.0, Organic zero
    - Returned objects are DGPConfig / BudgetConfig NamedTuple types
    - Hydra overrides take precedence over YAML defaults
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from part1_simulation import (
    BudgetConfig,
    ChannelDef,
    CostDef,
    CrossInfluence,
    DGPConfig,
    SegmentDef,
)
from part1_simulation.config_loader import (
    _DEFAULT_CONFIG_DIR,
    load_budget_config,
    load_dgp_config,
)

# Repo root computed independently of the module under test, so the explicit
# config_dir path is verified rather than trivially echoed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = str(_REPO_ROOT / "configs" / "dgp")


# ============================================================
# Path resolution
# ============================================================

def test_default_config_dir_points_at_repo_configs():
    """Module default _DEFAULT_CONFIG_DIR resolves to <repo>/configs/dgp."""
    assert Path(_DEFAULT_CONFIG_DIR) == _REPO_ROOT / "configs" / "dgp"
    assert (Path(_DEFAULT_CONFIG_DIR) / "default.yaml").is_file()


def test_explicit_config_dir_exists():
    """The independently-computed config dir holds the default.yaml under test."""
    assert (Path(_CONFIG_DIR) / "default.yaml").is_file()


# ============================================================
# load_dgp_config — type + structure
# ============================================================

def test_load_dgp_config_returns_dgpconfig_type():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    assert isinstance(cfg, DGPConfig)
    # NamedTuple, not a bare tuple/dict
    assert hasattr(cfg, "_fields")


def test_load_dgp_config_default_dir_matches_explicit_dir():
    """Calling with no config_dir uses _DEFAULT_CONFIG_DIR == explicit path."""
    cfg_default = load_dgp_config()
    cfg_explicit = load_dgp_config(config_dir=_CONFIG_DIR)
    assert cfg_default == cfg_explicit


def test_load_dgp_config_structure_counts():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    assert cfg.n_channels == 7
    assert len(cfg.channels) == 7
    assert len(cfg.segments) == 3
    assert len(cfg.cross_influences) == 3


def test_load_dgp_config_scalar_fields():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    assert cfg.n_users == 100_000
    np.testing.assert_allclose(cfg.target_conversion_rate, 0.025, atol=1e-9)
    np.testing.assert_allclose(cfg.alpha_0, -5.0, atol=1e-9)
    np.testing.assert_allclose(cfg.inter_arrival_lambda_hours, 48.0, atol=1e-9)
    assert cfg.max_touchpoints == 20
    assert cfg.random_seed == 42


# ============================================================
# load_dgp_config — element types + parsed values
# ============================================================

def test_load_dgp_config_element_types():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    assert all(isinstance(c, ChannelDef) for c in cfg.channels)
    assert all(isinstance(s, SegmentDef) for s in cfg.segments)
    assert all(isinstance(x, CrossInfluence) for x in cfg.cross_influences)


def test_load_dgp_config_paid_search_channel():
    """Paid Search: beta==1.2, decay_half_life_days==1.0 (fast-decay lower funnel)."""
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    paid = next(c for c in cfg.channels if c.name == "Paid Search")
    np.testing.assert_allclose(paid.beta, 1.2, atol=1e-9)
    np.testing.assert_allclose(paid.decay_half_life_days, 1.0, atol=1e-9)
    assert paid.funnel_position == "lower"


def test_load_dgp_config_channel_names_in_order():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    names = tuple(c.name for c in cfg.channels)
    assert names == (
        "Display",
        "Social",
        "Organic Search",
        "Paid Search",
        "Email",
        "Referral",
        "Direct",
    )


def test_load_dgp_config_new_segment_eta():
    """New segment heterogeneity eta == -0.3."""
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    new = next(s for s in cfg.segments if s.name == "New")
    np.testing.assert_allclose(new.eta, -0.3, atol=1e-9)
    np.testing.assert_allclose(new.proportion, 0.5, atol=1e-9)
    assert new.geometric_offset == 1
    assert new.start_channels == ("Display", "Social")


def test_load_dgp_config_segment_proportions_sum_to_one():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    total = sum(s.proportion for s in cfg.segments)
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_load_dgp_config_cross_influence_display_to_paid_search():
    """Display -> Paid Search synergy delta == 0.4."""
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    ci = next(
        x for x in cfg.cross_influences
        if x.source == "Display" and x.target == "Paid Search"
    )
    np.testing.assert_allclose(ci.delta, 0.4, atol=1e-9)


def test_load_dgp_config_all_cross_influences():
    cfg = load_dgp_config(config_dir=_CONFIG_DIR)
    triples = {(x.source, x.target, round(x.delta, 6)) for x in cfg.cross_influences}
    assert triples == {
        ("Display", "Paid Search", 0.4),
        ("Social", "Email", 0.3),
        ("Organic Search", "Direct", 0.2),
    }


# ============================================================
# load_dgp_config — Hydra overrides (precedence)
# ============================================================

def test_load_dgp_config_override_precedence():
    """CLI-style overrides win over YAML defaults; untouched fields unchanged."""
    cfg = load_dgp_config(
        config_dir=_CONFIG_DIR,
        overrides=["n_users=10000", "alpha_0=-4.5"],
    )
    assert cfg.n_users == 10_000
    np.testing.assert_allclose(cfg.alpha_0, -4.5, atol=1e-9)
    # Non-overridden fields keep YAML values.
    assert cfg.n_channels == 7
    assert cfg.random_seed == 42


def test_load_dgp_config_repeatable_in_process():
    """GlobalHydra is cleared each call → multiple loads in one process agree."""
    a = load_dgp_config(config_dir=_CONFIG_DIR)
    b = load_dgp_config(config_dir=_CONFIG_DIR)
    assert a == b


# ============================================================
# load_budget_config — type + structure + values
# ============================================================

def test_load_budget_config_returns_budgetconfig_type():
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    assert isinstance(bcfg, BudgetConfig)
    assert hasattr(bcfg, "_fields")


def test_load_budget_config_default_dir_matches_explicit_dir():
    bcfg_default = load_budget_config()
    bcfg_explicit = load_budget_config(config_dir=_CONFIG_DIR)
    assert bcfg_default == bcfg_explicit


def test_load_budget_config_scalars():
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    np.testing.assert_allclose(bcfg.total_budget, 200_000.0, atol=1e-9)
    np.testing.assert_allclose(bcfg.revenue_per_conversion, 100.0, atol=1e-9)
    np.testing.assert_allclose(bcfg.cost_noise_sigma, 0.1, atol=1e-9)


def test_load_budget_config_cost_defs_count_and_type():
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    assert len(bcfg.cost_defs) == 7
    assert all(isinstance(c, CostDef) for c in bcfg.cost_defs)


def test_load_budget_config_organic_search_zero_cost():
    """Organic Search is a zero-cost channel (cost_type=='zero', base==0)."""
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    organic = next(c for c in bcfg.cost_defs if c.channel_name == "Organic Search")
    assert organic.cost_type == "zero"
    np.testing.assert_allclose(organic.base_cost_per_touchpoint, 0.0, atol=1e-9)


def test_load_budget_config_zero_cost_channels_set():
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    zero_cost = {c.channel_name for c in bcfg.cost_defs if c.cost_type == "zero"}
    assert zero_cost == {"Organic Search", "Referral", "Direct"}


def test_load_budget_config_paid_search_cpc():
    """Paid Search is CPC-priced at 2.50 per touchpoint."""
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    paid = next(c for c in bcfg.cost_defs if c.channel_name == "Paid Search")
    assert paid.cost_type == "cpc"
    np.testing.assert_allclose(paid.base_cost_per_touchpoint, 2.50, atol=1e-9)


def test_load_budget_config_segment_multipliers_parsed_as_floats():
    """segment_multipliers values are coerced to float for Display."""
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    display = next(c for c in bcfg.cost_defs if c.channel_name == "Display")
    assert display.cost_type == "cpm"
    np.testing.assert_allclose(display.segment_multipliers["New"], 1.2, atol=1e-9)
    np.testing.assert_allclose(display.segment_multipliers["Exploratory"], 1.0, atol=1e-9)
    np.testing.assert_allclose(display.segment_multipliers["Loyal"], 0.8, atol=1e-9)
    assert all(isinstance(v, float) for v in display.segment_multipliers.values())


def test_load_budget_config_cost_def_channel_order_matches_dgp():
    """Budget cost_defs cover the same 7 channels as the DGP config, same order."""
    bcfg = load_budget_config(config_dir=_CONFIG_DIR)
    dgp = load_dgp_config(config_dir=_CONFIG_DIR)
    cost_channels = tuple(c.channel_name for c in bcfg.cost_defs)
    dgp_channels = tuple(c.name for c in dgp.channels)
    assert cost_channels == dgp_channels
