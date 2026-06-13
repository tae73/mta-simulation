"""Unit tests for optimization/budget_optimizer.py (Approach A: Linear Response).

Covers the two public functions:
    - compute_channel_costs(budget_config) → {channel: base_cost_per_touchpoint}
    - optimize_budget(attribution, budget_config, total_conversions) → dict

Hand-checked numerical invariants follow the implementation exactly:
    efficiency_k   = credit_k / cost_per_touchpoint_k          (paid channels only)
    fraction_k     = efficiency_k / Σ efficiency_j
    dollars_k      = fraction_k × total_budget
    ROAS_k         = (credit_k × conv × revenue) / dollars_k
    CPA_k          = dollars_k / (credit_k × conv)
Zero-cost channels (Organic Search, Referral, Direct) are excluded from allocation.
"""
from __future__ import annotations

import numpy as np

from part1_simulation import AttributionResult
from part1_simulation.optimization.budget_optimizer import (
    compute_channel_costs,
    optimize_budget,
)
from part1_simulation.tests._journey_factory import default_budget_config


# ============================================================
# Helpers
# ============================================================

# Zero-cost channels per configs/dgp/default.yaml budget_config.
_ZERO_COST = ("Organic Search", "Referral", "Direct")
# Paid channels with base_cost_per_touchpoint.
_PAID_COSTS = {
    "Display": 0.005,
    "Social": 0.008,
    "Paid Search": 2.50,
    "Email": 0.003,
}


def _make_attribution(credits: dict, method: str = "TestMethod") -> AttributionResult:
    """Wrap a normalized credit dict into an AttributionResult."""
    return AttributionResult(
        method=method,
        channel_credits=dict(credits),
        channel_credits_raw=dict(credits),
        metadata={},
    )


def _balanced_credits() -> dict:
    """All 7 channels, credits summing to 1.0 (paid channels carry real weight)."""
    credits = {
        "Display": 0.20,
        "Social": 0.10,
        "Paid Search": 0.10,
        "Email": 0.40,
        "Organic Search": 0.10,
        "Referral": 0.05,
        "Direct": 0.05,
    }
    np.testing.assert_allclose(sum(credits.values()), 1.0, atol=1e-9)
    return credits


# ============================================================
# 1. compute_channel_costs — direct extraction from cost_defs
# ============================================================

def test_compute_channel_costs_matches_cost_defs():
    """Returns base_cost_per_touchpoint per channel, all 7 present."""
    bc = default_budget_config()
    costs = compute_channel_costs(bc)

    expected = {
        "Display": 0.005,
        "Social": 0.008,
        "Organic Search": 0.0,
        "Paid Search": 2.50,
        "Email": 0.003,
        "Referral": 0.0,
        "Direct": 0.0,
    }
    assert set(costs.keys()) == set(expected.keys())
    for ch, val in expected.items():
        np.testing.assert_allclose(costs[ch], val, atol=1e-9)


def test_compute_channel_costs_round_trips_cost_defs():
    """Every CostDef.base_cost_per_touchpoint is reproduced exactly."""
    bc = default_budget_config()
    costs = compute_channel_costs(bc)
    for cd in bc.cost_defs:
        np.testing.assert_allclose(
            costs[cd.channel_name], cd.base_cost_per_touchpoint, atol=1e-9
        )


# ============================================================
# 2. optimize_budget — output structure & method passthrough
# ============================================================

def test_optimize_budget_keys_and_method():
    """Returned dict exposes the documented keys and passes through method name."""
    bc = default_budget_config()
    attr = _make_attribution(_balanced_credits(), method="Linear")
    out = optimize_budget(attr, bc, total_conversions=1000)

    assert set(out.keys()) == {
        "method",
        "allocation_fraction",
        "allocation_dollars",
        "channel_roas",
        "channel_cpa",
        "efficiency_ranking",
    }
    assert out["method"] == "Linear"


# ============================================================
# 3. Zero-cost channels excluded from allocation
# ============================================================

def test_zero_cost_channels_excluded():
    """Organic Search / Referral / Direct never appear in any allocation map."""
    bc = default_budget_config()
    attr = _make_attribution(_balanced_credits())
    out = optimize_budget(attr, bc, total_conversions=1000)

    for ch in _ZERO_COST:
        assert ch not in out["allocation_fraction"]
        assert ch not in out["allocation_dollars"]
        assert ch not in out["channel_roas"]
        assert ch not in out["channel_cpa"]
        assert ch not in out["efficiency_ranking"]

    # Exactly the 4 paid channels survive.
    assert set(out["allocation_fraction"].keys()) == set(_PAID_COSTS.keys())


# ============================================================
# 4. Efficiency = credit / cost (hand-checked single channel)
# ============================================================

def test_efficiency_equals_credit_over_cost():
    """allocation_fraction reflects efficiency = credit/cost normalized.

    Hand-check Display: credit 0.20 / cost 0.005 = 40.0.
    """
    bc = default_budget_config()
    credits = _balanced_credits()
    attr = _make_attribution(credits)
    out = optimize_budget(attr, bc, total_conversions=1000)

    eff = {ch: credits[ch] / _PAID_COSTS[ch] for ch in _PAID_COSTS}
    # Display hand value.
    np.testing.assert_allclose(eff["Display"], 40.0, atol=1e-9)
    total_eff = sum(eff.values())
    expected_fraction = {ch: e / total_eff for ch, e in eff.items()}

    for ch in _PAID_COSTS:
        np.testing.assert_allclose(
            out["allocation_fraction"][ch], expected_fraction[ch], rtol=1e-9
        )


# ============================================================
# 5. allocation_fraction sums to 1.0
# ============================================================

def test_allocation_fraction_sums_to_one():
    bc = default_budget_config()
    attr = _make_attribution(_balanced_credits())
    out = optimize_budget(attr, bc, total_conversions=1000)
    np.testing.assert_allclose(
        sum(out["allocation_fraction"].values()), 1.0, atol=1e-9
    )


# ============================================================
# 6. allocation_dollars sums to total_budget
# ============================================================

def test_allocation_dollars_sum_to_total_budget():
    bc = default_budget_config(total_budget=200_000.0)
    attr = _make_attribution(_balanced_credits())
    out = optimize_budget(attr, bc, total_conversions=1000)
    np.testing.assert_allclose(
        sum(out["allocation_dollars"].values()), 200_000.0, atol=1e-9
    )
    # And each dollar = fraction × budget.
    for ch, frac in out["allocation_fraction"].items():
        np.testing.assert_allclose(
            out["allocation_dollars"][ch], frac * 200_000.0, rtol=1e-9
        )


# ============================================================
# 7. ROAS and CPA — hand-checked for one paid channel
# ============================================================

def test_roas_and_cpa_hand_check_email():
    """ROAS = credit·conv·revenue / dollars ; CPA = dollars / (credit·conv).

    Hand-check Email (credit 0.40, conv 1000, revenue 100).
    """
    bc = default_budget_config(total_budget=200_000.0, revenue_per_conversion=100.0)
    credits = _balanced_credits()
    conv = 1000
    attr = _make_attribution(credits)
    out = optimize_budget(attr, bc, total_conversions=conv)

    eff = {ch: credits[ch] / _PAID_COSTS[ch] for ch in _PAID_COSTS}
    total_eff = sum(eff.values())
    dollars_email = (eff["Email"] / total_eff) * 200_000.0

    attributed_conv = credits["Email"] * conv          # 400
    attributed_rev = attributed_conv * 100.0            # 40,000
    expected_roas = attributed_rev / dollars_email
    expected_cpa = dollars_email / attributed_conv

    np.testing.assert_allclose(out["channel_roas"]["Email"], expected_roas, rtol=1e-9)
    np.testing.assert_allclose(out["channel_cpa"]["Email"], expected_cpa, rtol=1e-9)


def test_roas_cpa_consistency_all_paid():
    """ROAS·CPA = revenue_per_conversion for every paid channel (algebraic identity)."""
    bc = default_budget_config(revenue_per_conversion=100.0)
    attr = _make_attribution(_balanced_credits())
    out = optimize_budget(attr, bc, total_conversions=500)
    for ch in _PAID_COSTS:
        np.testing.assert_allclose(
            out["channel_roas"][ch] * out["channel_cpa"][ch], 100.0, rtol=1e-9
        )


# ============================================================
# 8. efficiency_ranking sorted descending
# ============================================================

def test_efficiency_ranking_sorted_desc():
    """efficiency_ranking lists paid channels by efficiency, highest first."""
    bc = default_budget_config()
    credits = _balanced_credits()
    attr = _make_attribution(credits)
    out = optimize_budget(attr, bc, total_conversions=1000)

    eff = {ch: credits[ch] / _PAID_COSTS[ch] for ch in _PAID_COSTS}
    expected_order = sorted(eff, key=eff.get, reverse=True)
    assert out["efficiency_ranking"] == expected_order
    # Email (highest eff) first, Paid Search (lowest) last.
    assert out["efficiency_ranking"][0] == "Email"
    assert out["efficiency_ranking"][-1] == "Paid Search"

    # Ranking is consistent with allocation_fraction magnitude.
    fracs = [out["allocation_fraction"][ch] for ch in out["efficiency_ranking"]]
    assert all(a >= b - 1e-12 for a, b in zip(fracs, fracs[1:]))


# ============================================================
# 9. Uniform fallback — all paid credits zero
# ============================================================

def test_uniform_fallback_all_paid_credits_zero():
    """When every paid channel has 0 credit, allocation splits uniformly.

    Credits live only on zero-cost channels → total efficiency = 0 → fallback.
    """
    bc = default_budget_config()
    credits = {
        "Display": 0.0,
        "Social": 0.0,
        "Paid Search": 0.0,
        "Email": 0.0,
        "Organic Search": 0.5,
        "Referral": 0.3,
        "Direct": 0.2,
    }
    np.testing.assert_allclose(sum(credits.values()), 1.0, atol=1e-9)
    attr = _make_attribution(credits)
    out = optimize_budget(attr, bc, total_conversions=1000)

    # 4 paid channels split evenly.
    assert set(out["allocation_fraction"].keys()) == set(_PAID_COSTS.keys())
    for ch in _PAID_COSTS:
        np.testing.assert_allclose(out["allocation_fraction"][ch], 0.25, atol=1e-9)
    np.testing.assert_allclose(
        sum(out["allocation_fraction"].values()), 1.0, atol=1e-9
    )
    # ROAS/CPA are 0 because attributed conversions are 0 for these channels.
    for ch in _PAID_COSTS:
        np.testing.assert_allclose(out["channel_roas"][ch], 0.0, atol=1e-9)
        np.testing.assert_allclose(out["channel_cpa"][ch], 0.0, atol=1e-9)


# ============================================================
# 10. Determinism — same inputs → identical output
# ============================================================

def test_optimize_budget_deterministic():
    bc = default_budget_config()
    attr = _make_attribution(_balanced_credits())
    out1 = optimize_budget(attr, bc, total_conversions=777)
    out2 = optimize_budget(attr, bc, total_conversions=777)
    assert out1["allocation_fraction"] == out2["allocation_fraction"]
    assert out1["allocation_dollars"] == out2["allocation_dollars"]
    assert out1["channel_roas"] == out2["channel_roas"]
    assert out1["channel_cpa"] == out2["channel_cpa"]
    assert out1["efficiency_ranking"] == out2["efficiency_ranking"]
