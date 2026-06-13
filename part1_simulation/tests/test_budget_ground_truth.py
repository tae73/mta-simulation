"""Unit tests for evaluation/budget_ground_truth.py (Approach A: Linear Response).

Ground truth optimal budget allocation derived from known DGP parameters:
    marginal_effect_k = β_k × E[f_k(Δt)]
    efficiency_k      = marginal_effect_k / c_k        (paid channels only)
    allocation_k      ∝ efficiency_k                   (linear response)

Sections:
    1.  compute_channel_marginal_effect  — non-negativity, decay structure
    2.  compute_channel_efficiency       — paid-only keys, effect/cost formula
    3.  compute_optimal_allocation       — allocation invariants, ranking
    4.  uniform fallback                 — total_eff == 0 → 1/n_paid split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.conversion_model import compute_temporal_decay
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.budget_ground_truth import (
    compute_channel_efficiency,
    compute_channel_marginal_effect,
    compute_optimal_allocation,
)
from part1_simulation.tests._journey_factory import (
    default_budget_config,
    default_dgp_config,
    make_journeys,
)

# Channels with non-zero cost (the only ones eligible for budget allocation).
PAID_CHANNELS = {"Display", "Social", "Paid Search", "Email"}
ZERO_COST_CHANNELS = {"Organic Search", "Referral", "Direct"}


# ============================================================
# Shared fixtures-as-functions (no pytest fixtures, per style)
# ============================================================

def _generated_journeys() -> pd.DataFrame:
    """Deterministic small DGP sample (calibrate=False for speed + reproducibility)."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    journeys, _ = generate_all_journeys(config, calibrate=False)
    return journeys


# ============================================================
# 1. compute_channel_marginal_effect
# ============================================================

def test_marginal_effect_all_nonnegative():
    """marginal_effect_k = β_k × E[f_k] ≥ 0 for every channel (β>0, f∈(0,1])."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    journeys = _generated_journeys()
    effects = compute_channel_marginal_effect(config, journeys)

    assert set(effects.keys()) == set(CHANNEL_NAMES)
    for ch, val in effects.items():
        assert val >= 0.0, f"channel {ch} has negative marginal effect {val}"
        assert np.isfinite(val)


def test_marginal_effect_zero_for_absent_channel():
    """A channel with no touchpoints in the data → marginal effect exactly 0.0."""
    config = default_dgp_config()
    # Journeys only ever touch Organic Search → all other channels absent.
    j = make_journeys([
        (1, "New", ["Organic Search"], [0.0], True),
        (2, "Loyal", ["Organic Search", "Organic Search"], [0.0, 24.0], False),
    ])
    effects = compute_channel_marginal_effect(config, j)
    for ch in CHANNEL_NAMES:
        if ch == "Organic Search":
            assert effects[ch] > 0.0
        else:
            np.testing.assert_allclose(effects[ch], 0.0, atol=1e-9)


def test_marginal_effect_single_touchpoint_closed_form():
    """One Display touchpoint at the observation time → Δt=0 → f=1 → effect=β_Display.

    With a single touchpoint per user, observation_time == timestamp, so
    delta_t = 0 and the decay factor is exactly 1.0. The marginal effect then
    reduces to β_Display = 0.3.
    """
    config = default_dgp_config()
    j = make_journeys([(1, "New", ["Display"], [5.0], True)])
    effects = compute_channel_marginal_effect(config, j)
    # β_Display × f(0) = 0.3 × 1.0
    np.testing.assert_allclose(effects["Display"], 0.3, atol=1e-9)


def test_marginal_effect_matches_manual_decay_average():
    """Effect = β × mean(decay over Δt) reproduced by hand for a 2-touchpoint user."""
    config = default_dgp_config()
    # Single user: Display at t=0 and t=48h; observation_time = last ts = 48h.
    # Δt = {48, 0} hours. half_life = 14 days.
    j = make_journeys([(1, "New", ["Display", "Display"], [0.0, 48.0], True)])
    effects = compute_channel_marginal_effect(config, j)

    d0 = compute_temporal_decay(14.0, 48.0)  # first touchpoint, Δt=48
    d1 = compute_temporal_decay(14.0, 0.0)   # last touchpoint, Δt=0
    expected = 0.3 * np.mean([d0, d1])
    np.testing.assert_allclose(effects["Display"], expected, rtol=1e-9)


# ============================================================
# 2. compute_channel_efficiency
# ============================================================

def test_efficiency_keys_are_paid_channels_only():
    """Zero-cost channels (Organic Search/Referral/Direct) excluded from efficiency."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    effects = compute_channel_marginal_effect(config, journeys)
    eff = compute_channel_efficiency(effects, budget)

    assert set(eff.keys()) == PAID_CHANNELS
    assert ZERO_COST_CHANNELS.isdisjoint(set(eff.keys()))


def test_efficiency_is_effect_over_base_cost():
    """efficiency_k = marginal_effect_k / base_cost_per_touchpoint_k (paid channels)."""
    budget = default_budget_config()
    cost_lookup = {cd.channel_name: cd for cd in budget.cost_defs}
    effects = {ch: 1.0 for ch in CHANNEL_NAMES}  # uniform effect to isolate cost
    eff = compute_channel_efficiency(effects, budget)

    for ch in PAID_CHANNELS:
        base_cost = cost_lookup[ch].base_cost_per_touchpoint
        np.testing.assert_allclose(eff[ch], 1.0 / base_cost, rtol=1e-9)


def test_efficiency_excludes_zero_effect_present_keys():
    """A paid channel with marginal_effect 0 still appears (key) but value is 0."""
    budget = default_budget_config()
    effects = {ch: 0.0 for ch in CHANNEL_NAMES}
    effects["Paid Search"] = 2.4
    eff = compute_channel_efficiency(effects, budget)

    assert set(eff.keys()) == PAID_CHANNELS
    np.testing.assert_allclose(eff["Display"], 0.0, atol=1e-9)
    np.testing.assert_allclose(eff["Paid Search"], 2.4 / 2.50, rtol=1e-9)


def test_efficiency_monotonic_in_effect():
    """Higher marginal effect at fixed cost → strictly higher efficiency."""
    budget = default_budget_config()
    eff_low = compute_channel_efficiency({**{c: 0.0 for c in CHANNEL_NAMES}, "Email": 1.0}, budget)
    eff_high = compute_channel_efficiency({**{c: 0.0 for c in CHANNEL_NAMES}, "Email": 2.0}, budget)
    assert eff_high["Email"] > eff_low["Email"]


# ============================================================
# 3. compute_optimal_allocation
# ============================================================

def test_allocation_fraction_sums_to_one():
    """optimal_allocation_fraction sums to 1.0 across paid channels."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    fracs = out["optimal_allocation_fraction"]
    np.testing.assert_allclose(sum(fracs.values()), 1.0, atol=1e-6)
    for v in fracs.values():
        assert v >= 0.0


def test_allocation_dollars_sum_to_total_budget():
    """optimal_allocation_dollars sums to total_budget (up to fraction rounding)."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    dollars = out["optimal_allocation_dollars"]
    np.testing.assert_allclose(
        sum(dollars.values()), budget.total_budget, atol=1e-6
    )
    np.testing.assert_allclose(out["total_budget"], budget.total_budget, atol=1e-9)


def test_allocation_keys_are_paid_only():
    """Both allocation dicts (fraction + dollars) key only paid channels."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    assert set(out["optimal_allocation_fraction"].keys()) == PAID_CHANNELS
    assert set(out["optimal_allocation_dollars"].keys()) == PAID_CHANNELS
    assert set(out["channel_efficiency"].keys()) == PAID_CHANNELS


def test_efficiency_ranking_sorted_desc():
    """efficiency_ranking is paid channels sorted by efficiency, descending."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    ranking = out["efficiency_ranking"]
    eff = out["channel_efficiency"]
    assert set(ranking) == PAID_CHANNELS
    ranked_vals = [eff[ch] for ch in ranking]
    assert ranked_vals == sorted(ranked_vals, reverse=True)


def test_allocation_fraction_proportional_to_efficiency():
    """frac_i / frac_j == eff_i / eff_j (linear response = proportional split)."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    eff = out["channel_efficiency"]
    fracs = out["optimal_allocation_fraction"]
    total_eff = sum(eff.values())
    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(fracs[ch], eff[ch] / total_eff, rtol=1e-9)


def test_allocation_dollars_equal_fraction_times_budget():
    """dollars_k == fraction_k × total_budget exactly."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    fracs = out["optimal_allocation_fraction"]
    dollars = out["optimal_allocation_dollars"]
    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(
            dollars[ch], fracs[ch] * budget.total_budget, rtol=1e-9
        )


def test_allocation_metadata_passthrough():
    """method + revenue_per_conversion echoed from config into output dict."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    out = compute_optimal_allocation(config, budget, journeys)

    assert out["method"] == "linear_response_efficiency"
    np.testing.assert_allclose(
        out["revenue_per_conversion"], budget.revenue_per_conversion, atol=1e-9
    )
    assert set(out["marginal_effects"].keys()) == set(CHANNEL_NAMES)


def test_allocation_deterministic():
    """Same inputs → identical allocation (pure function over fixed-seed data)."""
    config = default_dgp_config(n_users=1500, alpha_0=-2.5)
    budget = default_budget_config()
    journeys = _generated_journeys()
    a = compute_optimal_allocation(config, budget, journeys)
    b = compute_optimal_allocation(config, budget, journeys)
    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(
            a["optimal_allocation_fraction"][ch],
            b["optimal_allocation_fraction"][ch],
            atol=1e-12,
        )


# ============================================================
# 4. Uniform fallback (total efficiency == 0)
# ============================================================

def test_uniform_fallback_when_no_paid_touchpoints():
    """Journeys with only zero-cost channels → all paid effects 0 → uniform split.

    Targets the ``else`` branch of compute_optimal_allocation: when total_eff == 0
    it falls back to 1/n_paid across the paid-channel keys (which still exist in
    the efficiency dict because compute_channel_efficiency keys by paid CostDefs,
    independent of whether the effect is positive).
    """
    config = default_dgp_config()
    budget = default_budget_config()
    # Only zero-cost channels appear → every paid channel has 0 touchpoints.
    j = make_journeys([
        (1, "New", ["Organic Search", "Referral"], [0.0, 24.0], True),
        (2, "Loyal", ["Direct", "Organic Search"], [0.0, 12.0], False),
        (3, "Exploratory", ["Referral"], [5.0], True),
    ])
    out = compute_optimal_allocation(config, budget, j)

    eff = out["channel_efficiency"]
    fracs = out["optimal_allocation_fraction"]
    # All paid channels present with zero efficiency → uniform fallback.
    assert set(eff.keys()) == PAID_CHANNELS
    np.testing.assert_allclose(sum(eff.values()), 0.0, atol=1e-9)

    n_paid = len(PAID_CHANNELS)
    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(fracs[ch], 1.0 / n_paid, rtol=1e-9)
    np.testing.assert_allclose(sum(fracs.values()), 1.0, atol=1e-9)
    # Dollars still sum to the full budget under the fallback.
    np.testing.assert_allclose(
        sum(out["optimal_allocation_dollars"].values()),
        budget.total_budget,
        atol=1e-6,
    )


def test_uniform_fallback_dollars_equal_split():
    """Under fallback, each paid channel gets total_budget / n_paid dollars."""
    config = default_dgp_config()
    budget = default_budget_config()
    j = make_journeys([
        (1, "New", ["Organic Search"], [0.0], True),
        (2, "Loyal", ["Direct", "Referral"], [0.0, 6.0], True),
    ])
    out = compute_optimal_allocation(config, budget, j)
    dollars = out["optimal_allocation_dollars"]
    n_paid = len(PAID_CHANNELS)
    for ch in PAID_CHANNELS:
        np.testing.assert_allclose(
            dollars[ch], budget.total_budget / n_paid, rtol=1e-9
        )
