"""Unit tests for dgp/cost_model.py (observation-only cost layer).

The cost layer attaches per-touchpoint costs AFTER conversion decisions and must
NOT mutate any DGP mechanics. Tests assert:
    1.  Observation-only invariant — only ``touchpoint_cost`` is (re)populated.
    2.  Zero-cost channels (Organic Search / Referral / Direct) → exactly 0.0.
    3.  Paid channels (Display / Social / Paid Search / Email) → strictly > 0.
    4.  Noise model: σ=0 → cost = base × segment_multiplier exactly; determinism
        under a fixed Generator seed; log-normal noise is multiplicative.
    5.  compute_cost_summary — key set, total_spend == Σ touchpoint_cost,
        per-channel totals, and the 0-converter guard (no ZeroDivision).
"""
from __future__ import annotations

import numpy as np

from part1_simulation.dgp.cost_model import (
    _build_cost_lookup,
    assign_touchpoint_costs,
    compute_cost_summary,
)
from part1_simulation.tests._journey_factory import (
    default_budget_config,
    make_journeys,
)

ZERO_COST_CHANNELS = ("Organic Search", "Referral", "Direct")
PAID_CHANNELS = ("Display", "Social", "Paid Search", "Email")

PRE_EXISTING_COLS = (
    "user_id",
    "segment",
    "touchpoint_idx",
    "channel",
    "timestamp",
    "is_last_touchpoint",
    "converted",
    "journey_length",
    "conversion_intensity",
)


def _mixed_journeys():
    """Mix of paid + zero channels, converters + non-converters, all segments."""
    return make_journeys([
        (1, "New", ["Display", "Organic Search", "Paid Search"], [1.0, 2.0, 3.0], True),
        (2, "Loyal", ["Email", "Referral", "Direct"], [1.0, 2.0, 3.0], False),
        (3, "Exploratory", ["Social", "Display"], [1.0, 4.0], True),
        (4, "New", ["Referral"], [2.0], False),
    ])


# ============================================================
# 1. Observation-only invariant
# ============================================================

def test_observation_only_pre_existing_columns_unchanged():
    """Every pre-existing column is byte-for-byte identical after cost assignment."""
    j = _mixed_journeys()
    snapshot = j.copy(deep=True)
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(0))

    for col in PRE_EXISTING_COLS:
        # equal_nan handled by direct array equality on object/numeric dtypes
        assert (out[col].values == snapshot[col].values).all(), f"column {col} mutated"


def test_observation_only_does_not_mutate_input_frame():
    """The input DataFrame itself is not modified in place (assign returns a copy)."""
    j = _mixed_journeys()
    before = j["touchpoint_cost"].copy()
    _ = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(0))
    # Original frame's touchpoint_cost still the factory default (all zeros).
    np.testing.assert_allclose(j["touchpoint_cost"].values, before.values, atol=1e-9)
    np.testing.assert_allclose(j["touchpoint_cost"].values, 0.0, atol=1e-9)


def test_only_touchpoint_cost_column_added():
    """Output has exactly the input columns (touchpoint_cost already present)."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(0))
    assert set(out.columns) == set(j.columns)
    assert "touchpoint_cost" in out.columns
    assert len(out) == len(j)


# ============================================================
# 2. Zero-cost channels → exactly 0.0 (no noise)
# ============================================================

def test_zero_cost_channels_exactly_zero():
    """Organic Search / Referral / Direct get touchpoint_cost == 0.0 with no noise."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(11))
    zero_mask = out["channel"].isin(ZERO_COST_CHANNELS)
    assert zero_mask.any(), "test fixture must include zero-cost channels"
    np.testing.assert_allclose(out.loc[zero_mask, "touchpoint_cost"].values, 0.0, atol=1e-9)


def test_zero_cost_channels_zero_for_every_seed():
    """Zero-cost stays exactly 0.0 regardless of RNG draw (noise never applied)."""
    j = _mixed_journeys()
    bc = default_budget_config()
    for seed in range(5):
        out = assign_touchpoint_costs(j, bc, np.random.default_rng(seed))
        zero_mask = out["channel"].isin(ZERO_COST_CHANNELS)
        np.testing.assert_allclose(
            out.loc[zero_mask, "touchpoint_cost"].values, 0.0, atol=1e-9
        )


# ============================================================
# 3. Paid channels → strictly positive
# ============================================================

def test_paid_channels_strictly_positive():
    """Display / Social / Paid Search / Email get touchpoint_cost > 0."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(3))
    paid_mask = out["channel"].isin(PAID_CHANNELS)
    assert paid_mask.any(), "test fixture must include paid channels"
    assert (out.loc[paid_mask, "touchpoint_cost"].values > 0.0).all()


# ============================================================
# 4. Noise model — σ=0 exactness, determinism, multiplicativity
# ============================================================

def test_sigma_zero_gives_base_times_multiplier_exactly():
    """With cost_noise_sigma=0, paid cost == base_cost × segment_multiplier exactly."""
    # Display(New): 0.005 * 1.2 = 0.006 ; Paid Search(New): 2.50 * 1.1 = 2.75
    # Social(Loyal): 0.008 * 0.7 = 0.0056 ; Email(Loyal): 0.003 * 1.0 = 0.003
    j = make_journeys([
        (1, "New", ["Display", "Paid Search"], [1.0, 2.0], True),
        (2, "Loyal", ["Social", "Email"], [1.0, 2.0], True),
    ])
    bc = default_budget_config(cost_noise_sigma=0.0)
    out = assign_touchpoint_costs(j, bc, np.random.default_rng(0))
    costs = dict(zip(out["channel"], out["touchpoint_cost"]))
    np.testing.assert_allclose(costs["Display"], 0.005 * 1.2, atol=1e-9)
    np.testing.assert_allclose(costs["Paid Search"], 2.50 * 1.1, atol=1e-9)
    np.testing.assert_allclose(costs["Social"], 0.008 * 0.7, atol=1e-9)
    np.testing.assert_allclose(costs["Email"], 0.003 * 1.0, atol=1e-9)


def test_assignment_deterministic_under_fixed_seed():
    """Same Generator seed → identical touchpoint_cost vectors."""
    j = _mixed_journeys()
    bc = default_budget_config()
    a = assign_touchpoint_costs(j, bc, np.random.default_rng(123))
    b = assign_touchpoint_costs(j, bc, np.random.default_rng(123))
    np.testing.assert_allclose(
        a["touchpoint_cost"].values, b["touchpoint_cost"].values, atol=1e-9
    )


def test_noise_is_multiplicative_lognormal_around_base():
    """Mean paid cost over many draws ≈ base × mult × exp(σ²/2) (log-normal mean)."""
    # Single paid touchpoint repeated → empirical mean of exp(N(0,σ²)) ≈ exp(σ²/2).
    sigma = 0.1
    base = 0.005 * 1.2  # Display(New)
    rng = np.random.default_rng(2024)
    n = 40_000
    j = make_journeys([(uid, "New", ["Display"], [1.0], True) for uid in range(n)])
    bc = default_budget_config(cost_noise_sigma=sigma)
    out = assign_touchpoint_costs(j, bc, rng)
    empirical_mean = out["touchpoint_cost"].mean()
    expected_mean = base * np.exp(sigma ** 2 / 2.0)
    # 40k samples → loose tolerance on the Monte-Carlo mean.
    np.testing.assert_allclose(empirical_mean, expected_mean, rtol=2e-2)
    # All draws strictly positive (exp() is positive, base > 0).
    assert (out["touchpoint_cost"].values > 0.0).all()


def test_cost_lookup_effective_base_costs():
    """_build_cost_lookup keys (channel, segment) → (base×mult, cost_type)."""
    lookup = _build_cost_lookup(default_budget_config())
    # Effective base = base_cost × segment_multiplier.
    np.testing.assert_allclose(lookup[("Display", "New")][0], 0.005 * 1.2, atol=1e-9)
    np.testing.assert_allclose(lookup[("Paid Search", "Loyal")][0], 2.50 * 0.9, atol=1e-9)
    assert lookup[("Display", "New")][1] == "cpm"
    assert lookup[("Organic Search", "Loyal")][1] == "zero"
    np.testing.assert_allclose(lookup[("Direct", "New")][0], 0.0, atol=1e-9)


# ============================================================
# 5. compute_cost_summary
# ============================================================

def test_cost_summary_keys():
    """Summary dict exposes the documented key set."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(5))
    summary = compute_cost_summary(out)
    assert set(summary.keys()) == {
        "channel_total_cost",
        "channel_avg_cost_per_touchpoint",
        "total_spend",
        "cost_per_conversion",
        "n_converters",
    }


def test_cost_summary_total_spend_equals_sum():
    """total_spend == Σ touchpoint_cost across all rows."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(6))
    summary = compute_cost_summary(out)
    np.testing.assert_allclose(
        summary["total_spend"], out["touchpoint_cost"].sum(), atol=1e-9
    )
    # total_spend == Σ of the per-channel totals as well.
    np.testing.assert_allclose(
        summary["total_spend"],
        sum(summary["channel_total_cost"].values()),
        atol=1e-9,
    )


def test_cost_summary_channel_totals_match_groupby():
    """Per-channel totals equal a direct groupby sum; zero channels report 0.0."""
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(8))
    summary = compute_cost_summary(out)
    expected = out.groupby("channel", observed=True)["touchpoint_cost"].sum().to_dict()
    for ch, total in expected.items():
        np.testing.assert_allclose(
            summary["channel_total_cost"][ch], total, atol=1e-9
        )
    # Referral / Organic Search totals are exactly 0.0.
    for ch in ("Referral", "Organic Search"):
        if ch in summary["channel_total_cost"]:
            np.testing.assert_allclose(summary["channel_total_cost"][ch], 0.0, atol=1e-9)


def test_cost_summary_n_converters_counts_unique_users():
    """n_converters counts distinct converted user_ids (not touchpoints)."""
    # Users 1 & 3 convert (multi-touch); user 2 & 4 do not.
    j = _mixed_journeys()
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(9))
    summary = compute_cost_summary(out)
    assert summary["n_converters"] == 2
    np.testing.assert_allclose(
        summary["cost_per_conversion"],
        summary["total_spend"] / summary["n_converters"],
        atol=1e-9,
    )


def test_cost_summary_zero_converters_no_zero_division():
    """With 0 converters, cost_per_conversion == 0.0 (guarded, no ZeroDivision)."""
    j = make_journeys([
        (1, "New", ["Display", "Paid Search"], [1.0, 2.0], False),
        (2, "Loyal", ["Email", "Direct"], [1.0, 2.0], False),
    ])
    out = assign_touchpoint_costs(j, default_budget_config(), np.random.default_rng(1))
    summary = compute_cost_summary(out)
    assert summary["n_converters"] == 0
    np.testing.assert_allclose(summary["cost_per_conversion"], 0.0, atol=1e-9)
    # total_spend is still the real (positive) spend even though no one converted.
    assert summary["total_spend"] > 0.0
