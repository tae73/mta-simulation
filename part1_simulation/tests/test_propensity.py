"""Unit tests for propensity.py (IPW + Doubly Robust causal attribution).

Maps to the public API of ``part1_simulation/models/causal/propensity.py``:
    - compute_ipw_attribution          — Inverse Propensity Weighting ATE
    - compute_doubly_robust_attribution — Outcome model + propensity (DR)
    - _build_user_level_data           — user-level confounder/exposure matrix

Tests assert the *verified* contract: ``channel_credits`` are clamped-then-
normalized to sum 1.0, all outputs are finite (no NaN/Inf), structure matches
``AttributionResult``, and estimation is deterministic under the source's fixed
``random_state=42``. Where a precise ATE is not analytically pinned down we
assert invariants (sums, bounds, key sets, determinism).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.models.causal.propensity import (
    _build_user_level_data,
    compute_doubly_robust_attribution,
    compute_ipw_attribution,
)
from part1_simulation.tests._journey_factory import default_dgp_config, make_journeys


# ============================================================
# Shared fixtures-as-functions (no pytest fixtures, per house style)
# ============================================================

def _signal_journeys(n_users: int = 1000) -> pd.DataFrame:
    """Generate journeys with enough conversion signal for propensity models.

    ``alpha_0=-2.5`` lifts the conversion rate well above the calibrated 2.5%
    target so both treated and control groups have converters/non-converters
    for every channel (avoids single-class logistic failures).
    """
    journeys, _ = generate_all_journeys(
        default_dgp_config(n_users=n_users, alpha_0=-2.5),
        calibrate=False,
    )
    return journeys


def _assert_attribution_contract(result: AttributionResult, method: str) -> None:
    """Shared structural + numeric invariants for any AttributionResult here."""
    assert isinstance(result, AttributionResult)
    assert result.method == method

    # channel_credits: keys are exactly the 7 channels, normalized, finite, [0,1]
    assert set(result.channel_credits.keys()) == set(CHANNEL_NAMES)
    credits = np.array([result.channel_credits[c] for c in CHANNEL_NAMES], dtype=float)
    assert np.all(np.isfinite(credits))
    assert np.all(credits >= 0.0)
    assert np.all(credits <= 1.0 + 1e-9)
    np.testing.assert_allclose(credits.sum(), 1.0, atol=1e-9)

    # raw credits: same keys, finite (may be negative — pre-clamp ATE estimates)
    assert set(result.channel_credits_raw.keys()) == set(CHANNEL_NAMES)
    raw = np.array(
        [result.channel_credits_raw[c] for c in CHANNEL_NAMES], dtype=float
    )
    assert np.all(np.isfinite(raw))

    # metadata is a dict carrying the estimator label
    assert isinstance(result.metadata, dict)
    assert "estimator" in result.metadata


# ============================================================
# 1. User-level design matrix — _build_user_level_data
# ============================================================

def test_build_user_level_data_one_row_per_user():
    """User-level frame has exactly one row per user_id with binary channel presence."""
    j = make_journeys([
        (1, "New", ["Display", "Email", "Display"], [1.0, 5.0, 9.0], True),
        (2, "Loyal", ["Paid Search"], [2.0], False),
        (3, "Exploratory", ["Social", "Email"], [3.0, 7.0], True),
    ])
    user_data = _build_user_level_data(j)

    assert len(user_data) == 3
    assert user_data["user_id"].nunique() == 3
    # all 7 channels are present as columns
    for ch in CHANNEL_NAMES:
        assert ch in user_data.columns
    # binary presence: Display seen twice by user 1 → clipped to 1
    row1 = user_data.set_index("user_id").loc[1]
    np.testing.assert_allclose(float(row1["Display"]), 1.0, atol=1e-9)
    np.testing.assert_allclose(float(row1["Email"]), 1.0, atol=1e-9)
    np.testing.assert_allclose(float(row1["Paid Search"]), 0.0, atol=1e-9)
    # converted is carried at user level
    assert bool(row1["converted"]) is True


def test_build_user_level_data_segment_dummies_drop_first():
    """Segment dummies use prefix 'seg_' and drop_first (k segments → k-1 dummies)."""
    j = make_journeys([
        (1, "New", ["Display"], [1.0], True),
        (2, "Loyal", ["Email"], [2.0], False),
        (3, "Exploratory", ["Social"], [3.0], True),
    ])
    user_data = _build_user_level_data(j)
    seg_cols = [c for c in user_data.columns if c.startswith("seg_")]
    # 3 distinct segments → 2 dummies (drop_first=True)
    assert len(seg_cols) == 2


# ============================================================
# 2. IPW — structure, normalization, finiteness
# ============================================================

@pytest.mark.slow
def test_ipw_attribution_contract():
    """IPW returns a well-formed AttributionResult with credits summing to 1.0."""
    j = _signal_journeys()
    r = compute_ipw_attribution(j)
    _assert_attribution_contract(r, method="IPW")
    assert r.metadata["estimator"] == "inverse_propensity_weighting"


@pytest.mark.slow
def test_ipw_no_nan_inf_in_raw_ate():
    """Raw IPW ATE estimates are finite for every channel (clipping prevents blowups)."""
    j = _signal_journeys()
    r = compute_ipw_attribution(j)
    for ch in CHANNEL_NAMES:
        assert np.isfinite(r.channel_credits_raw[ch]), f"non-finite raw ATE for {ch}"


@pytest.mark.slow
def test_ipw_deterministic():
    """IPW is deterministic on identical input (source pins random_state=42)."""
    j = _signal_journeys(n_users=600)
    r1 = compute_ipw_attribution(j)
    r2 = compute_ipw_attribution(j)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9
        )
        np.testing.assert_allclose(
            r1.channel_credits_raw[ch], r2.channel_credits_raw[ch], atol=1e-9
        )


# ============================================================
# 3. Doubly Robust — structure, normalization, finiteness
# ============================================================

@pytest.mark.slow
def test_doubly_robust_attribution_contract():
    """DR returns a well-formed AttributionResult with credits summing to 1.0."""
    j = _signal_journeys()
    r = compute_doubly_robust_attribution(j)
    _assert_attribution_contract(r, method="Doubly Robust")
    assert r.metadata["estimator"] == "doubly_robust"


@pytest.mark.slow
def test_doubly_robust_no_nan_inf_in_raw_ate():
    """Raw DR ATE estimates are finite for every channel."""
    j = _signal_journeys()
    r = compute_doubly_robust_attribution(j)
    for ch in CHANNEL_NAMES:
        assert np.isfinite(r.channel_credits_raw[ch]), f"non-finite raw ATE for {ch}"


@pytest.mark.slow
def test_doubly_robust_deterministic():
    """DR is deterministic on identical input (random_state=42 pinned in source)."""
    j = _signal_journeys(n_users=600)
    r1 = compute_doubly_robust_attribution(j)
    r2 = compute_doubly_robust_attribution(j)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9
        )
        np.testing.assert_allclose(
            r1.channel_credits_raw[ch], r2.channel_credits_raw[ch], atol=1e-9
        )


# ============================================================
# 4. Cross-method invariants
# ============================================================

@pytest.mark.slow
def test_ipw_and_dr_normalize_identically_to_unit_sum():
    """Both estimators apply the same clamp-then-normalize contract (Σ credits = 1)."""
    j = _signal_journeys()
    r_ipw = compute_ipw_attribution(j)
    r_dr = compute_doubly_robust_attribution(j)
    np.testing.assert_allclose(sum(r_ipw.channel_credits.values()), 1.0, atol=1e-9)
    np.testing.assert_allclose(sum(r_dr.channel_credits.values()), 1.0, atol=1e-9)


@pytest.mark.slow
def test_normalization_matches_manual_clamp_then_divide_ipw():
    """channel_credits == max(0, raw) / Σ max(0, raw) — the documented IPW contract."""
    j = _signal_journeys(n_users=800)
    r = compute_ipw_attribution(j)
    raw = r.channel_credits_raw
    clamped = {k: max(0.0, v) for k, v in raw.items()}
    total = sum(clamped.values())
    assert total > 0.0, "expected at least one positive ATE on real signal"
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r.channel_credits[ch], clamped[ch] / total, atol=1e-9
        )


@pytest.mark.slow
def test_clamped_channels_get_zero_credit_ipw():
    """Channels with negative raw ATE are clamped to exactly 0 credit (no leakage)."""
    j = _signal_journeys(n_users=800)
    r = compute_ipw_attribution(j)
    for ch in CHANNEL_NAMES:
        if r.channel_credits_raw[ch] < 0.0:
            np.testing.assert_allclose(r.channel_credits[ch], 0.0, atol=1e-9)
