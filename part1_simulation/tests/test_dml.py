"""Smoke tests for dml.py (Double Machine Learning / partialling-out DML).

The implementation uses partialling-out DML with sklearn LogisticRegression
nuisance models and KFold cross-fitting (NOT EconML LinearDML). These tests
exercise the real public API end-to-end on a small generated DGP sample and
assert structural invariants: finite estimates, normalized credits, expected
metadata, and determinism.

Sections:
    1. User-level data construction (_build_user_level_data)
    2. DML ATE estimator (_estimate_dml_ate)
    3. Full attribution smoke test on generated journeys (compute_dml_attribution)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.models.causal.dml import (
    _build_user_level_data,
    _estimate_dml_ate,
    compute_dml_attribution,
)
from part1_simulation.tests._journey_factory import default_dgp_config, make_journeys


# ============================================================
# 1. User-level data construction — _build_user_level_data
# ============================================================

def test_build_user_level_one_row_per_user():
    """One row per user; all 7 channels present as binary presence columns."""
    j = make_journeys([
        (1, "New", ["Display", "Email", "Display"], [1.0, 2.0, 3.0], True),
        (2, "Loyal", ["Paid Search"], [1.0], False),
        (3, "Exploratory", ["Organic Search", "Direct"], [1.0, 2.0], True),
    ])
    ud = _build_user_level_data(j)

    assert ud["user_id"].nunique() == len(ud) == 3
    for ch in CHANNEL_NAMES:
        assert ch in ud.columns
    # Channel presence is clipped to {0, 1} even with repeated touchpoints.
    u1 = ud[ud["user_id"] == 1].iloc[0]
    assert u1["Display"] == 1  # appeared twice → clipped to 1
    assert u1["Email"] == 1
    assert u1["Paid Search"] == 0
    # converted/segment carried through from journey rows.
    assert bool(u1["converted"]) is True
    assert ud[ud["user_id"] == 2].iloc[0]["Paid Search"] == 1


def test_build_user_level_segment_dummies_present():
    """Segment dummies (drop_first) appear with the seg_ prefix; one fewer than #segments."""
    j = make_journeys([
        (1, "New", ["Display"], [1.0], True),
        (2, "Loyal", ["Email"], [1.0], False),
        (3, "Exploratory", ["Paid Search"], [1.0], True),
    ])
    ud = _build_user_level_data(j)
    seg_cols = [c for c in ud.columns if c.startswith("seg_")]
    # 3 distinct segments, drop_first=True → 2 dummy columns.
    assert len(seg_cols) == 2


# ============================================================
# 2. DML ATE estimator — _estimate_dml_ate
# ============================================================

def test_estimate_dml_ate_returns_finite_float():
    """ATE estimator returns a finite python float on a small random design."""
    rng = np.random.default_rng(0)
    n = 200
    W = rng.normal(size=(n, 3))
    T = (rng.random(n) < 0.5).astype(float)
    Y = (rng.random(n) < 0.4).astype(float)
    ate = _estimate_dml_ate(Y, T, W, n_folds=3)
    assert isinstance(ate, float)
    assert np.isfinite(ate)


def test_estimate_dml_ate_recovers_positive_effect_sign():
    """A strong positive treatment effect yields a clearly positive ATE, while a
    null-effect outcome yields an estimate near zero (sign/magnitude recovery)."""
    rng = np.random.default_rng(7)
    n = 600
    W = rng.normal(size=(n, 3))
    # Treatment depends weakly on W (both classes present → residual variance > 0).
    p_t = 1.0 / (1.0 + np.exp(-(0.3 * W[:, 0])))
    T = (rng.random(n) < p_t).astype(float)

    # Outcome strongly increased by treatment.
    p_y = 1.0 / (1.0 + np.exp(-(-0.5 + 1.5 * T + 0.2 * W[:, 1])))
    Y = (rng.random(n) < p_y).astype(float)
    ate_effect = _estimate_dml_ate(Y, T, W, n_folds=5)

    # Null outcome (no dependence on T).
    p_y0 = 1.0 / (1.0 + np.exp(-(-0.5 + 0.2 * W[:, 1])))
    Y0 = (rng.random(n) < p_y0).astype(float)
    ate_null = _estimate_dml_ate(Y0, T, W, n_folds=5)

    assert np.isfinite(ate_effect) and np.isfinite(ate_null)
    assert ate_effect > 0.2, f"expected clearly positive ATE, got {ate_effect}"
    assert ate_effect > ate_null, (
        f"effect ATE ({ate_effect}) should exceed null ATE ({ate_null})"
    )


# ============================================================
# 3. Full attribution smoke test — compute_dml_attribution
# ============================================================

@pytest.mark.slow
def test_dml_attribution_structure_and_finiteness():
    """End-to-end: DML attribution on generated journeys returns the expected
    structure with finite, normalized credits (no NaN/Inf)."""
    config = default_dgp_config(n_users=1200, alpha_0=-2.5)
    journeys, _ = generate_all_journeys(config, calibrate=False)

    result = compute_dml_attribution(journeys, n_folds=3)

    # Structure
    assert isinstance(result, AttributionResult)
    assert result.method == "DML"
    assert set(result.channel_credits.keys()) == set(CHANNEL_NAMES)
    assert set(result.channel_credits_raw.keys()) == set(CHANNEL_NAMES)
    assert result.metadata["n_folds"] == 3
    assert result.metadata["estimator"] == "partialling_out_dml"

    # Finiteness — raw ATE estimates and normalized credits.
    for ch in CHANNEL_NAMES:
        assert np.isfinite(result.channel_credits_raw[ch]), f"raw {ch} not finite"
        assert np.isfinite(result.channel_credits[ch]), f"credit {ch} not finite"

    # Normalized credits are a probability vector summing to 1.0.
    for v in result.channel_credits.values():
        assert v >= 0.0
    total = sum(result.channel_credits.values())
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


@pytest.mark.slow
def test_dml_attribution_deterministic():
    """Fixed seed + fixed KFold random_state → identical credits across runs."""
    config = default_dgp_config(n_users=1200, alpha_0=-2.5)
    journeys, _ = generate_all_journeys(config, calibrate=False)

    r1 = compute_dml_attribution(journeys, n_folds=3)
    r2 = compute_dml_attribution(journeys, n_folds=3)

    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9,
            err_msg=f"non-deterministic credit on {ch}",
        )
        np.testing.assert_allclose(
            r1.channel_credits_raw[ch], r2.channel_credits_raw[ch], atol=1e-9,
            err_msg=f"non-deterministic raw ATE on {ch}",
        )
