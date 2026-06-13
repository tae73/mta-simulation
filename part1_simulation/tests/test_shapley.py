"""Unit tests for shapley.py (exact Shapley Value attribution, 128 coalitions).

Two value-function versions (module docstring):
    Version A (conversion_rate): v(S) = conversion rate of journeys whose
        channel set is a subset of S.  -> compute_shapley_conversion_rate
    Version B (model_based): logistic regression, v(S) = mean predicted prob
        with non-S channels masked to 0. -> compute_shapley_model_based

Sections:
    1.  Coalition matrix construction (binary presence, all channels present)
    2.  Value function A — conversion-rate semantics (subset filter)
    3.  Efficiency — channel_credits sums to 1, all present, non-negative
    4.  Symmetry — interchangeable channels get equal credit
    5.  Hand-checkable 2-channel Shapley split (raw + normalized)
    6.  Empty / all-zero edge cases (uniform fallback)
    7.  Determinism + metadata invariants
    8.  Model-based version (Version B) shape / symmetry / determinism
"""
from __future__ import annotations

import itertools
import math
from typing import FrozenSet, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.shapley import (
    _build_user_channel_matrix,
    _compute_coalition_value_conversion_rate,
    _compute_exact_shapley,
    compute_shapley_conversion_rate,
    compute_shapley_model_based,
)
from part1_simulation.tests._journey_factory import make_journeys


# ============================================================
# Toy builders for the two-channel hand-checkable scenarios
# ============================================================

def _two_channel_specs(
    ch_a: str,
    ch_b: str,
    a_only: Tuple[int, int],
    b_only: Tuple[int, int],
    both: Tuple[int, int],
) -> List[Tuple[int, str, List[str], List[float], bool]]:
    """Build journeys for a 2-channel scenario.

    Each ``(total, converted)`` pair gives a group: ``total`` users, the first
    ``converted`` of which convert. Groups: A-only, B-only, A+B.
    """
    specs: List[Tuple[int, str, List[str], List[float], bool]] = []
    uid = 0
    for chans, (total, conv) in (
        ([ch_a], a_only),
        ([ch_b], b_only),
        ([ch_a, ch_b], both),
    ):
        ts = [float(i + 1) for i in range(len(chans))]
        for k in range(total):
            specs.append((uid, "New", chans, ts, k < conv))
            uid += 1
    return specs


# ============================================================
# 1. Coalition matrix construction
# ============================================================

def test_user_channel_matrix_binary_presence():
    """_build_user_channel_matrix → one row per user, binary 0/1 channel columns,
    all 7 canonical channels present, converted carried through."""
    j = make_journeys([
        (1, "New", ["Display", "Display", "Email"], [1.0, 2.0, 3.0], True),
        (2, "Loyal", ["Paid Search"], [1.0], False),
    ])
    um = _build_user_channel_matrix(j)

    # All canonical channels appear as columns (even unused ones).
    for ch in CHANNEL_NAMES:
        assert ch in um.columns
    # One row per user.
    assert sorted(um["user_id"].tolist()) == [1, 2]

    u1 = um[um["user_id"] == 1].iloc[0]
    # Repeated Display collapses to a single 1 (presence, not count).
    np.testing.assert_allclose(u1["Display"], 1.0, atol=1e-9)
    np.testing.assert_allclose(u1["Email"], 1.0, atol=1e-9)
    np.testing.assert_allclose(u1["Paid Search"], 0.0, atol=1e-9)
    assert bool(u1["converted"]) is True

    u2 = um[um["user_id"] == 2].iloc[0]
    np.testing.assert_allclose(u2["Paid Search"], 1.0, atol=1e-9)
    np.testing.assert_allclose(u2["Display"], 0.0, atol=1e-9)
    assert bool(u2["converted"]) is False


# ============================================================
# 2. Value function A — conversion-rate subset semantics
# ============================================================

def test_coalition_value_conversion_rate_subset_filter():
    """v(S) = conversion rate over users whose channel set ⊆ S.

    Display-only: 4 users, 2 convert. Email-only: 4 users, 2 convert.
    Both: 4 users, all 4 convert.
      v(∅)        = 0  (no eligible users)
      v({D})      = 2/4 = 0.5         (only Display-only users qualify)
      v({E})      = 2/4 = 0.5
      v({D,E})    = (2+2+4)/12 = 8/12 (all qualify)
    """
    specs = _two_channel_specs("Display", "Email", (4, 2), (4, 2), (4, 4))
    um = _build_user_channel_matrix(make_journeys(specs))

    def v(members: Sequence[str]) -> float:
        return _compute_coalition_value_conversion_rate(um, frozenset(members))

    np.testing.assert_allclose(v([]), 0.0, atol=1e-9)
    np.testing.assert_allclose(v(["Display"]), 0.5, atol=1e-9)
    np.testing.assert_allclose(v(["Email"]), 0.5, atol=1e-9)
    np.testing.assert_allclose(v(["Display", "Email"]), 8.0 / 12.0, rtol=1e-9)
    # Adding an unused channel to S does not change the value (no users have it).
    np.testing.assert_allclose(
        v(["Display", "Social"]), v(["Display"]), atol=1e-9
    )


# ============================================================
# 3. Efficiency — normalized credits sum to 1, all present, non-negative
# ============================================================

def test_efficiency_conversion_rate_sums_to_one():
    """channel_credits: every channel present, non-negative, sums to 1.0."""
    specs = _two_channel_specs("Display", "Email", (4, 2), (4, 2), (4, 4))
    r = compute_shapley_conversion_rate(make_journeys(specs))

    assert isinstance(r, AttributionResult)
    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
    for v in r.channel_credits.values():
        assert v >= 0.0
        assert np.isfinite(v)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)


def test_efficiency_model_based_sums_to_one():
    """Model-based version also yields a normalized, non-negative, complete dict."""
    specs = _two_channel_specs("Display", "Email", (6, 3), (6, 3), (6, 5))
    r = compute_shapley_model_based(make_journeys(specs))

    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
    for v in r.channel_credits.values():
        assert v >= 0.0
        assert np.isfinite(v)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)


# ============================================================
# 4. Symmetry — interchangeable channels get equal credit
# ============================================================

def test_symmetry_conversion_rate():
    """Display and Email are perfectly interchangeable in the constructed data →
    equal raw and normalized Shapley credit."""
    # A-only and B-only have identical conversion rates; the A+B group is
    # symmetric in the two channels.
    specs = _two_channel_specs("Display", "Email", (6, 3), (6, 3), (6, 4))
    r = compute_shapley_conversion_rate(make_journeys(specs))

    np.testing.assert_allclose(
        r.channel_credits_raw["Display"],
        r.channel_credits_raw["Email"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        r.channel_credits["Display"], r.channel_credits["Email"], atol=1e-9
    )
    # And they split the entire normalized credit (only two active channels).
    np.testing.assert_allclose(r.channel_credits["Display"], 0.5, atol=1e-9)


def test_symmetry_model_based():
    """Interchangeable channels → equal model-based raw Shapley credit."""
    specs = _two_channel_specs("Display", "Email", (6, 3), (6, 3), (6, 4))
    r = compute_shapley_model_based(make_journeys(specs))

    np.testing.assert_allclose(
        r.channel_credits_raw["Display"],
        r.channel_credits_raw["Email"],
        atol=1e-9,
    )


# ============================================================
# 5. Hand-checkable 2-channel Shapley split
# ============================================================

def test_two_channel_shapley_hand_value():
    """Exact 2-player Shapley on the conversion-rate value function.

    With only Display & Email present, all coalitions reduce to the 2-channel
    sub-game (extra channels add no value). Values:
      v(∅)=0, v({D})=0.5, v({E})=0.5, v({D,E})=2/3.
    Shapley (2-player):
      φ(D) = ½[(v({D})−v(∅)) + (v({D,E})−v({E}))]
           = ½[0.5 + (2/3 − 0.5)] = ½[0.5 + 1/6] = 1/3.
    By symmetry φ(E) = 1/3. Normalized → 0.5 each.
    """
    specs = _two_channel_specs("Display", "Email", (4, 2), (4, 2), (4, 4))
    r = compute_shapley_conversion_rate(make_journeys(specs))

    np.testing.assert_allclose(r.channel_credits_raw["Display"], 1.0 / 3.0, atol=1e-9)
    np.testing.assert_allclose(r.channel_credits_raw["Email"], 1.0 / 3.0, atol=1e-9)
    # Unused channels get exactly zero raw credit.
    for ch in CHANNEL_NAMES:
        if ch not in ("Display", "Email"):
            np.testing.assert_allclose(r.channel_credits_raw[ch], 0.0, atol=1e-9)
    np.testing.assert_allclose(r.channel_credits["Display"], 0.5, atol=1e-9)
    np.testing.assert_allclose(r.channel_credits["Email"], 0.5, atol=1e-9)


def test_two_channel_matches_2player_formula_directly():
    """_compute_exact_shapley over CHANNEL_NAMES reduces to the 2-player formula
    for the 2-channel sub-game (independent re-derivation of the weights)."""
    specs = _two_channel_specs("Display", "Email", (4, 2), (4, 2), (4, 4))
    um = _build_user_channel_matrix(make_journeys(specs))

    cache: dict = {}

    def value_fn(coalition: FrozenSet[str]) -> float:
        if coalition not in cache:
            cache[coalition] = _compute_coalition_value_conversion_rate(um, coalition)
        return cache[coalition]

    raw = _compute_exact_shapley(CHANNEL_NAMES, value_fn)

    # Independent 2-player reference on the reduced sub-game.
    v_empty = value_fn(frozenset())
    v_d = value_fn(frozenset({"Display"}))
    v_e = value_fn(frozenset({"Email"}))
    v_de = value_fn(frozenset({"Display", "Email"}))
    phi_d = 0.5 * ((v_d - v_empty) + (v_de - v_e))
    phi_e = 0.5 * ((v_e - v_empty) + (v_de - v_d))

    np.testing.assert_allclose(raw["Display"], phi_d, atol=1e-9)
    np.testing.assert_allclose(raw["Email"], phi_e, atol=1e-9)


def test_exact_shapley_weights_sum_to_one_per_channel():
    """For any single channel, the marginal-contribution weights over all
    coalitions of the other channels sum to 1 (Shapley weight normalization)."""
    n = len(CHANNEL_NAMES)
    others = list(CHANNEL_NAMES[1:])
    total_weight = 0.0
    for r in range(n):
        for S_tuple in itertools.combinations(others, r):
            weight = (
                math.factorial(len(S_tuple))
                * math.factorial(n - len(S_tuple) - 1)
                / math.factorial(n)
            )
            total_weight += weight
    np.testing.assert_allclose(total_weight, 1.0, atol=1e-9)


# ============================================================
# 6. Edge cases — empty coalition, all-zero conversions
# ============================================================

def test_empty_coalition_value_is_zero():
    """v(∅) = 0 for the conversion-rate value function."""
    specs = _two_channel_specs("Display", "Email", (2, 1), (2, 1), (2, 2))
    um = _build_user_channel_matrix(make_journeys(specs))
    np.testing.assert_allclose(
        _compute_coalition_value_conversion_rate(um, frozenset()), 0.0, atol=1e-9
    )


def test_no_conversions_uniform_fallback():
    """If no user converts, every coalition value is 0 → raw Shapley all 0 →
    normalized credits fall back to uniform 1/7."""
    j = make_journeys([
        (0, "New", ["Display"], [1.0], False),
        (1, "New", ["Email"], [1.0], False),
        (2, "New", ["Display", "Email"], [1.0, 2.0], False),
    ])
    r = compute_shapley_conversion_rate(j)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r.channel_credits[ch], 1.0 / len(CHANNEL_NAMES), atol=1e-9
        )
        np.testing.assert_allclose(r.channel_credits_raw[ch], 0.0, atol=1e-9)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)


# ============================================================
# 7. Determinism + metadata invariants
# ============================================================

def test_conversion_rate_determinism_and_metadata():
    """Conversion-rate Shapley is deterministic and exposes 128-coalition metadata."""
    specs = _two_channel_specs("Display", "Email", (4, 2), (4, 2), (4, 4))
    j = make_journeys(specs)
    r1 = compute_shapley_conversion_rate(j)
    r2 = compute_shapley_conversion_rate(j)

    assert r1.method == "Shapley (conv. rate)"
    assert r1.metadata["value_function"] == "conversion_rate"
    # All 2^7 coalitions are evaluated and cached.
    assert r1.metadata["n_coalitions"] == 128
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9
        )


def test_model_based_determinism_and_metadata():
    """Model-based Shapley uses a fixed random_state → deterministic across runs;
    metadata carries logistic coefficients for all channels."""
    specs = _two_channel_specs("Display", "Email", (6, 3), (6, 3), (6, 5))
    j = make_journeys(specs)
    r1 = compute_shapley_model_based(j)
    r2 = compute_shapley_model_based(j)

    assert r1.method == "Shapley (model-based)"
    assert r1.metadata["value_function"] == "model_based"
    assert r1.metadata["n_coalitions"] == 128
    assert set(r1.metadata["logistic_coefs"].keys()) == set(CHANNEL_NAMES)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9
        )
        np.testing.assert_allclose(
            r1.channel_credits_raw[ch], r2.channel_credits_raw[ch], atol=1e-9
        )
