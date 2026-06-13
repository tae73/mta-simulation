"""Unit tests for rule_based.py (5 heuristic attribution methods).

Each method assigns per-touchpoint credit under a simple rule, aggregates to
channel level, and normalizes to sum=1.0. Tests verify:
    - output contract (dict over all channels, >= 0, sums to 1.0)
    - hand-checked credit splits on small deterministic journeys
    - the per-touchpoint vs per-unique-channel convention actually used
    - only converted journeys receive credit
    - the uniform fallback when no converted journeys exist
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.rule_based import (
    _get_converted_journeys,
    _normalize_credits,
    compute_first_click,
    compute_last_click,
    compute_linear,
    compute_position_based,
    compute_time_decay,
    run_all_rule_based,
)
from part1_simulation.tests._journey_factory import make_journeys


# ============================================================
# Helpers
# ============================================================

# The canonical single converted journey for hand-checks.
_SINGLE_PATH = ["Display", "Email", "Paid Search"]


def _single_journey() -> pd.DataFrame:
    """One converted 3-touchpoint journey: Display -> Email -> Paid Search."""
    return make_journeys([(1, "New", _SINGLE_PATH, [5.0, 10.0, 20.0], True)])


_ALL_METHODS = (
    compute_last_click,
    compute_first_click,
    compute_linear,
    compute_time_decay,
    compute_position_based,
)


# ============================================================
# 1. Output contract — every method, on a multi-user dataset
# ============================================================

def _multi_user_journeys() -> pd.DataFrame:
    return make_journeys([
        (1, "New", ["Display", "Email", "Paid Search"], [5.0, 10.0, 20.0], True),
        (2, "Loyal", ["Social", "Email"], [3.0, 8.0], True),
        (3, "New", ["Display"], [2.0], False),  # non-converter, must be ignored
        (4, "Exploratory", ["Organic Search", "Direct", "Referral"], [1.0, 4.0, 9.0], True),
        (5, "Loyal", ["Paid Search"], [6.0], True),
    ])


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_output_is_attribution_result(method):
    result = method(_multi_user_journeys())
    assert isinstance(result, AttributionResult)
    assert isinstance(result.method, str) and result.method
    assert isinstance(result.channel_credits, dict)
    assert isinstance(result.channel_credits_raw, dict)
    assert isinstance(result.metadata, dict)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_credits_cover_all_channels(method):
    credits = method(_multi_user_journeys()).channel_credits
    assert set(credits.keys()) == set(CHANNEL_NAMES)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_credits_nonnegative(method):
    credits = method(_multi_user_journeys()).channel_credits
    for ch, v in credits.items():
        assert v >= 0.0, f"{method.__name__}: negative credit for {ch}: {v}"


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_credits_sum_to_one(method):
    credits = method(_multi_user_journeys()).channel_credits
    np.testing.assert_allclose(sum(credits.values()), 1.0, atol=1e-9)


@pytest.mark.parametrize("method", _ALL_METHODS)
def test_credit_values_are_floats(method):
    credits = method(_multi_user_journeys()).channel_credits
    for v in credits.values():
        assert isinstance(v, float)


# ============================================================
# 2. Last Click — 100% to the last touchpoint
# ============================================================

def test_last_click_single_journey():
    result = compute_last_click(_single_journey())
    credits = result.channel_credits
    np.testing.assert_allclose(credits["Paid Search"], 1.0, atol=1e-9)
    for ch in CHANNEL_NAMES:
        if ch != "Paid Search":
            np.testing.assert_allclose(credits[ch], 0.0, atol=1e-9)
    assert result.method == "Last Click"


def test_last_click_uses_is_last_touchpoint_flag():
    """Credit follows the is_last_touchpoint flag (per-user last touch counts)."""
    j = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], True),   # last = Email
        (2, "Loyal", ["Social", "Email"], [3.0, 5.0], True),  # last = Email
        (3, "New", ["Paid Search", "Display"], [1.0, 9.0], True),  # last = Display
    ])
    raw = compute_last_click(j).channel_credits_raw
    # Two journeys end on Email, one on Display.
    np.testing.assert_allclose(raw["Email"], 2.0, atol=1e-9)
    np.testing.assert_allclose(raw["Display"], 1.0, atol=1e-9)
    credits = compute_last_click(j).channel_credits
    np.testing.assert_allclose(credits["Email"], 2.0 / 3.0, rtol=1e-9)
    np.testing.assert_allclose(credits["Display"], 1.0 / 3.0, rtol=1e-9)


# ============================================================
# 3. First Click — 100% to the first touchpoint (touchpoint_idx == 0)
# ============================================================

def test_first_click_single_journey():
    result = compute_first_click(_single_journey())
    credits = result.channel_credits
    np.testing.assert_allclose(credits["Display"], 1.0, atol=1e-9)
    for ch in CHANNEL_NAMES:
        if ch != "Display":
            np.testing.assert_allclose(credits[ch], 0.0, atol=1e-9)
    assert result.method == "First Click"


def test_first_click_counts_idx_zero():
    j = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], True),       # first = Display
        (2, "Loyal", ["Display", "Paid Search"], [3.0, 5.0], True),  # first = Display
        (3, "New", ["Social", "Email"], [1.0, 9.0], True),        # first = Social
    ])
    raw = compute_first_click(j).channel_credits_raw
    np.testing.assert_allclose(raw["Display"], 2.0, atol=1e-9)
    np.testing.assert_allclose(raw["Social"], 1.0, atol=1e-9)


# ============================================================
# 4. Linear — equal credit per touchpoint (1 / journey_length)
# ============================================================

def test_linear_single_journey_distinct_channels():
    """3 distinct channels in a 3-touchpoint journey -> each gets 1/3."""
    result = compute_linear(_single_journey())
    credits = result.channel_credits
    for ch in _SINGLE_PATH:
        np.testing.assert_allclose(credits[ch], 1.0 / 3.0, rtol=1e-9)
    for ch in CHANNEL_NAMES:
        if ch not in _SINGLE_PATH:
            np.testing.assert_allclose(credits[ch], 0.0, atol=1e-9)


def test_linear_is_per_touchpoint_not_per_unique_channel():
    """A channel appearing twice in a journey accrues 2x the per-touchpoint share.

    journey_length=3, so each touchpoint = 1/3. Display appears twice -> 2/3,
    Email once -> 1/3 (raw, pre-normalization for a single user already sums to 1).
    """
    j = make_journeys([(1, "New", ["Display", "Display", "Email"], [1.0, 2.0, 3.0], True)])
    raw = compute_linear(j).channel_credits_raw
    np.testing.assert_allclose(raw["Display"], 2.0 / 3.0, rtol=1e-9)
    np.testing.assert_allclose(raw["Email"], 1.0 / 3.0, rtol=1e-9)
    credits = compute_linear(j).channel_credits
    np.testing.assert_allclose(credits["Display"], 2.0 / 3.0, rtol=1e-9)
    np.testing.assert_allclose(credits["Email"], 1.0 / 3.0, rtol=1e-9)


# ============================================================
# 5. Time Decay — more recent touchpoints get >= earlier ones
# ============================================================

def test_time_decay_monotonic_by_recency():
    """Distinct channels at increasing timestamps -> credit increases toward the
    most recent touchpoint (largest credit on the last channel)."""
    j = make_journeys([(1, "New", _SINGLE_PATH, [5.0, 10.0, 20.0], True)])
    credits = compute_time_decay(j).channel_credits
    # recency order: Display (oldest) < Email < Paid Search (most recent)
    assert credits["Display"] <= credits["Email"] <= credits["Paid Search"]
    # The most recent touch (time_before_last = 0 -> weight 1) is strictly largest.
    assert credits["Paid Search"] > credits["Display"]


def test_time_decay_last_touchpoint_weight_is_one_before_norm():
    """The last touchpoint has time_before_last=0 -> raw decay weight 2^0 = 1.

    For a single journey the normalized credit equals weight_i / Σ weights.
    Verify the exact split using the 7-day default half-life (in hours).
    """
    timestamps = [5.0, 10.0, 20.0]
    half_life_hours = 7.0 * 24.0
    last = max(timestamps)
    weights = np.power(2.0, -(last - np.array(timestamps)) / half_life_hours)
    expected = weights / weights.sum()
    credits = compute_time_decay(_single_journey()).channel_credits
    np.testing.assert_allclose(credits["Display"], expected[0], rtol=1e-9)
    np.testing.assert_allclose(credits["Email"], expected[1], rtol=1e-9)
    np.testing.assert_allclose(credits["Paid Search"], expected[2], rtol=1e-9)


def test_time_decay_half_life_changes_concentration():
    """Shorter half-life -> more credit concentrated on the most recent touch."""
    j = _single_journey()
    short = compute_time_decay(j, half_life_days=1.0).channel_credits
    long = compute_time_decay(j, half_life_days=30.0).channel_credits
    # Shorter half-life decays faster, so the last channel keeps more relative credit.
    assert short["Paid Search"] > long["Paid Search"]
    assert short["Display"] < long["Display"]


def test_time_decay_method_string_reflects_half_life():
    result = compute_time_decay(_single_journey(), half_life_days=7.0)
    assert result.method == "Time Decay (7.0d)"
    assert result.metadata["half_life_days"] == 7.0


# ============================================================
# 6. Position-Based (U-shaped) — first/last larger than middle
# ============================================================

def test_position_based_three_touchpoints_40_20_40():
    """length=3 -> first 0.4, middle 0.2, last 0.4 (default weights)."""
    result = compute_position_based(_single_journey())
    credits = result.channel_credits
    np.testing.assert_allclose(credits["Display"], 0.4, rtol=1e-9)       # first
    np.testing.assert_allclose(credits["Email"], 0.2, rtol=1e-9)         # middle
    np.testing.assert_allclose(credits["Paid Search"], 0.4, rtol=1e-9)   # last
    # U-shape: endpoints >= middle
    assert credits["Display"] >= credits["Email"]
    assert credits["Paid Search"] >= credits["Email"]


def test_position_based_single_touchpoint_gets_full_credit():
    j = make_journeys([(1, "New", ["Email"], [3.0], True)])
    credits = compute_position_based(j).channel_credits
    np.testing.assert_allclose(credits["Email"], 1.0, atol=1e-9)


def test_position_based_two_touchpoints_5050():
    j = make_journeys([(1, "New", ["Display", "Paid Search"], [1.0, 5.0], True)])
    credits = compute_position_based(j).channel_credits
    np.testing.assert_allclose(credits["Display"], 0.5, rtol=1e-9)
    np.testing.assert_allclose(credits["Paid Search"], 0.5, rtol=1e-9)


def test_position_based_four_touchpoints_middle_split():
    """length=4 -> first 0.4, last 0.4, two middle each 0.2/2 = 0.1."""
    j = make_journeys([
        (1, "New", ["Display", "Email", "Social", "Paid Search"], [1.0, 2.0, 3.0, 4.0], True)
    ])
    credits = compute_position_based(j).channel_credits
    np.testing.assert_allclose(credits["Display"], 0.4, rtol=1e-9)       # first
    np.testing.assert_allclose(credits["Email"], 0.1, rtol=1e-9)         # middle
    np.testing.assert_allclose(credits["Social"], 0.1, rtol=1e-9)        # middle
    np.testing.assert_allclose(credits["Paid Search"], 0.4, rtol=1e-9)   # last


def test_position_based_custom_weights():
    """Custom 0.3/0.5 first/last -> middle = 0.2 for a 3-touch journey."""
    result = compute_position_based(_single_journey(), first_weight=0.3, last_weight=0.5)
    credits = result.channel_credits
    np.testing.assert_allclose(credits["Display"], 0.3, rtol=1e-9)
    np.testing.assert_allclose(credits["Email"], 0.2, rtol=1e-9)
    np.testing.assert_allclose(credits["Paid Search"], 0.5, rtol=1e-9)
    assert result.method == "Position-Based (30%/50%)"


# ============================================================
# 7. Converted-only behavior + uniform fallback
# ============================================================

def test_only_converted_journeys_credited():
    """A non-converter sharing a channel must not change credits vs its absence."""
    converted_only = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], True),
    ])
    with_non_converter = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], True),
        (2, "Loyal", ["Paid Search", "Social"], [1.0, 2.0], False),  # ignored
    ])
    for method in _ALL_METHODS:
        a = method(converted_only).channel_credits
        b = method(with_non_converter).channel_credits
        for ch in CHANNEL_NAMES:
            np.testing.assert_allclose(
                a[ch], b[ch], atol=1e-9,
                err_msg=f"{method.__name__} changed by non-converter on {ch}",
            )


def test_no_converted_journeys_uniform_fallback():
    """With zero converted journeys, _normalize_credits returns a uniform dict."""
    j = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], False),
        (2, "Loyal", ["Paid Search"], [3.0], False),
    ])
    for method in _ALL_METHODS:
        credits = method(j).channel_credits
        np.testing.assert_allclose(sum(credits.values()), 1.0, atol=1e-9)
        for ch in CHANNEL_NAMES:
            np.testing.assert_allclose(
                credits[ch], 1.0 / len(CHANNEL_NAMES), rtol=1e-9,
                err_msg=f"{method.__name__} not uniform on empty input for {ch}",
            )


def test_get_converted_journeys_filters_correctly():
    j = make_journeys([
        (1, "New", ["Display", "Email"], [1.0, 2.0], True),
        (2, "Loyal", ["Paid Search"], [3.0], False),
    ])
    conv = _get_converted_journeys(j)
    assert set(conv["user_id"]) == {1}
    assert conv["converted"].all()


# ============================================================
# 8. _normalize_credits helper
# ============================================================

def test_normalize_credits_sums_to_one_and_covers_channels():
    s = pd.Series({"Display": 2.0, "Email": 6.0})
    out = _normalize_credits(s)
    assert set(out.keys()) == set(CHANNEL_NAMES)
    np.testing.assert_allclose(sum(out.values()), 1.0, atol=1e-9)
    np.testing.assert_allclose(out["Display"], 0.25, rtol=1e-9)
    np.testing.assert_allclose(out["Email"], 0.75, rtol=1e-9)


def test_normalize_credits_zero_total_uniform():
    s = pd.Series({"Display": 0.0})
    out = _normalize_credits(s)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(out[ch], 1.0 / len(CHANNEL_NAMES), rtol=1e-9)


# ============================================================
# 9. Determinism + run_all_rule_based
# ============================================================

def test_methods_are_deterministic():
    j = _multi_user_journeys()
    for method in _ALL_METHODS:
        first = method(j).channel_credits
        second = method(j).channel_credits
        for ch in CHANNEL_NAMES:
            np.testing.assert_allclose(first[ch], second[ch], atol=1e-9)


def test_run_all_rule_based_returns_five_results():
    results = run_all_rule_based(_multi_user_journeys())
    assert len(results) == 5
    methods = [r.method for r in results]
    assert methods[0] == "Last Click"
    assert methods[1] == "First Click"
    assert methods[2] == "Linear"
    assert methods[3].startswith("Time Decay")
    assert methods[4].startswith("Position-Based")
    for r in results:
        assert isinstance(r, AttributionResult)
        np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)
