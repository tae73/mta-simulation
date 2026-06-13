"""Regression tests for models/markov.py (Markov Chain + Removal Effect).

Covers the public API:
    - compute_markov_attribution(journeys, order=1|2, laplace_alpha)
    - build_transition_matrix_order1 / build_transition_matrix_order2
    - compute_removal_effect / compute_absorption_probability
    - _build_sequences (private sequence extractor)

The Markov chain absorbs into (Conversion)/(Null); per-channel credit is the
normalized Removal Effect (drop in absorption probability when the channel's
states are redirected to Null).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.markov import (
    _build_sequences,
    build_transition_matrix_order1,
    build_transition_matrix_order2,
    compute_absorption_probability,
    compute_markov_attribution,
    compute_removal_effect,
)
from part1_simulation.tests._journey_factory import make_journeys


# ============================================================
# Shared toy journeys
# ============================================================

# A mix of converting and non-converting paths. Spans every one of the 7
# channels at least once so the order-1 transition matrix has no all-zero rows.
_FULL_COVERAGE_SPECS: List[Tuple[int, str, List[str], List[float], bool]] = [
    (1, "New", ["Display", "Paid Search"], [0.0, 24.0], True),
    (2, "New", ["Social", "Email"], [0.0, 12.0], True),
    (3, "Exploratory", ["Organic Search", "Direct"], [0.0, 6.0], False),
    (4, "Loyal", ["Email", "Display", "Paid Search"], [0.0, 5.0, 30.0], True),
    (5, "New", ["Referral", "Display"], [0.0, 9.0], True),
    (6, "Loyal", ["Direct"], [0.0], False),
    (7, "Exploratory", ["Organic Search", "Referral"], [0.0, 14.0], False),
]


def _full_coverage_journeys() -> pd.DataFrame:
    return make_journeys(_FULL_COVERAGE_SPECS)


# ============================================================
# 1. _build_sequences — sequence + label extraction
# ============================================================

def test_build_sequences_extracts_ordered_channels_and_labels():
    j = make_journeys([
        (1, "New", ["Display", "Email", "Paid Search"], [5.0, 10.0, 20.0], True),
        (2, "Loyal", ["Direct"], [3.0], False),
    ])
    seqs = _build_sequences(j)
    assert seqs == [
        (["Display", "Email", "Paid Search"], True),
        (["Direct"], False),
    ]


def test_build_sequences_respects_touchpoint_order_not_row_order():
    """Sequence is sorted by touchpoint_idx, independent of DataFrame row order."""
    j = make_journeys([(1, "New", ["Display", "Email", "Paid Search"], [5.0, 10.0, 20.0], True)])
    shuffled = j.iloc[::-1].reset_index(drop=True)  # reverse row order
    (channels, converted), = _build_sequences(shuffled)
    assert channels == ["Display", "Email", "Paid Search"]
    assert converted is True


# ============================================================
# 2. Transition matrices — row-stochastic + absorbing structure
# ============================================================

def test_order1_transition_rows_sum_to_one_full_coverage():
    """Every state with outgoing transitions has a row summing to 1.0."""
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order1(seqs)
    row_sums = matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(states)), atol=1e-9)


def test_order2_transition_rows_sum_to_one_or_zero():
    """2nd-order rows are normalized to 1.0 (Laplace smoothing populates every row)."""
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order2(seqs)
    row_sums = matrix.sum(axis=1)
    # Laplace alpha (>=1e-3) gives every row a positive mass → all sum to 1.0.
    np.testing.assert_allclose(row_sums, np.ones(len(states)), atol=1e-9)


def test_order1_absorbing_states_are_self_loops():
    """(Conversion) and (Null) absorb: row is a one-hot on themselves."""
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order1(seqs)
    idx = {s: i for i, s in enumerate(states)}
    conv, null = idx["(Conversion)"], idx["(Null)"]
    np.testing.assert_allclose(matrix[conv, conv], 1.0, atol=1e-9)
    np.testing.assert_allclose(matrix[null, null], 1.0, atol=1e-9)
    np.testing.assert_allclose(matrix[conv].sum() - matrix[conv, conv], 0.0, atol=1e-9)
    np.testing.assert_allclose(matrix[null].sum() - matrix[null, null], 0.0, atol=1e-9)


def test_order1_unused_channel_row_is_all_zero():
    """A channel that never appears in any sequence has no outgoing transitions;
    its row stays all-zero (the guard `row_sums[row_sums==0]=1` divides 0 by 1)."""
    # Only Display/Email appear → the other 5 channels are unused.
    seqs = _build_sequences(make_journeys([
        (1, "New", ["Display", "Email"], [0.0, 5.0], True),
        (2, "New", ["Email", "Display"], [0.0, 5.0], False),
    ]))
    matrix, states = build_transition_matrix_order1(seqs)
    idx = {s: i for i, s in enumerate(states)}
    # Referral never appears → its row sums to 0.
    np.testing.assert_allclose(matrix[idx["Referral"]].sum(), 0.0, atol=1e-9)
    # Display does appear → its row is a proper distribution.
    np.testing.assert_allclose(matrix[idx["Display"]].sum(), 1.0, atol=1e-9)


def test_order1_matrix_entries_in_unit_interval():
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, _ = build_transition_matrix_order1(seqs)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0 + 1e-9


# ============================================================
# 3. Absorption probability
# ============================================================

def test_absorption_probability_is_a_probability():
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order1(seqs)
    p = compute_absorption_probability(matrix, states)
    assert 0.0 <= p <= 1.0 + 1e-9


def test_absorption_probability_all_converters_is_one():
    """If every observed path converts, P(Conversion | Start) == 1.0."""
    seqs = _build_sequences(make_journeys([
        (1, "New", ["Display", "Paid Search"], [0.0, 5.0], True),
        (2, "Loyal", ["Email"], [0.0], True),
        (3, "New", ["Social", "Email"], [0.0, 8.0], True),
    ]))
    matrix, states = build_transition_matrix_order1(seqs)
    p = compute_absorption_probability(matrix, states)
    np.testing.assert_allclose(p, 1.0, atol=1e-9)


def test_absorption_probability_all_non_converters_is_zero():
    seqs = _build_sequences(make_journeys([
        (1, "New", ["Display", "Paid Search"], [0.0, 5.0], False),
        (2, "Loyal", ["Email"], [0.0], False),
    ]))
    matrix, states = build_transition_matrix_order1(seqs)
    p = compute_absorption_probability(matrix, states)
    np.testing.assert_allclose(p, 0.0, atol=1e-9)


# ============================================================
# 4. Removal Effect — non-negative
# ============================================================

def test_removal_effects_non_negative():
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order1(seqs)
    for ch in CHANNEL_NAMES:
        effect = compute_removal_effect(matrix, states, ch)
        assert effect >= 0.0, f"removal effect for {ch} negative: {effect}"


def test_removal_effect_bounded_by_base_probability():
    """Removal effect = base - removed <= base (removed prob >= 0)."""
    seqs = _build_sequences(_full_coverage_journeys())
    matrix, states = build_transition_matrix_order1(seqs)
    base = compute_absorption_probability(matrix, states)
    for ch in CHANNEL_NAMES:
        effect = compute_removal_effect(matrix, states, ch)
        assert effect <= base + 1e-9


def test_removal_effect_sole_channel_equals_base():
    """If a single channel carries every path, removing it drops conversion to 0,
    so its removal effect equals the base conversion probability."""
    seqs = _build_sequences(make_journeys([
        (1, "New", ["Paid Search"], [0.0], True),
        (2, "Loyal", ["Paid Search"], [0.0], True),
        (3, "New", ["Paid Search"], [0.0], False),
    ]))
    matrix, states = build_transition_matrix_order1(seqs)
    base = compute_absorption_probability(matrix, states)
    effect = compute_removal_effect(matrix, states, "Paid Search")
    np.testing.assert_allclose(effect, base, atol=1e-9)


# ============================================================
# 5. compute_markov_attribution — output contract (order 1 & 2)
# ============================================================

@pytest.mark.parametrize("order", [1, 2])
def test_attribution_is_attribution_result(order):
    r = compute_markov_attribution(_full_coverage_journeys(), order=order)
    assert isinstance(r, AttributionResult)
    assert r.method == f"Markov (order={order})"


@pytest.mark.parametrize("order", [1, 2])
def test_attribution_credits_sum_to_one(order):
    r = compute_markov_attribution(_full_coverage_journeys(), order=order)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_attribution_covers_all_channels(order):
    r = compute_markov_attribution(_full_coverage_journeys(), order=order)
    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
    assert set(r.channel_credits_raw.keys()) == set(CHANNEL_NAMES)


@pytest.mark.parametrize("order", [1, 2])
def test_attribution_credits_non_negative(order):
    r = compute_markov_attribution(_full_coverage_journeys(), order=order)
    for ch, v in r.channel_credits.items():
        assert v >= 0.0, f"normalized credit for {ch} negative: {v}"
    for ch, v in r.channel_credits_raw.items():
        assert v >= 0.0, f"raw removal effect for {ch} negative: {v}"


@pytest.mark.parametrize("order", [1, 2])
def test_attribution_metadata_fields(order):
    r = compute_markov_attribution(_full_coverage_journeys(), order=order)
    assert r.metadata["order"] == order
    assert 0.0 <= r.metadata["base_conversion_prob"] <= 1.0 + 1e-9
    assert r.metadata["n_states"] > len(CHANNEL_NAMES)


def test_order2_has_more_states_than_order1():
    j = _full_coverage_journeys()
    r1 = compute_markov_attribution(j, order=1)
    r2 = compute_markov_attribution(j, order=2)
    # order-2 adds all (ch_i, ch_j) pair states.
    assert r2.metadata["n_states"] > r1.metadata["n_states"]


# ============================================================
# 6. Determinism
# ============================================================

@pytest.mark.parametrize("order", [1, 2])
def test_attribution_is_deterministic(order):
    j = _full_coverage_journeys()
    r_a = compute_markov_attribution(j, order=order)
    r_b = compute_markov_attribution(j, order=order)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r_a.channel_credits[ch], r_b.channel_credits[ch], atol=1e-9
        )
        np.testing.assert_allclose(
            r_a.channel_credits_raw[ch], r_b.channel_credits_raw[ch], atol=1e-9
        )


# ============================================================
# 7. Order validation
# ============================================================

@pytest.mark.parametrize("bad_order", [0, 3, 5, -1])
def test_unsupported_order_raises(bad_order):
    with pytest.raises(ValueError):
        compute_markov_attribution(_full_coverage_journeys(), order=bad_order)


# ============================================================
# 8. Degenerate / empty inputs are handled without crashing
# ============================================================

def test_empty_journeys_uniform_fallback():
    """No sequences → uniform 1/7 credit (graceful, no crash)."""
    empty = pd.DataFrame(
        {"user_id": [], "touchpoint_idx": [], "channel": [], "converted": []}
    )
    r = compute_markov_attribution(empty, order=1)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)
    for v in r.channel_credits.values():
        np.testing.assert_allclose(v, 1.0 / len(CHANNEL_NAMES), atol=1e-9)


def test_single_user_single_touch_runs():
    j = make_journeys([(1, "New", ["Display"], [0.0], True)])
    r = compute_markov_attribution(j, order=1)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)
    # Display is the only path to conversion → it should carry all the credit.
    np.testing.assert_allclose(r.channel_credits["Display"], 1.0, atol=1e-9)


def test_all_non_converters_uniform_fallback():
    """No conversions anywhere → all removal effects 0 → uniform fallback."""
    j = make_journeys([
        (1, "New", ["Display", "Email"], [0.0, 5.0], False),
        (2, "Loyal", ["Paid Search"], [0.0], False),
    ])
    r = compute_markov_attribution(j, order=1)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)
    for v in r.channel_credits.values():
        np.testing.assert_allclose(v, 1.0 / len(CHANNEL_NAMES), atol=1e-9)


def test_single_touch_order2_runs():
    """order-2 with only length-1 paths exercises the len(channels)==1 branch."""
    j = make_journeys([
        (1, "New", ["Display"], [0.0], True),
        (2, "Loyal", ["Email"], [0.0], False),
    ])
    r = compute_markov_attribution(j, order=2)
    np.testing.assert_allclose(sum(r.channel_credits.values()), 1.0, atol=1e-9)
    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
