"""Regression tests for evaluation/ground_truth.py.

Two ground-truth definitions are exercised:
    GT-A  compute_ground_truth_intensity  — intensity decomposition (primary).
    GT-B  compute_ground_truth_shapley    — counterfactual 128-coalition Shapley.
Plus the combined packager compute_all_ground_truths and the
_decompose_user_intensity cross-influence direction invariant.

Tests assert verified behavior (sum==1.0 invariants, channel coverage,
ranking order, no-converter edge) and a robust source-before-target
cross-influence check on a controlled toy journey set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.ground_truth import (
    _decompose_user_intensity,
    compute_all_ground_truths,
    compute_ground_truth_intensity,
    compute_ground_truth_shapley,
)
from part1_simulation.tests._journey_factory import (
    default_dgp_config,
    make_journeys,
    segment_by_name,
)


# ============================================================
# Shared data builder — small journeys WITH converters
# ============================================================

def _journeys_with_converters(n_users: int = 1500, alpha_0: float = -2.5):
    """Generate a small journey set with enough converters (raised alpha_0)."""
    config = default_dgp_config(n_users=n_users, alpha_0=alpha_0)
    journeys, _ = generate_all_journeys(config, calibrate=False)
    return journeys, config


# ============================================================
# 1. GT-A intensity decomposition — normalization + coverage
# ============================================================

def test_gt_a_sums_to_one_all_channels_present():
    """compute_ground_truth_intensity → credits over all 7 channels, sum == 1.0."""
    journeys, config = _journeys_with_converters()
    n_converters = journeys.groupby("user_id")["converted"].first().sum()
    assert n_converters >= 1, f"need >=1 converter, got {n_converters}"

    gt_a = compute_ground_truth_intensity(journeys, config)

    # All seven canonical channels present as keys.
    assert set(gt_a.keys()) == set(CHANNEL_NAMES)
    # Normalized to a probability simplex.
    np.testing.assert_allclose(sum(gt_a.values()), 1.0, atol=1e-6)
    for v in gt_a.values():
        assert v >= 0.0
        assert np.isfinite(v)


def test_gt_a_deterministic_under_fixed_seed():
    """Same config → identical decomposition (pure function of fixed-seed data)."""
    journeys, config = _journeys_with_converters()
    gt1 = compute_ground_truth_intensity(journeys, config)
    gt2 = compute_ground_truth_intensity(journeys, config)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(gt1[ch], gt2[ch], atol=1e-9)


# ============================================================
# 2. GT-A no-converter edge — all-zero (NOT normalized)
# ============================================================

def test_gt_a_no_converter_returns_all_zero_unnormalized():
    """All-False ``converted`` → total contribution 0 → all-zero dict (not 1/7)."""
    config = default_dgp_config(n_users=1)
    journeys = make_journeys([
        (1, "New", ["Display", "Paid Search"], [0.0, 10.0], False),
        (2, "Loyal", ["Email", "Direct"], [0.0, 5.0], False),
    ])
    gt_a = compute_ground_truth_intensity(journeys, config)

    assert set(gt_a.keys()) == set(CHANNEL_NAMES)
    # Source clamps to all-zero and returns WITHOUT normalizing.
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(gt_a[ch], 0.0, atol=1e-9)
    np.testing.assert_allclose(sum(gt_a.values()), 0.0, atol=1e-9)


# ============================================================
# 3. _decompose_user_intensity — channel coverage + non-negativity
# ============================================================

def test_decompose_user_intensity_covers_all_channels_nonneg():
    """Per-user decomposition returns a non-negative value for every channel."""
    config = default_dgp_config()
    seg = segment_by_name("Exploratory")  # eta = 0.0 (no heterogeneity term)
    contrib = _decompose_user_intensity(
        ["Display", "Paid Search"], [0.0, 10.0], 10.0, config, seg,
    )
    assert set(contrib.keys()) == set(CHANNEL_NAMES)
    for v in contrib.values():
        assert v >= 0.0


def test_decompose_beta_decay_recovers_channel_effect():
    """Single touchpoint, eta=0 → contribution == beta * decay (exact)."""
    config = default_dgp_config()
    seg = segment_by_name("Exploratory")  # eta = 0.0
    # Display: beta 0.3, half-life 14 days = 336 h. Δt = 0 → decay 1.0.
    contrib = _decompose_user_intensity(
        ["Display"], [0.0], 0.0, config, seg,
    )
    np.testing.assert_allclose(contrib["Display"], 0.3, atol=1e-9)
    # All other channels exactly zero.
    for ch in CHANNEL_NAMES:
        if ch != "Display":
            np.testing.assert_allclose(contrib[ch], 0.0, atol=1e-9)


# ============================================================
# 4. Cross-influence direction — source-before-target only
# ============================================================

def test_cross_influence_only_when_source_precedes_target():
    """Display->Paid Search (delta 0.4) bonus credited only when Display first.

    Robust invariant: with Display before Paid Search, the combined
    (Display + Paid Search) contribution strictly exceeds the reversed-order
    case, because the cross-influence delta is split between the two channels
    only in the forward order.
    """
    config = default_dgp_config()
    seg = segment_by_name("Exploratory")  # eta = 0.0 → isolate channel + synergy

    # Forward: Display (idx 0) then Paid Search (idx 1) → synergy fires.
    fwd = _decompose_user_intensity(
        ["Display", "Paid Search"], [0.0, 10.0], 10.0, config, seg,
    )
    # Reverse: Paid Search (idx 0) then Display (idx 1) → no Display->PS synergy.
    rev = _decompose_user_intensity(
        ["Paid Search", "Display"], [0.0, 10.0], 10.0, config, seg,
    )

    fwd_pair = fwd["Display"] + fwd["Paid Search"]
    rev_pair = rev["Display"] + rev["Paid Search"]
    # Forward order carries the extra (decayed) delta synergy mass.
    assert fwd_pair > rev_pair + 1e-9


def test_cross_influence_split_by_beta_ratio_exact():
    """Synergy mass is split src/tgt by beta ratio; total split == decayed delta.

    Reverse-order baseline removes the synergy, so the per-channel difference
    equals the decayed-delta split. Display beta 0.3, Paid Search beta 1.2,
    total 1.5. With both touchpoints at t=0 and obs_time=0, source decay = 1.0
    so decayed_delta == delta == 0.4.
    """
    config = default_dgp_config()
    seg = segment_by_name("Exploratory")  # eta = 0.0

    fwd = _decompose_user_intensity(
        ["Display", "Paid Search"], [0.0, 0.0], 0.0, config, seg,
    )
    rev = _decompose_user_intensity(
        ["Paid Search", "Display"], [0.0, 0.0], 0.0, config, seg,
    )

    delta = 0.4
    src_beta, tgt_beta = 0.3, 1.2
    total_beta = src_beta + tgt_beta
    # Forward adds delta * (src_beta/total) to source, delta*(tgt_beta/total) to target.
    np.testing.assert_allclose(
        fwd["Display"] - rev["Display"], delta * (src_beta / total_beta), atol=1e-9,
    )
    np.testing.assert_allclose(
        fwd["Paid Search"] - rev["Paid Search"], delta * (tgt_beta / total_beta), atol=1e-9,
    )
    # Total synergy mass == decayed delta.
    np.testing.assert_allclose(
        (fwd["Display"] - rev["Display"]) + (fwd["Paid Search"] - rev["Paid Search"]),
        delta, atol=1e-9,
    )


# ============================================================
# 5. GT-B counterfactual Shapley — normalization
# ============================================================

@pytest.mark.slow
def test_gt_b_shapley_sums_to_one():
    """compute_ground_truth_shapley → normalized credits over 7 channels, sum==1.0."""
    journeys, config = _journeys_with_converters()
    gt_b = compute_ground_truth_shapley(journeys, config, sample_users=200)

    assert set(gt_b.keys()) == set(CHANNEL_NAMES)
    np.testing.assert_allclose(sum(gt_b.values()), 1.0, atol=1e-6)
    for v in gt_b.values():
        assert v >= 0.0
        assert np.isfinite(v)


@pytest.mark.slow
def test_gt_b_shapley_deterministic():
    """Fixed config.random_seed → deterministic subsample → identical credits."""
    journeys, config = _journeys_with_converters()
    g1 = compute_ground_truth_shapley(journeys, config, sample_users=200)
    g2 = compute_ground_truth_shapley(journeys, config, sample_users=200)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(g1[ch], g2[ch], atol=1e-9)


# ============================================================
# 6. compute_all_ground_truths — combined package structure
# ============================================================

@pytest.mark.slow
def test_compute_all_ground_truths_structure_and_ranking():
    """GT-A and GT-B credits sum to 1.0; channel_ranking sorted descending."""
    journeys, config = _journeys_with_converters()
    result = compute_all_ground_truths(journeys, config, sample_users_shapley=200)

    for key in ("ground_truth_A", "ground_truth_B"):
        block = result[key]
        credits = block["channel_credits"]
        ranking = block["channel_ranking"]

        # Credits sum to 1.0 and cover all channels.
        assert set(credits.keys()) == set(CHANNEL_NAMES)
        np.testing.assert_allclose(sum(credits.values()), 1.0, atol=1e-6)

        # Ranking is a permutation of channels sorted descending by credit.
        assert set(ranking) == set(CHANNEL_NAMES)
        ranked_credits = [credits[ch] for ch in ranking]
        assert ranked_credits == sorted(ranked_credits, reverse=True)

    # Method tags wired through.
    assert result["ground_truth_A"]["method"] == "intensity_decomposition"
    assert result["ground_truth_B"]["method"] == "counterfactual_shapley"

    # Data statistics consistent with the input data.
    stats = result["data_statistics"]
    n_users_truth = journeys["user_id"].nunique()
    n_conv_truth = int(journeys.groupby("user_id")["converted"].first().sum())
    assert stats["n_users"] == n_users_truth
    assert stats["n_converters"] == n_conv_truth
    np.testing.assert_allclose(
        stats["conversion_rate"], n_conv_truth / n_users_truth, atol=1e-9,
    )

    # DGP parameters echoed faithfully.
    dgp = result["dgp_parameters"]
    np.testing.assert_allclose(dgp["alpha_0"], config.alpha_0, atol=1e-9)
    assert dgp["betas"]["Paid Search"] == 1.2
