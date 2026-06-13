"""Unit tests for evaluation/metrics.py.

Hand-computed values + edge cases for the attribution / budget metrics:
    compute_mae, compute_rmse, compute_kendall_tau, compute_channel_bias,
    compute_top_k_accuracy, compute_all_metrics, compute_allocation_mae,
    compute_allocation_kendall_tau, compute_channel_roas, compute_channel_cpa.

All tolerances follow the project convention:
    - atol=1e-9 for scalar equality
    - rtol=1e-9 for ratios / sqrt results
"""
from __future__ import annotations

import numpy as np
import pytest

from part1_simulation.evaluation.metrics import (
    compute_all_metrics,
    compute_allocation_kendall_tau,
    compute_allocation_mae,
    compute_channel_bias,
    compute_channel_cpa,
    compute_channel_roas,
    compute_kendall_tau,
    compute_mae,
    compute_rmse,
    compute_top_k_accuracy,
)


# ============================================================
# compute_mae
# ============================================================

def test_mae_hand_computed():
    """|.3-.2|+|.5-.6|+|.2-.2| = 0.2; /3 channels = 0.2/3."""
    pred = {"A": 0.3, "B": 0.5, "C": 0.2}
    truth = {"A": 0.2, "B": 0.6, "C": 0.2}
    np.testing.assert_allclose(compute_mae(pred, truth), 0.2 / 3, atol=1e-9)


def test_mae_identical_is_zero():
    d = {"A": 0.4, "B": 0.6}
    np.testing.assert_allclose(compute_mae(d, dict(d)), 0.0, atol=1e-9)


def test_mae_missing_predicted_key_treated_as_zero():
    """Channels iterate over truth.keys(); missing predicted key -> 0.0."""
    pred = {"A": 0.5}  # B absent
    truth = {"A": 0.5, "B": 0.5}
    # |0.5-0.5| + |0.0-0.5| = 0.5; /2 = 0.25
    np.testing.assert_allclose(compute_mae(pred, truth), 0.25, atol=1e-9)


def test_mae_ignores_extra_predicted_keys():
    """Keys in predicted but not in truth do not contribute."""
    pred = {"A": 0.5, "B": 0.5, "EXTRA": 99.0}
    truth = {"A": 0.5, "B": 0.5}
    np.testing.assert_allclose(compute_mae(pred, truth), 0.0, atol=1e-9)


def test_mae_empty_truth_is_nan():
    """Mean over zero channels is NaN (documented edge behavior)."""
    assert np.isnan(compute_mae({}, {}))


# ============================================================
# compute_rmse
# ============================================================

def test_rmse_hand_computed():
    """sq errors .01+.01+0 = .02; mean = .02/3; sqrt = sqrt(.02/3)."""
    pred = {"A": 0.3, "B": 0.5, "C": 0.2}
    truth = {"A": 0.2, "B": 0.6, "C": 0.2}
    np.testing.assert_allclose(
        compute_rmse(pred, truth), np.sqrt(0.02 / 3), rtol=1e-9
    )


def test_rmse_identical_is_zero():
    d = {"A": 0.4, "B": 0.6}
    np.testing.assert_allclose(compute_rmse(d, dict(d)), 0.0, atol=1e-9)


def test_rmse_ge_mae():
    """RMSE >= MAE always (Jensen / power-mean inequality)."""
    pred = {"A": 0.1, "B": 0.7, "C": 0.2}
    truth = {"A": 0.4, "B": 0.3, "C": 0.3}
    assert compute_rmse(pred, truth) >= compute_mae(pred, truth) - 1e-12


def test_rmse_empty_truth_is_nan():
    assert np.isnan(compute_rmse({}, {}))


# ============================================================
# compute_kendall_tau
# ============================================================

def test_kendall_identical_ranking_is_one():
    pred = {"A": 0.1, "B": 0.5, "C": 0.9}
    truth = {"A": 0.2, "B": 0.4, "C": 0.7}  # same ordering A<B<C
    np.testing.assert_allclose(compute_kendall_tau(pred, truth), 1.0, atol=1e-9)


def test_kendall_reversed_ranking_is_minus_one():
    pred = {"A": 0.0, "B": 1.0}
    truth = {"A": 1.0, "B": 0.0}
    np.testing.assert_allclose(compute_kendall_tau(pred, truth), -1.0, atol=1e-9)


def test_kendall_single_channel_nan_guard_returns_zero():
    """Single channel -> scipy returns NaN -> guarded to 0.0."""
    np.testing.assert_allclose(
        compute_kendall_tau({"A": 0.5}, {"A": 0.5}), 0.0, atol=1e-9
    )


def test_kendall_empty_returns_zero():
    """Empty inputs -> NaN from scipy -> guarded to 0.0."""
    np.testing.assert_allclose(compute_kendall_tau({}, {}), 0.0, atol=1e-9)


def test_kendall_always_in_unit_interval():
    rng = np.random.default_rng(7)
    chans = [f"c{i}" for i in range(7)]
    for _ in range(20):
        pred = {c: float(v) for c, v in zip(chans, rng.random(len(chans)))}
        truth = {c: float(v) for c, v in zip(chans, rng.random(len(chans)))}
        tau = compute_kendall_tau(pred, truth)
        assert -1.0 - 1e-12 <= tau <= 1.0 + 1e-12


# ============================================================
# compute_channel_bias
# ============================================================

def test_bias_hand_computed():
    pred = {"A": 0.3, "B": 0.5}
    truth = {"A": 0.2, "B": 0.6}
    bias = compute_channel_bias(pred, truth)
    np.testing.assert_allclose(bias["A"], 0.1, atol=1e-9)
    np.testing.assert_allclose(bias["B"], -0.1, atol=1e-9)


def test_bias_sums_to_zero_when_both_normalized():
    pred = {"A": 0.2, "B": 0.3, "C": 0.5}
    truth = {"A": 0.4, "B": 0.4, "C": 0.2}
    bias = compute_channel_bias(pred, truth)
    np.testing.assert_allclose(sum(bias.values()), 0.0, atol=1e-9)


def test_bias_missing_predicted_key_is_negative_truth():
    pred = {"A": 0.6}
    truth = {"A": 0.6, "B": 0.4}
    bias = compute_channel_bias(pred, truth)
    np.testing.assert_allclose(bias["B"], -0.4, atol=1e-9)


def test_bias_keys_match_truth():
    pred = {"A": 0.5, "EXTRA": 1.0}
    truth = {"A": 0.5, "B": 0.5}
    bias = compute_channel_bias(pred, truth)
    assert set(bias.keys()) == {"A", "B"}


# ============================================================
# compute_top_k_accuracy
# ============================================================

def test_top_k_perfect_overlap_is_one():
    pred = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.05}
    truth = {"A": 0.4, "B": 0.35, "C": 0.2, "D": 0.05}
    np.testing.assert_allclose(
        compute_top_k_accuracy(pred, truth, k=3), 1.0, atol=1e-9
    )


def test_top_k_no_overlap_is_zero():
    pred = {"A": 0.9, "B": 0.1, "C": 0.0, "D": 0.0}
    truth = {"A": 0.0, "B": 0.0, "C": 0.6, "D": 0.4}
    # truth top-1 = {C}, pred top-1 = {A} -> overlap 0
    np.testing.assert_allclose(
        compute_top_k_accuracy(pred, truth, k=1), 0.0, atol=1e-9
    )


def test_top_k_in_unit_interval():
    rng = np.random.default_rng(11)
    chans = [f"c{i}" for i in range(7)]
    for _ in range(20):
        pred = {c: float(v) for c, v in zip(chans, rng.random(len(chans)))}
        truth = {c: float(v) for c, v in zip(chans, rng.random(len(chans)))}
        acc = compute_top_k_accuracy(pred, truth, k=3)
        assert 0.0 <= acc <= 1.0


def test_top_k_denominator_is_k_even_if_fewer_channels():
    """Overlap is divided by k, not by min(k, n_channels)."""
    pred = {"A": 1.0, "B": 0.0}
    truth = {"A": 1.0, "B": 0.0}
    # both channels in top-3 -> overlap 2, /3
    np.testing.assert_allclose(
        compute_top_k_accuracy(pred, truth, k=3), 2.0 / 3.0, atol=1e-9
    )


# ============================================================
# compute_all_metrics
# ============================================================

def test_all_metrics_keys_and_values():
    pred = {"A": 0.3, "B": 0.5, "C": 0.2}
    truth = {"A": 0.2, "B": 0.6, "C": 0.2}
    res = compute_all_metrics(pred, truth)
    assert {"mae", "rmse", "kendall_tau", "top3_accuracy"} <= set(res.keys())
    assert {"bias_A", "bias_B", "bias_C"} <= set(res.keys())
    np.testing.assert_allclose(res["mae"], 0.2 / 3, atol=1e-9)
    np.testing.assert_allclose(res["rmse"], np.sqrt(0.02 / 3), rtol=1e-9)
    np.testing.assert_allclose(res["bias_A"], 0.1, atol=1e-9)
    np.testing.assert_allclose(res["bias_B"], -0.1, atol=1e-9)
    np.testing.assert_allclose(res["bias_C"], 0.0, atol=1e-9)


def test_all_metrics_consistent_with_individual_functions():
    pred = {"A": 0.1, "B": 0.7, "C": 0.2}
    truth = {"A": 0.4, "B": 0.3, "C": 0.3}
    res = compute_all_metrics(pred, truth)
    np.testing.assert_allclose(res["mae"], compute_mae(pred, truth), atol=1e-9)
    np.testing.assert_allclose(res["rmse"], compute_rmse(pred, truth), atol=1e-9)
    np.testing.assert_allclose(
        res["kendall_tau"], compute_kendall_tau(pred, truth), atol=1e-9
    )
    np.testing.assert_allclose(
        res["top3_accuracy"],
        compute_top_k_accuracy(pred, truth, k=3),
        atol=1e-9,
    )


# ============================================================
# compute_allocation_mae / compute_allocation_kendall_tau
# ============================================================

def test_allocation_mae_hand_computed():
    """Only truth channels compared; missing predicted key -> 0.0."""
    pred = {"Email": 0.5, "Display": 0.5}  # Social absent
    truth = {"Email": 0.4, "Display": 0.3, "Social": 0.3}
    # |0.5-0.4| + |0.5-0.3| + |0.0-0.3| = 0.1 + 0.2 + 0.3 = 0.6; /3 = 0.2
    np.testing.assert_allclose(compute_allocation_mae(pred, truth), 0.2, atol=1e-9)


def test_allocation_mae_extra_predicted_keys_ignored():
    pred = {"Email": 0.4, "Display": 0.6, "Organic Search": 99.0}
    truth = {"Email": 0.4, "Display": 0.6}
    np.testing.assert_allclose(compute_allocation_mae(pred, truth), 0.0, atol=1e-9)


def test_allocation_kendall_identical_ranking():
    pred = {"Email": 0.1, "Display": 0.3, "Social": 0.6}
    truth = {"Email": 0.2, "Display": 0.3, "Social": 0.5}  # same order
    np.testing.assert_allclose(
        compute_allocation_kendall_tau(pred, truth), 1.0, atol=1e-9
    )


def test_allocation_kendall_missing_key_zero_then_reversed():
    """Missing predicted key counts as 0.0 in the ranking."""
    # truth ranks: Display(0.6) > Email(0.4); pred: Email present, Display absent->0
    pred = {"Email": 0.4}
    truth = {"Display": 0.6, "Email": 0.4}
    # pred vals (sorted by truth keys = Display, Email): [0.0, 0.4]
    # truth vals: [0.6, 0.4] -> reversed ordering -> tau = -1
    np.testing.assert_allclose(
        compute_allocation_kendall_tau(pred, truth), -1.0, atol=1e-9
    )


def test_allocation_kendall_single_channel_guard():
    np.testing.assert_allclose(
        compute_allocation_kendall_tau({"Email": 1.0}, {"Email": 1.0}),
        0.0,
        atol=1e-9,
    )


# ============================================================
# compute_channel_roas
# ============================================================

def test_roas_hand_computed_skips_zero_cost():
    """ROAS = credit * total_conv * revenue / cost; zero-cost channel absent."""
    credits = {"Paid Search": 0.5, "Email": 0.5, "Direct": 0.0}
    costs = {"Paid Search": 100.0, "Email": 50.0, "Direct": 0.0}
    roas = compute_channel_roas(
        credits, costs, total_conversions=1000, revenue_per_conversion=100.0
    )
    # Paid Search: 0.5 * 1000 * 100 / 100 = 500
    # Email:       0.5 * 1000 * 100 / 50  = 1000
    np.testing.assert_allclose(roas["Paid Search"], 500.0, atol=1e-9)
    np.testing.assert_allclose(roas["Email"], 1000.0, atol=1e-9)
    assert "Direct" not in roas  # zero cost -> skipped


def test_roas_missing_cost_key_skipped():
    """Channel with no cost entry defaults to 0.0 cost -> skipped."""
    credits = {"Paid Search": 0.7, "Organic Search": 0.3}
    costs = {"Paid Search": 100.0}  # Organic Search absent
    roas = compute_channel_roas(credits, costs, 500, 100.0)
    assert "Organic Search" not in roas
    assert "Paid Search" in roas


def test_roas_zero_credit_is_zero_not_skipped():
    """A paid channel with zero credit yields ROAS 0.0 (still in dict)."""
    credits = {"Paid Search": 0.0}
    costs = {"Paid Search": 100.0}
    roas = compute_channel_roas(credits, costs, 1000, 100.0)
    np.testing.assert_allclose(roas["Paid Search"], 0.0, atol=1e-9)


def test_roas_empty_inputs():
    assert compute_channel_roas({}, {}, 1000, 100.0) == {}


# ============================================================
# compute_channel_cpa
# ============================================================

def test_cpa_hand_computed_skips_zero_cost():
    """CPA = cost / (credit * total_conv); zero-cost channel absent."""
    credits = {"Paid Search": 0.5, "Email": 0.5, "Direct": 0.0}
    costs = {"Paid Search": 100.0, "Email": 50.0, "Direct": 0.0}
    cpa = compute_channel_cpa(credits, costs, total_conversions=1000)
    # Paid Search: 100 / (0.5*1000) = 0.2
    # Email:       50  / (0.5*1000) = 0.1
    np.testing.assert_allclose(cpa["Paid Search"], 0.2, atol=1e-9)
    np.testing.assert_allclose(cpa["Email"], 0.1, atol=1e-9)
    assert "Direct" not in cpa


def test_cpa_zero_credit_guard_returns_zero():
    """credit 0 -> attributed_conv 0 -> guarded divide-by-zero -> 0.0."""
    cpa = compute_channel_cpa({"Paid Search": 0.0}, {"Paid Search": 100.0}, 1000)
    np.testing.assert_allclose(cpa["Paid Search"], 0.0, atol=1e-9)


def test_cpa_zero_total_conversions_guard():
    """total_conversions 0 -> attributed_conv 0 -> guarded -> 0.0."""
    cpa = compute_channel_cpa({"Paid Search": 0.5}, {"Paid Search": 100.0}, 0)
    np.testing.assert_allclose(cpa["Paid Search"], 0.0, atol=1e-9)


def test_cpa_missing_cost_key_skipped():
    credits = {"Paid Search": 0.7, "Organic Search": 0.3}
    costs = {"Paid Search": 100.0}
    cpa = compute_channel_cpa(credits, costs, 500)
    assert "Organic Search" not in cpa
    assert "Paid Search" in cpa


def test_cpa_empty_inputs():
    assert compute_channel_cpa({}, {}, 1000) == {}


# ============================================================
# Cross-check ROAS / CPA relationship
# ============================================================

def test_roas_cpa_reciprocal_relationship():
    """For a paid channel: ROAS = revenue_per_conv / CPA (when credit>0)."""
    credits = {"Paid Search": 0.5, "Email": 0.5}
    costs = {"Paid Search": 100.0, "Email": 50.0}
    rev = 100.0
    roas = compute_channel_roas(credits, costs, 1000, rev)
    cpa = compute_channel_cpa(credits, costs, 1000)
    for ch in ("Paid Search", "Email"):
        np.testing.assert_allclose(roas[ch], rev / cpa[ch], rtol=1e-9)
