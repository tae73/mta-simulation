"""Unit tests for evaluation/evaluate.py — the unified evaluation runner.

Covers:
    evaluate_method        — single AttributionResult → EvaluationResult,
                             whose mae/kendall_tau/rmse match metrics.py
                             functions computed independently on the same inputs.
    evaluate_all_methods   — list[AttributionResult] → comparison DataFrame
                             (sorted by MAE ascending, expected columns, empty OK).
    evaluate_budget_allocation — list[AttributionResult] → allocation DataFrame
                             (integration with optimize_budget).

The metrics themselves are exercised in test_metrics.py; here we assert that the
runner is a faithful pass-through / aggregator of those metric functions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES, EvaluationResult
from part1_simulation.evaluation.evaluate import (
    evaluate_all_methods,
    evaluate_budget_allocation,
    evaluate_method,
)
from part1_simulation.evaluation.metrics import (
    compute_all_metrics,
    compute_channel_bias,
    compute_kendall_tau,
    compute_mae,
    compute_rmse,
    compute_top_k_accuracy,
)
from part1_simulation.tests._journey_factory import default_budget_config


# ============================================================
# Builders — AttributionResults + ground truth
# ============================================================

def _make_result(method: str, credits: Dict[str, float]) -> AttributionResult:
    """Build an AttributionResult from a normalized credit dict (sum=1.0)."""
    return AttributionResult(
        method=method,
        channel_credits=dict(credits),
        channel_credits_raw=dict(credits),
        metadata={},
    )


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    return {ch: w / total for ch, w in weights.items()}


# A 7-channel ground truth summing to 1.0 (Paid Search dominant, mirrors DGP betas).
_GROUND_TRUTH: Dict[str, float] = _normalize({
    "Display": 0.3,
    "Social": 0.4,
    "Organic Search": 0.5,
    "Paid Search": 1.2,
    "Email": 0.8,
    "Referral": 0.5,
    "Direct": 0.7,
})

# Three contrasting predictions over the same 7 channels (each sums to 1.0).
# "good" — close to GT; "uniform" — equal split; "skewed" — concentrated on Display.
_PRED_GOOD: Dict[str, float] = _normalize({
    "Display": 0.32,
    "Social": 0.38,
    "Organic Search": 0.48,
    "Paid Search": 1.25,
    "Email": 0.78,
    "Referral": 0.52,
    "Direct": 0.67,
})
_PRED_UNIFORM: Dict[str, float] = {ch: 1.0 / len(CHANNEL_NAMES) for ch in CHANNEL_NAMES}
_PRED_SKEWED: Dict[str, float] = _normalize({
    "Display": 5.0,
    "Social": 0.1,
    "Organic Search": 0.1,
    "Paid Search": 0.1,
    "Email": 0.1,
    "Referral": 0.1,
    "Direct": 0.1,
})


# ============================================================
# 1. evaluate_method — return type + field pass-through
# ============================================================

def test_evaluate_method_returns_evaluation_result():
    """evaluate_method yields an EvaluationResult carrying the source method name."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    assert isinstance(ev, EvaluationResult)
    assert ev.method == "Good"


def test_evaluate_method_mae_matches_metrics():
    """ev.mae == compute_mae(credits, gt) computed independently."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    expected = compute_mae(_PRED_GOOD, _GROUND_TRUTH)
    np.testing.assert_allclose(ev.mae, expected, atol=1e-9)


def test_evaluate_method_rmse_matches_metrics():
    """ev.rmse == compute_rmse(credits, gt) computed independently."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    expected = compute_rmse(_PRED_GOOD, _GROUND_TRUTH)
    np.testing.assert_allclose(ev.rmse, expected, atol=1e-9)


def test_evaluate_method_kendall_tau_matches_metrics():
    """ev.kendall_tau == compute_kendall_tau(credits, gt) computed independently."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    expected = compute_kendall_tau(_PRED_GOOD, _GROUND_TRUTH)
    np.testing.assert_allclose(ev.kendall_tau, expected, atol=1e-9)


def test_evaluate_method_channel_bias_matches_metrics():
    """ev.channel_bias is the per-channel predicted-minus-truth dict."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    expected = compute_channel_bias(_PRED_GOOD, _GROUND_TRUTH)
    assert set(ev.channel_bias) == set(expected)
    for ch in expected:
        np.testing.assert_allclose(ev.channel_bias[ch], expected[ch], atol=1e-9)


def test_evaluate_method_perfect_prediction_zero_error():
    """When prediction == ground truth, MAE = RMSE = 0 and tau = 1."""
    result = _make_result("Perfect", _GROUND_TRUTH)
    ev = evaluate_method(result, _GROUND_TRUTH)
    np.testing.assert_allclose(ev.mae, 0.0, atol=1e-9)
    np.testing.assert_allclose(ev.rmse, 0.0, atol=1e-9)
    np.testing.assert_allclose(ev.kendall_tau, 1.0, atol=1e-9)
    for b in ev.channel_bias.values():
        np.testing.assert_allclose(b, 0.0, atol=1e-9)


def test_evaluate_method_bias_sums_to_zero_for_normalized_inputs():
    """Both pred and GT sum to 1 → Σ(pred-truth) = 0 across channels."""
    result = _make_result("Skewed", _PRED_SKEWED)
    ev = evaluate_method(result, _GROUND_TRUTH)
    np.testing.assert_allclose(sum(ev.channel_bias.values()), 0.0, atol=1e-9)


# ============================================================
# 2. evaluate_all_methods — DataFrame shape + ordering
# ============================================================

def test_evaluate_all_methods_returns_dataframe():
    results = [
        _make_result("Good", _PRED_GOOD),
        _make_result("Uniform", _PRED_UNIFORM),
        _make_result("Skewed", _PRED_SKEWED),
    ]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_evaluate_all_methods_has_expected_columns():
    """Main metric columns + one bias_{channel} column per ground-truth channel."""
    results = [_make_result("Good", _PRED_GOOD)]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    for col in ("method", "mae", "rmse", "kendall_tau", "top3_accuracy"):
        assert col in df.columns, f"missing column {col}"
    bias_cols = [c for c in df.columns if c.startswith("bias_")]
    assert len(bias_cols) == len(_GROUND_TRUTH)
    for ch in _GROUND_TRUTH:
        assert f"bias_{ch}" in df.columns


def test_evaluate_all_methods_sorted_by_mae_ascending():
    """Rows are ordered by MAE ascending; the best (lowest-MAE) method is first."""
    results = [
        _make_result("Skewed", _PRED_SKEWED),
        _make_result("Good", _PRED_GOOD),
        _make_result("Uniform", _PRED_UNIFORM),
    ]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    mae = df["mae"].values
    assert np.all(np.diff(mae) >= -1e-12), f"mae not ascending: {mae}"
    # "Good" is closest to GT → lowest MAE → first row.
    assert df.iloc[0]["method"] == "Good"
    # Index is reset after sort.
    assert list(df.index) == list(range(len(df)))


def test_evaluate_all_methods_column_order():
    """Column layout: method, mae, rmse, kendall_tau, top3_accuracy, then sorted bias_*."""
    results = [_make_result("Good", _PRED_GOOD)]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    main_cols = ["method", "mae", "rmse", "kendall_tau", "top3_accuracy"]
    assert list(df.columns[: len(main_cols)]) == main_cols
    bias_cols = list(df.columns[len(main_cols):])
    assert bias_cols == sorted(bias_cols)
    assert all(c.startswith("bias_") for c in bias_cols)


def test_evaluate_all_methods_values_match_compute_all_metrics():
    """Each row's metric values equal compute_all_metrics on that method's credits."""
    results = [
        _make_result("Good", _PRED_GOOD),
        _make_result("Uniform", _PRED_UNIFORM),
        _make_result("Skewed", _PRED_SKEWED),
    ]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    creds = {
        "Good": _PRED_GOOD,
        "Uniform": _PRED_UNIFORM,
        "Skewed": _PRED_SKEWED,
    }
    for _, row in df.iterrows():
        expected = compute_all_metrics(creds[row["method"]], _GROUND_TRUTH)
        np.testing.assert_allclose(row["mae"], expected["mae"], atol=1e-9)
        np.testing.assert_allclose(row["rmse"], expected["rmse"], atol=1e-9)
        np.testing.assert_allclose(row["kendall_tau"], expected["kendall_tau"], atol=1e-9)
        np.testing.assert_allclose(
            row["top3_accuracy"], expected["top3_accuracy"], atol=1e-9
        )
        for ch in _GROUND_TRUTH:
            np.testing.assert_allclose(
                row[f"bias_{ch}"], expected[f"bias_{ch}"], atol=1e-9
            )


def test_evaluate_all_methods_top3_accuracy_bounds():
    """top3_accuracy is a fraction in [0, 1]; perfect-rank method scores 1.0."""
    results = [
        _make_result("Perfect", _GROUND_TRUTH),
        _make_result("Skewed", _PRED_SKEWED),
    ]
    df = evaluate_all_methods(results, _GROUND_TRUTH)
    assert (df["top3_accuracy"] >= 0.0).all()
    assert (df["top3_accuracy"] <= 1.0).all()
    perfect_row = df[df["method"] == "Perfect"].iloc[0]
    expected = compute_top_k_accuracy(_GROUND_TRUTH, _GROUND_TRUTH, k=3)
    np.testing.assert_allclose(perfect_row["top3_accuracy"], expected, atol=1e-9)
    np.testing.assert_allclose(perfect_row["top3_accuracy"], 1.0, atol=1e-9)


def test_evaluate_all_methods_empty_list_returns_empty_dataframe():
    """Empty input → empty DataFrame with expected schema, no crash (spec requirement)."""
    df = evaluate_all_methods([], _GROUND_TRUTH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    # Schema preserved: main metric columns + one bias_ column per ground-truth channel.
    assert "method" in df.columns
    assert all(f"bias_{ch}" in df.columns for ch in _GROUND_TRUTH)


def test_evaluate_all_methods_single_method_consistent_with_evaluate_method():
    """One-method DataFrame row agrees with evaluate_method's scalar metrics."""
    result = _make_result("Good", _PRED_GOOD)
    ev = evaluate_method(result, _GROUND_TRUTH)
    df = evaluate_all_methods([result], _GROUND_TRUTH)
    row = df.iloc[0]
    np.testing.assert_allclose(row["mae"], ev.mae, atol=1e-9)
    np.testing.assert_allclose(row["rmse"], ev.rmse, atol=1e-9)
    np.testing.assert_allclose(row["kendall_tau"], ev.kendall_tau, atol=1e-9)


# ============================================================
# 3. evaluate_budget_allocation — integration with optimize_budget
# ============================================================

def test_evaluate_budget_allocation_dataframe_shape():
    """Returns one row per method, sorted by allocation_mae ascending, with the
    expected main + per-paid-channel columns."""
    budget_config = default_budget_config()
    # GT optimal allocation over the 4 paid channels (fractions sum to 1.0).
    gt_optimal = _normalize({
        "Display": 0.2,
        "Social": 0.2,
        "Paid Search": 0.4,
        "Email": 0.2,
    })
    results = [
        _make_result("Good", _PRED_GOOD),
        _make_result("Uniform", _PRED_UNIFORM),
        _make_result("Skewed", _PRED_SKEWED),
    ]
    df = evaluate_budget_allocation(
        results, budget_config, gt_optimal, total_conversions=1000
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    for col in ("method", "allocation_mae", "allocation_tau"):
        assert col in df.columns
    # alloc_ and gt_alloc_ columns for each paid channel
    for ch in gt_optimal:
        assert f"alloc_{ch}" in df.columns
        assert f"gt_alloc_{ch}" in df.columns
    # Sorted ascending by allocation_mae.
    mae = df["allocation_mae"].values
    assert np.all(np.diff(mae) >= -1e-12), f"allocation_mae not ascending: {mae}"
    assert list(df.index) == list(range(len(df)))


def test_evaluate_budget_allocation_gt_columns_constant():
    """gt_alloc_{ch} columns echo the ground-truth optimal regardless of method."""
    budget_config = default_budget_config()
    gt_optimal = _normalize({
        "Display": 0.2,
        "Social": 0.2,
        "Paid Search": 0.4,
        "Email": 0.2,
    })
    results = [
        _make_result("Good", _PRED_GOOD),
        _make_result("Uniform", _PRED_UNIFORM),
    ]
    df = evaluate_budget_allocation(
        results, budget_config, gt_optimal, total_conversions=1000
    )
    for ch in gt_optimal:
        col = df[f"gt_alloc_{ch}"].values
        np.testing.assert_allclose(col, gt_optimal[ch], atol=1e-9)
