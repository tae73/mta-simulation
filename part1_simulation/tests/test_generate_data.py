"""Regression tests for dgp/generate_data.py — the end-to-end DGP pipeline.

Covers:
    - assign_timestamps        : per-user non-decreasing, first touchpoint t=0
    - generate_all_journeys    : schema completeness, structural invariants
    - validate_generated_data  : timestamp_violations == 0
    - calibrate_alpha_0        : convergence + bound on returned alpha_0
    - full pipeline (calibrate): conversion-rate range, right-skew, segment ordering

FAST tests use small n + calibrate=False. SLOW tests (full DGP @ n>=8000) are
marked @pytest.mark.slow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import JOURNEY_SCHEMA
from part1_simulation.dgp.generate_data import (
    assign_timestamps,
    calibrate_alpha_0,
    compute_conversions,
    generate_all_journeys,
    generate_channel_sequences,
    validate_generated_data,
)
from part1_simulation.dgp.user_segments import assign_segments
from part1_simulation.tests._journey_factory import default_dgp_config, make_journeys


# ============================================================
# assign_timestamps — per-user monotonicity + first touchpoint at t=0
# ============================================================

def test_assign_timestamps_first_touchpoint_is_zero():
    """The first touchpoint of every user is at timestamp 0.0."""
    j = make_journeys([
        (0, "New", ["Display", "Social", "Email"], [0.0, 0.0, 0.0], False),
        (1, "Loyal", ["Email", "Direct"], [0.0, 0.0], True),
        (2, "Exploratory", ["Organic Search"], [0.0], False),
    ])
    rng = np.random.default_rng(123)
    out = assign_timestamps(j, inter_arrival_lambda_hours=48.0, rng=rng)

    first_ts = out.groupby("user_id", sort=False)["timestamp"].first()
    np.testing.assert_allclose(first_ts.values, np.zeros(len(first_ts)), atol=1e-9)


def test_assign_timestamps_non_decreasing_within_user():
    """Within each user, timestamps are non-decreasing (cumsum of positive draws)."""
    j = make_journeys([
        (0, "New", ["Display", "Social", "Email", "Direct"], [0.0] * 4, False),
        (1, "Loyal", ["Email", "Direct", "Paid Search"], [0.0] * 3, True),
        (2, "Exploratory", ["Organic Search", "Referral"], [0.0, 0.0], False),
    ])
    rng = np.random.default_rng(7)
    out = assign_timestamps(j, inter_arrival_lambda_hours=48.0, rng=rng)

    diffs = out.groupby("user_id", sort=False)["timestamp"].diff().dropna()
    assert (diffs.values >= -1e-12).all(), f"found decreasing timestamp: min diff={diffs.min()}"


def test_assign_timestamps_deterministic_under_fixed_seed():
    """Fixed seed → identical timestamps across runs."""
    specs = [
        (0, "New", ["Display", "Social"], [0.0, 0.0], False),
        (1, "Loyal", ["Email", "Direct", "Paid Search"], [0.0, 0.0, 0.0], True),
    ]
    j1 = make_journeys(specs)
    j2 = make_journeys(specs)
    a = assign_timestamps(j1, 48.0, np.random.default_rng(2024))
    b = assign_timestamps(j2, 48.0, np.random.default_rng(2024))
    np.testing.assert_allclose(a["timestamp"].values, b["timestamp"].values, atol=1e-9)


# ============================================================
# generate_all_journeys (calibrate=False) — schema + structural invariants
# ============================================================

def _small_pipeline():
    """Run the fast (uncalibrated) pipeline on a small sample once."""
    config = default_dgp_config(n_users=500, alpha_0=-3.0)
    return generate_all_journeys(config, calibrate=False)


def test_pipeline_has_all_journey_schema_columns():
    """Output DataFrame contains every JOURNEY_SCHEMA column except cost layer
    (touchpoint_cost only appears when a budget_config is supplied)."""
    df, _stats = _small_pipeline()
    expected = set(JOURNEY_SCHEMA) - {"touchpoint_cost"}
    assert expected.issubset(set(df.columns)), (
        f"missing columns: {expected - set(df.columns)}"
    )


def test_pipeline_no_timestamp_violations():
    """validate_generated_data reports zero timestamp monotonicity violations."""
    df, _stats = _small_pipeline()
    config = default_dgp_config(n_users=500, alpha_0=-3.0)
    stats = validate_generated_data(df, config)
    assert stats["timestamp_violations"] == 0


def test_pipeline_stats_reports_zero_violations():
    """The stats dict returned by the pipeline also carries timestamp_violations == 0."""
    _df, stats = _small_pipeline()
    assert stats["timestamp_violations"] == 0


def test_pipeline_converted_constant_within_user():
    """`converted` is identical for all rows of a given user (per-user decision)."""
    df, _stats = _small_pipeline()
    nunique = df.groupby("user_id", sort=False)["converted"].nunique()
    assert (nunique == 1).all(), "converted flag varies within a user"


def test_pipeline_intensity_constant_within_user():
    """`conversion_intensity` is identical for all rows of a given user."""
    df, _stats = _small_pipeline()
    spread = df.groupby("user_id", sort=False)["conversion_intensity"].agg(
        lambda s: s.max() - s.min()
    )
    np.testing.assert_allclose(spread.values, np.zeros(len(spread)), atol=1e-9)


def test_pipeline_is_last_touchpoint_structure():
    """Exactly one last-touchpoint per user, located at touchpoint_idx == journey_length-1."""
    df, _stats = _small_pipeline()
    # exactly one True per user
    n_last = df.groupby("user_id", sort=False)["is_last_touchpoint"].sum()
    assert (n_last == 1).all(), "expected exactly one last touchpoint per user"
    # the True row is the max touchpoint_idx, which equals journey_length - 1
    last_rows = df[df["is_last_touchpoint"]]
    assert (last_rows["touchpoint_idx"] == last_rows["journey_length"] - 1).all()
    max_idx = df.groupby("user_id", sort=False)["touchpoint_idx"].transform("max")
    np.testing.assert_array_equal(
        df["is_last_touchpoint"].to_numpy(),
        (df["touchpoint_idx"] == max_idx).to_numpy(),
    )


def test_pipeline_touchpoint_idx_contiguous_per_user():
    """touchpoint_idx runs 0..journey_length-1 contiguously for each user."""
    df, _stats = _small_pipeline()
    for _uid, g in df.groupby("user_id", sort=False):
        jl = int(g["journey_length"].iloc[0])
        np.testing.assert_array_equal(
            np.sort(g["touchpoint_idx"].to_numpy()), np.arange(jl)
        )


def test_pipeline_calibrated_alpha_0_passthrough():
    """With calibrate=False the stats record the (unchanged) input alpha_0."""
    _df, stats = _small_pipeline()
    np.testing.assert_allclose(stats["calibrated_alpha_0"], -3.0, atol=1e-9)


def test_pipeline_deterministic_under_fixed_seed():
    """Same config + seed → byte-identical converted flags and timestamps."""
    cfg = default_dgp_config(n_users=400, alpha_0=-3.0, seed=99)
    d1, _ = generate_all_journeys(cfg, calibrate=False)
    d2, _ = generate_all_journeys(cfg, calibrate=False)
    np.testing.assert_array_equal(d1["converted"].to_numpy(), d2["converted"].to_numpy())
    np.testing.assert_allclose(d1["timestamp"].to_numpy(), d2["timestamp"].to_numpy(), atol=1e-9)


# ============================================================
# compute_conversions / validate_generated_data — direct invariants
# ============================================================

def test_compute_conversions_adds_expected_columns():
    """compute_conversions augments the frame with converted/intensity/is_last."""
    config = default_dgp_config(n_users=300, alpha_0=-3.0)
    rng = np.random.default_rng(config.random_seed)
    users = assign_segments(config.n_users, config.segments, config.max_touchpoints, rng)
    journeys = generate_channel_sequences(users, config, rng)
    journeys = assign_timestamps(journeys, config.inter_arrival_lambda_hours, rng)
    journeys = compute_conversions(journeys, config, rng)
    for col in ("converted", "conversion_intensity", "is_last_touchpoint"):
        assert col in journeys.columns
    assert journeys["converted"].dtype == bool
    assert journeys["is_last_touchpoint"].dtype == bool


def test_validate_stats_conversion_rate_consistency():
    """stats['conversion_rate'] equals n_converted / n_users."""
    df, _stats = _small_pipeline()
    config = default_dgp_config(n_users=500, alpha_0=-3.0)
    stats = validate_generated_data(df, config)
    np.testing.assert_allclose(
        stats["conversion_rate"],
        stats["n_converted"] / stats["n_users"],
        rtol=1e-9,
    )


# ============================================================
# SLOW: full calibrated pipeline — DGP validity properties
# ============================================================

@pytest.mark.slow
def test_full_calibrated_pipeline_dgp_properties():
    """Calibrated DGP at scale: conversion rate in target band, right-skewed
    journey lengths, and per-segment ordering Exploratory > Loyal > New."""
    config = default_dgp_config(n_users=8000)
    df, stats = generate_all_journeys(config, calibrate=True)

    # 1. Conversion rate within the documented [1.5%, 4%] band.
    assert 0.015 <= stats["conversion_rate"] <= 0.04, (
        f"conversion_rate={stats['conversion_rate']:.4f} outside [0.015, 0.04]"
    )

    # 2. Journey-length distribution is right-skewed.
    assert stats["journey_length_skew"] > 1.0, (
        f"journey_length_skew={stats['journey_length_skew']:.3f} not > 1.0"
    )

    # 3. Per-segment conversion-rate ordering: Exploratory > Loyal > New.
    by_seg = stats["conversion_rate_by_segment"]
    assert by_seg["Exploratory"] > by_seg["Loyal"] > by_seg["New"], (
        f"unexpected segment ordering: {by_seg}"
    )


@pytest.mark.slow
def test_calibrate_alpha_0_converges_and_in_bounds():
    """calibrate_alpha_0 returns a value within [-7.0, -4.0]."""
    config = default_dgp_config(n_users=5000)
    alpha_0 = calibrate_alpha_0(config)
    assert -7.0 <= alpha_0 <= -4.0, f"alpha_0={alpha_0:.4f} outside [-7.0, -4.0]"
