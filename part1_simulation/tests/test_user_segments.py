"""Unit tests for dgp/user_segments.py (Du et al. 2019 user heterogeneity).

Covers the three public functions:
    - generate_journey_length(segment, n, max_touchpoints, rng)
    - sample_start_channels(segment, n, rng)
    - assign_segments(n_users, segments, max_touchpoints, rng)

Asserts the contracts that downstream DGP code depends on: segment-proportion
recovery, journey-length bounds, start-channel membership, determinism under a
fixed seed, and the proportions-sum-to-1 invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from part1_simulation import SegmentDef
from part1_simulation.dgp.user_segments import (
    assign_segments,
    generate_journey_length,
    sample_start_channels,
)
from part1_simulation.tests._journey_factory import (
    default_segments,
    segment_by_name,
)


# ============================================================
# assign_segments — end-to-end DataFrame contract
# ============================================================

def test_assign_segments_columns_and_row_count():
    """Returns one row per user with the documented columns."""
    df = assign_segments(5000, default_segments(), 20, np.random.default_rng(42))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5000
    for col in ("user_id", "segment", "journey_length", "start_channel"):
        assert col in df.columns


def test_assign_segments_proportions_recovered():
    """Empirical segment proportions match the configured 0.5/0.3/0.2 (atol=0.03)."""
    df = assign_segments(5000, default_segments(), 20, np.random.default_rng(42))
    counts = df["segment"].value_counts(normalize=True)
    np.testing.assert_allclose(counts["New"], 0.5, atol=0.03)
    np.testing.assert_allclose(counts["Exploratory"], 0.3, atol=0.03)
    np.testing.assert_allclose(counts["Loyal"], 0.2, atol=0.03)


def test_assign_segments_journey_length_bounds():
    """Every journey_length lies within [1, max_touchpoints]."""
    max_tp = 20
    df = assign_segments(5000, default_segments(), max_tp, np.random.default_rng(42))
    assert df["journey_length"].min() >= 1
    assert df["journey_length"].max() <= max_tp


def test_assign_segments_user_ids_unique_contiguous():
    """user_id is a contiguous 0..n-1 range (offset accumulates across segments)."""
    df = assign_segments(5000, default_segments(), 20, np.random.default_rng(42))
    assert sorted(df["user_id"].tolist()) == list(range(5000))


def test_assign_segments_start_channel_membership():
    """Each user's start_channel belongs to that user's segment start_channels."""
    df = assign_segments(5000, default_segments(), 20, np.random.default_rng(42))
    allowed = {seg.name: set(seg.start_channels) for seg in default_segments()}
    for seg_name, group in df.groupby("segment", observed=True):
        assert set(group["start_channel"]).issubset(allowed[seg_name])


def test_assign_segments_deterministic_under_seed():
    """Same seed → identical DataFrame."""
    df1 = assign_segments(2000, default_segments(), 20, np.random.default_rng(7))
    df2 = assign_segments(2000, default_segments(), 20, np.random.default_rng(7))
    pd.testing.assert_frame_equal(df1, df2)


def test_assign_segments_bad_proportions_raise():
    """Proportions not summing to 1.0 trip the AssertionError in assign_segments."""
    bad = (
        SegmentDef("New", 0.5, 0.25, 1, -0.3, ("Display", "Social")),
        SegmentDef("Loyal", 0.2, 0.5, 1, 0.5, ("Email", "Direct")),
    )  # 0.5 + 0.2 = 0.7 != 1.0
    with pytest.raises(AssertionError):
        assign_segments(100, bad, 20, np.random.default_rng(0))


# ============================================================
# generate_journey_length — Geometric(p) + offset, clipped
# ============================================================

def test_generate_journey_length_bounds_new_segment():
    """Large 'New' sample: all lengths within [1, max_touchpoints]; mean finite >= 1."""
    seg = segment_by_name("New")
    rng = np.random.default_rng(123)
    lengths = generate_journey_length(seg, 50_000, 20, rng)
    assert lengths.shape == (50_000,)
    assert lengths.min() >= 1
    assert lengths.max() <= 20
    mean = float(lengths.mean())
    assert np.isfinite(mean)
    assert mean >= 1.0


def test_generate_journey_length_reproducible():
    """Two generators seeded identically yield element-wise identical arrays."""
    seg = segment_by_name("New")
    a = generate_journey_length(seg, 10_000, 20, np.random.default_rng(99))
    b = generate_journey_length(seg, 10_000, 20, np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_generate_journey_length_respects_offset_minimum():
    """Geometric draw (>= 1) + offset → every length >= 1 + offset (when below the cap).

    'New' has geometric_offset=1, so the minimum length is 1 (draw) + 1 = 2,
    which is still <= max_touchpoints and therefore not clipped to a lower value.
    """
    seg = segment_by_name("New")
    lengths = generate_journey_length(seg, 50_000, 20, np.random.default_rng(5))
    assert lengths.min() >= 1 + seg.geometric_offset


def test_generate_journey_length_cap_dominates():
    """With max_touchpoints=1, every length is clipped down to exactly 1."""
    seg = segment_by_name("Exploratory")  # offset=2, would otherwise be >= 3
    lengths = generate_journey_length(seg, 1000, 1, np.random.default_rng(11))
    np.testing.assert_array_equal(lengths, np.ones(1000, dtype=lengths.dtype))


def test_generate_journey_length_returns_integer_array():
    """Lengths are an integer dtype (used directly as touchpoint counts)."""
    seg = segment_by_name("Loyal")
    lengths = generate_journey_length(seg, 100, 20, np.random.default_rng(3))
    assert np.issubdtype(lengths.dtype, np.integer)


# ============================================================
# sample_start_channels — uniform draw from segment's preferred list
# ============================================================

def test_sample_start_channels_membership_all_segments():
    """Every sampled channel is a member of the segment's start_channels."""
    rng = np.random.default_rng(2024)
    for seg in default_segments():
        sampled = sample_start_channels(seg, 5000, rng)
        assert sampled.shape == (5000,)
        assert set(sampled).issubset(set(seg.start_channels))


def test_sample_start_channels_single_channel_segment():
    """A segment with a single start_channel always returns that channel."""
    seg = segment_by_name("Exploratory")  # ("Organic Search",)
    sampled = sample_start_channels(seg, 1000, np.random.default_rng(1))
    assert set(sampled) == {"Organic Search"}


def test_sample_start_channels_covers_full_list():
    """A multi-channel segment, sampled enough, uses every listed channel."""
    seg = segment_by_name("New")  # ("Display", "Social")
    sampled = sample_start_channels(seg, 5000, np.random.default_rng(8))
    assert set(sampled) == set(seg.start_channels)


def test_sample_start_channels_reproducible():
    """Same seed → identical channel array."""
    seg = segment_by_name("Loyal")
    a = sample_start_channels(seg, 2000, np.random.default_rng(44))
    b = sample_start_channels(seg, 2000, np.random.default_rng(44))
    np.testing.assert_array_equal(a, b)
