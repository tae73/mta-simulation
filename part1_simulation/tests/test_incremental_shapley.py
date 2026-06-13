"""Unit tests for incremental_shapley.py (Du et al. 2019 Incremental Shapley).

Two-step pipeline under test:
    1. Response model: LogisticRegression on user-level features (present/count/
       recency per channel + segment dummies + journey length/duration).
    2. Credit allocation: exact Shapley over the INCREMENTAL value function
       v(S) = max(0, P̂(conv | S active) − P̂(conv | no channels)) across the 7
       channels (2^7 = 128 coalitions), then clamp-negatives + normalize-to-1.

Sections:
    1.  Feature engineering — _build_user_features shapes/encodings
    2.  Exact Shapley — efficiency & symmetry axioms (_compute_exact_shapley)
    3.  Coalition prediction — masking semantics (_predict_coalition)
    4.  Public API on a tiny handcrafted dataset — output contract
    5.  Public API on a DGP-generated dataset — output contract + metadata (slow)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.models.causal.incremental_shapley import (
    _ABSENT_VALUES,
    _CHANNEL_FEATURE_PREFIXES,
    _build_user_features,
    _compute_exact_shapley,
    _get_feature_columns,
    _predict_coalition,
    _train_response_model,
    compute_incremental_shapley,
)
from part1_simulation.tests._journey_factory import (
    default_dgp_config,
    make_journeys,
)


# ============================================================
# Small deterministic dataset builder
# ============================================================

def _toy_mixed_journeys() -> pd.DataFrame:
    """40 users, 1-3 touches each, ~35% converters — enough to fit the model."""
    chans = list(CHANNEL_NAMES)
    rng = np.random.default_rng(0)
    specs: List[Tuple[int, str, List[str], List[float], bool]] = []
    segs = ("New", "Exploratory", "Loyal")
    for u in range(40):
        k = int(rng.integers(1, 4))
        chs = list(rng.choice(chans, size=k, replace=True))
        ts = sorted(float(x) for x in rng.uniform(0.0, 48.0, size=k))
        converted = bool(rng.random() < 0.4)
        seg = str(rng.choice(segs))
        specs.append((u, seg, chs, ts, converted))
    return make_journeys(specs)


# ============================================================
# 1. Feature engineering — _build_user_features
# ============================================================

def test_build_user_features_one_row_per_user():
    """One feature row per unique user; 'converted' column present."""
    j = make_journeys([
        (1, "New", ["Display", "Email"], [5.0, 10.0], True),
        (2, "Loyal", ["Paid Search"], [3.0], False),
        (3, "Exploratory", ["Organic Search", "Direct"], [1.0, 7.0], True),
    ])
    df, ch_map = _build_user_features(j)
    assert len(df) == j["user_id"].nunique() == 3
    assert "converted" in df.columns


def test_build_user_features_per_channel_columns_and_map():
    """Each channel contributes present_/count_/recency_ columns; the map indexes
    those three columns (relative to feature_cols) per channel."""
    j = _toy_mixed_journeys()
    df, ch_map = _build_user_features(j)
    feature_cols = _get_feature_columns(df)

    for ch in CHANNEL_NAMES:
        for prefix in _CHANNEL_FEATURE_PREFIXES:
            assert f"{prefix}_{ch}" in df.columns
        # map gives exactly the 3 feature-column indices for this channel
        idxs = ch_map[ch]
        assert len(idxs) == len(_CHANNEL_FEATURE_PREFIXES)
        recovered = [feature_cols[i] for i in idxs]
        assert recovered == [f"{p}_{ch}" for p in _CHANNEL_FEATURE_PREFIXES]


def test_build_user_features_present_count_recency_values():
    """Single user with 2 Display + 1 Email touches encodes present/count/recency
    exactly; absent channels get present=0, count=0, recency=1.0."""
    # Display at t=2,8 ; Email at t=10. last_ts=10, first_ts=2, duration=8.
    j = make_journeys([(1, "New", ["Display", "Display", "Email"], [2.0, 8.0, 10.0], True)])
    df, _ = _build_user_features(j)
    row = df.iloc[0]

    assert row["converted"] == 1.0
    assert row["journey_length"] == 3.0
    np.testing.assert_allclose(row["journey_duration_hours"], 8.0, atol=1e-9)

    # Display: present, 2 touches, last Display at t=8 → recency (10-8)/8 = 0.25
    assert row["present_Display"] == 1.0
    np.testing.assert_allclose(row["count_Display"], 2.0, atol=1e-9)
    np.testing.assert_allclose(row["recency_Display"], (10.0 - 8.0) / 8.0, rtol=1e-9)

    # Email: present, 1 touch, recency (10-10)/8 = 0
    assert row["present_Email"] == 1.0
    np.testing.assert_allclose(row["count_Email"], 1.0, atol=1e-9)
    np.testing.assert_allclose(row["recency_Email"], 0.0, atol=1e-9)

    # An absent channel → the "_ABSENT_VALUES" encoding (0, 0, 1.0)
    absent_present, absent_count, absent_recency = _ABSENT_VALUES
    np.testing.assert_allclose(row["present_Social"], absent_present, atol=1e-9)
    np.testing.assert_allclose(row["count_Social"], absent_count, atol=1e-9)
    np.testing.assert_allclose(row["recency_Social"], absent_recency, atol=1e-9)


def test_build_user_features_segment_dummies_drop_first():
    """Segment dummies are drop-first over the SORTED unique segments."""
    j = make_journeys([
        (1, "New", ["Display"], [1.0], True),
        (2, "Loyal", ["Email"], [2.0], False),
        (3, "Exploratory", ["Direct"], [3.0], True),
    ])
    df, _ = _build_user_features(j)
    # sorted(['New','Loyal','Exploratory']) = ['Exploratory','Loyal','New']; drop first
    seg_cols = sorted(c for c in df.columns if c.startswith("seg_"))
    assert seg_cols == ["seg_Loyal", "seg_New"]


# ============================================================
# 2. Exact Shapley — efficiency & symmetry axioms
# ============================================================

def test_exact_shapley_efficiency_additive_value():
    """Additive v(S)=Σ_i w_i → φ_i = w_i and Σφ = v(N) − v(∅) (efficiency)."""
    weights = {"A": 2.0, "B": 3.0, "C": 5.0}
    value_fn = lambda S: sum(weights[c] for c in S)
    sv = _compute_exact_shapley(("A", "B", "C"), value_fn)
    for ch, w in weights.items():
        np.testing.assert_allclose(sv[ch], w, atol=1e-9)
    np.testing.assert_allclose(
        sum(sv.values()),
        value_fn(frozenset({"A", "B", "C"})) - value_fn(frozenset()),
        atol=1e-9,
    )


def test_exact_shapley_symmetry_equal_players():
    """A symmetric value function splits credit equally between interchangeable
    channels (here: v(S)=1 if any present, else 0)."""
    value_fn = lambda S: float(len(S) >= 1)
    sv = _compute_exact_shapley(("A", "B"), value_fn)
    np.testing.assert_allclose(sv["A"], sv["B"], atol=1e-9)
    np.testing.assert_allclose(sv["A"], 0.5, atol=1e-9)


# ============================================================
# 3. Coalition prediction — masking semantics
# ============================================================

def test_predict_coalition_empty_vs_full_bounds():
    """_predict_coalition returns a probability in [0,1]; masking out all
    channels (empty coalition) zeroes the per-channel feature block."""
    j = _toy_mixed_journeys()
    df, ch_map = _build_user_features(j)
    model, scaler = _train_response_model(df)
    feature_cols = _get_feature_columns(df)
    X_raw = df[feature_cols].values.astype(np.float32)

    p_empty = _predict_coalition(model, scaler, X_raw, frozenset(), ch_map)
    p_full = _predict_coalition(model, scaler, X_raw, frozenset(CHANNEL_NAMES), ch_map)
    for p in (p_empty, p_full):
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0


def test_predict_coalition_does_not_mutate_input():
    """Masking happens on a copy; the caller's X_raw is untouched."""
    j = _toy_mixed_journeys()
    df, ch_map = _build_user_features(j)
    model, scaler = _train_response_model(df)
    feature_cols = _get_feature_columns(df)
    X_raw = df[feature_cols].values.astype(np.float32)
    X_before = X_raw.copy()

    _predict_coalition(model, scaler, X_raw, frozenset({"Display"}), ch_map)
    np.testing.assert_allclose(X_raw, X_before, atol=1e-9)


# ============================================================
# 4. Public API — tiny handcrafted dataset
# ============================================================

def test_api_returns_attribution_result_structure():
    """Output is an AttributionResult with the documented fields/labels."""
    j = _toy_mixed_journeys()
    r = compute_incremental_shapley(j)
    assert isinstance(r, AttributionResult)
    assert r.method == "Incremental Shapley"
    assert isinstance(r.channel_credits, dict)
    assert isinstance(r.channel_credits_raw, dict)
    assert isinstance(r.metadata, dict)
    # all 7 channels present in both credit dicts
    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
    assert set(r.channel_credits_raw.keys()) == set(CHANNEL_NAMES)


def test_api_credits_sum_to_one_nonneg_finite():
    """Normalized channel_credits sum to 1.0, are finite and non-negative."""
    j = _toy_mixed_journeys()
    r = compute_incremental_shapley(j)
    vals = np.array(list(r.channel_credits.values()), dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)
    np.testing.assert_allclose(vals.sum(), 1.0, atol=1e-6)


def test_api_normalization_equals_clamp_then_renormalize():
    """channel_credits == clamp(raw, 0)/Σclamp — the documented normalization."""
    j = _toy_mixed_journeys()
    r = compute_incremental_shapley(j)
    clamped = {k: max(0.0, v) for k, v in r.channel_credits_raw.items()}
    total = sum(clamped.values())
    if total > 0:
        expected = {k: v / total for k, v in clamped.items()}
    else:
        expected = {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r.channel_credits[ch], expected[ch], atol=1e-9,
            err_msg=f"normalization mismatch on {ch}",
        )


def test_api_metadata_fields_present_and_bounded():
    """Metadata carries base/full rates (probabilities) and coalition cache size."""
    j = _toy_mixed_journeys()
    r = compute_incremental_shapley(j)
    meta = r.metadata
    for key in (
        "base_conversion_rate",
        "full_coalition_rate",
        "incremental_fraction",
        "n_coalitions",
    ):
        assert key in meta
    assert 0.0 <= meta["base_conversion_rate"] <= 1.0
    assert 0.0 <= meta["full_coalition_rate"] <= 1.0
    # 7 channels → cache holds all 2^7 coalition values
    assert meta["n_coalitions"] == 2 ** len(CHANNEL_NAMES)


def test_api_deterministic_under_fixed_seed():
    """Same input + same random_seed → identical credits (RNG only drives
    the subsample, which here is a no-op since N < sample_users)."""
    j = _toy_mixed_journeys()
    r1 = compute_incremental_shapley(j, random_seed=42)
    r2 = compute_incremental_shapley(j, random_seed=42)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-12,
            err_msg=f"non-deterministic credit on {ch}",
        )


def test_api_raw_credits_can_be_negative_but_finite():
    """channel_credits_raw is the UN-clamped Shapley → may be negative, must be
    finite; the clamped/normalized credits stay non-negative."""
    j = _toy_mixed_journeys()
    r = compute_incremental_shapley(j)
    raw = np.array(list(r.channel_credits_raw.values()), dtype=float)
    assert np.all(np.isfinite(raw))
    # normalized credits never go below zero regardless of raw sign
    assert min(r.channel_credits.values()) >= 0.0


# ============================================================
# 5. Public API — DGP-generated dataset (end-to-end, slow)
# ============================================================

@pytest.mark.slow
def test_api_on_generated_journeys_contract():
    """Full DGP pipeline → incremental Shapley honors the output contract:
    7 channels, credits sum to 1.0, finite & non-negative, 128 coalitions."""
    cfg = default_dgp_config(n_users=800, alpha_0=-2.5)
    j, _ = generate_all_journeys(cfg, calibrate=False)
    # sanity: the generated frame must contain converters to fit a response model
    n_conv = int(j.groupby("user_id")["converted"].first().sum())
    assert n_conv > 0

    r = compute_incremental_shapley(j)
    assert isinstance(r, AttributionResult)
    assert set(r.channel_credits.keys()) == set(CHANNEL_NAMES)
    vals = np.array(list(r.channel_credits.values()), dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)
    np.testing.assert_allclose(vals.sum(), 1.0, atol=1e-6)
    assert r.metadata["n_coalitions"] == 2 ** len(CHANNEL_NAMES)
