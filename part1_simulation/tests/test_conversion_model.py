"""Unit tests for conversion_model.py (log-linear conversion intensity core).

Hand-computed checks of the integrated ground-truth conversion model:
    log(λ_i(t)) = α₀ + Σ βₖ·f_k(Δt) + Σ δ_ij·f_src(Δt)·I[src<tgt] + η_segment
    P(conversion) = 1 - exp(-exp(log_intensity))

Functions under test:
    compute_temporal_decay, compute_cross_influence_bonus, compute_log_intensity,
    intensity_to_conversion_prob, decide_conversion (+ vectorized form).
"""
from __future__ import annotations

import numpy as np

from part1_simulation.dgp.conversion_model import (
    compute_cross_influence_bonus,
    compute_log_intensity,
    compute_log_intensity_vectorized,
    compute_temporal_decay,
    decide_conversion,
    intensity_to_conversion_prob,
)
from part1_simulation.tests._journey_factory import (
    default_channels,
    default_cross_influences,
    default_dgp_config,
    segment_by_name,
)


# ============================================================
# 1. compute_temporal_decay — f(Δt) = exp(-Δt / (half_life_days × 24))
# ============================================================

def test_temporal_decay_half_life_14_days_one_day_gap():
    """14-day half-life, Δt=24h → exp(-24/336) == exp(-1/14)."""
    actual = compute_temporal_decay(14.0, 24.0)
    np.testing.assert_allclose(actual, np.exp(-24.0 / 336.0), rtol=1e-9)


def test_temporal_decay_one_day_half_life_one_day_gap():
    """1-day half-life, Δt=24h → exp(-24/24) == exp(-1) ≈ 0.3678794."""
    actual = compute_temporal_decay(1.0, 24.0)
    np.testing.assert_allclose(actual, np.exp(-1.0), rtol=1e-9)
    np.testing.assert_allclose(actual, 0.36787944117144233, rtol=1e-9)


def test_temporal_decay_zero_gap_is_one():
    """Δt=0 → decay factor is exactly 1.0 regardless of half-life."""
    np.testing.assert_allclose(compute_temporal_decay(7.0, 0.0), 1.0, atol=1e-9)


def test_temporal_decay_output_in_unit_interval():
    """f(Δt) ∈ (0, 1] for all non-negative gaps; monotonically decreasing in Δt."""
    gaps = [0.0, 1.0, 5.0, 24.0, 100.0, 1000.0]
    vals = [compute_temporal_decay(7.0, g) for g in gaps]
    for v in vals:
        assert 0.0 < v <= 1.0
    # strictly decreasing as the gap grows
    for a, b in zip(vals, vals[1:]):
        assert a > b


# ============================================================
# 2. intensity_to_conversion_prob — P = 1 - exp(-exp(log_intensity))
# ============================================================

def test_prob_at_zero_log_intensity():
    """log λ = 0 → λ = 1 → P = 1 - exp(-1) ≈ 0.6321206."""
    actual = intensity_to_conversion_prob(0.0)
    np.testing.assert_allclose(actual, 1.0 - np.exp(-1.0), rtol=1e-9)


def test_prob_at_negative_five():
    """log λ = -5 → P = 1 - exp(-exp(-5)) ≈ 0.0067153."""
    actual = intensity_to_conversion_prob(-5.0)
    np.testing.assert_allclose(actual, 1.0 - np.exp(-np.exp(-5.0)), rtol=1e-9)


def test_prob_clamped_at_ten():
    """Large log-intensity (50) is clamped to 10.0 → finite, ≈ 1.0, in [0, 1]."""
    actual = intensity_to_conversion_prob(50.0)
    # Clamp means the result equals the value at exactly 10.0
    expected = 1.0 - np.exp(-np.exp(10.0))
    np.testing.assert_allclose(actual, expected, rtol=1e-9)
    assert np.isfinite(actual)
    assert 0.0 <= actual <= 1.0
    # exp(exp(10)) underflows to ~1.0 conversion probability
    np.testing.assert_allclose(actual, 1.0, atol=1e-9)


def test_prob_clamp_equals_value_at_ten():
    """Inputs above the clamp threshold all collapse to the value at 10.0."""
    np.testing.assert_allclose(
        intensity_to_conversion_prob(50.0),
        intensity_to_conversion_prob(10.0),
        atol=1e-9,
    )


def test_prob_monotone_increasing_and_bounded():
    """P(conversion) is non-decreasing in log-intensity, bounded in [0, 1].

    Strictly increasing in the unsaturated regime; it saturates to exactly 1.0
    once exp(-exp(x)) underflows, so high inputs are only non-strictly ordered.
    """
    xs = [-20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    ps = [intensity_to_conversion_prob(x) for x in xs]
    for p in ps:
        assert 0.0 <= p <= 1.0
    # non-decreasing everywhere
    for a, b in zip(ps, ps[1:]):
        assert a <= b
    # strictly increasing in the unsaturated regime (x ≤ 1)
    unsat = [intensity_to_conversion_prob(x) for x in [-20.0, -10.0, -5.0, -1.0, 0.0, 1.0]]
    for a, b in zip(unsat, unsat[1:]):
        assert a < b


# ============================================================
# 3. compute_cross_influence_bonus — δ_ij · f_src(Δt) · I[src before tgt]
# ============================================================

def test_cross_influence_source_before_target():
    """Display→Paid Search active: 0.4 · f_Display(24h) = 0.4 · exp(-24/336)."""
    bonus = compute_cross_influence_bonus(
        journey_channels=["Display", "Paid Search"],
        journey_timestamps=[0.0, 10.0],
        observation_time=24.0,
        cross_influences=default_cross_influences(),
        channel_defs=default_channels(),
    )
    expected = 0.4 * np.exp(-24.0 / (14.0 * 24.0))
    np.testing.assert_allclose(bonus, expected, rtol=1e-9)


def test_cross_influence_target_absent():
    """No target channel in journey → no synergy → bonus 0.0."""
    bonus = compute_cross_influence_bonus(
        journey_channels=["Display"],
        journey_timestamps=[0.0],
        observation_time=24.0,
        cross_influences=default_cross_influences(),
        channel_defs=default_channels(),
    )
    np.testing.assert_allclose(bonus, 0.0, atol=1e-9)


def test_cross_influence_reversed_order_no_bonus():
    """Source after target (Paid Search before Display) → I[src<tgt]=0 → 0.0."""
    bonus = compute_cross_influence_bonus(
        journey_channels=["Paid Search", "Display"],
        journey_timestamps=[0.0, 10.0],
        observation_time=24.0,
        cross_influences=default_cross_influences(),
        channel_defs=default_channels(),
    )
    np.testing.assert_allclose(bonus, 0.0, atol=1e-9)


def test_cross_influence_empty_definitions():
    """Empty cross-influence tuple short-circuits to 0.0."""
    bonus = compute_cross_influence_bonus(
        journey_channels=["Display", "Paid Search"],
        journey_timestamps=[0.0, 10.0],
        observation_time=24.0,
        cross_influences=(),
        channel_defs=default_channels(),
    )
    np.testing.assert_allclose(bonus, 0.0, atol=1e-9)


def test_cross_influence_uses_source_first_occurrence_timestamp():
    """Decay applied at the source's FIRST-occurrence timestamp.

    Display first at t=0, repeated at t=20; Paid Search at t=10 (after first
    Display). Bonus uses the first Display timestamp (0.0), not the repeat.
    """
    bonus = compute_cross_influence_bonus(
        journey_channels=["Display", "Paid Search", "Display"],
        journey_timestamps=[0.0, 10.0, 20.0],
        observation_time=30.0,
        cross_influences=default_cross_influences(),
        channel_defs=default_channels(),
    )
    expected = 0.4 * np.exp(-30.0 / (14.0 * 24.0))  # decay from t_source=0
    np.testing.assert_allclose(bonus, expected, rtol=1e-9)


def test_cross_influence_multiple_pairs_additive():
    """Two active synergy pairs sum: Display→Paid Search and Social→Email."""
    channels = ["Display", "Social", "Paid Search", "Email"]
    timestamps = [0.0, 2.0, 10.0, 12.0]
    obs = 24.0
    bonus = compute_cross_influence_bonus(
        journey_channels=channels,
        journey_timestamps=timestamps,
        observation_time=obs,
        cross_influences=default_cross_influences(),
        channel_defs=default_channels(),
    )
    # Display (half-life 14d) source at t=0 ; Social (half-life 3d) source at t=2
    expected = (
        0.4 * np.exp(-(obs - 0.0) / (14.0 * 24.0))
        + 0.3 * np.exp(-(obs - 2.0) / (3.0 * 24.0))
    )
    np.testing.assert_allclose(bonus, expected, rtol=1e-9)


# ============================================================
# 4. compute_log_intensity — full additive log-scale assembly
# ============================================================

def test_log_intensity_single_display_at_zero_new_segment():
    """Single Display at t=0, obs=0, New segment → α₀ + β_Display + η_New.

    = -5.0 (α₀) + 0.3·1.0 (decay at Δt=0) + 0.0 (no cross) + (-0.3) (η_New)
    = -5.0 exactly.
    """
    config = default_dgp_config(alpha_0=-5.0)
    val = compute_log_intensity(
        touchpoint_channels=["Display"],
        touchpoint_timestamps=[0.0],
        observation_time=0.0,
        config=config,
        segment=segment_by_name("New"),
    )
    np.testing.assert_allclose(val, -5.0, atol=1e-9)


def test_log_intensity_components_additive():
    """log λ assembled from α₀ + channel-decay + cross + η (hand-computed).

    Display(t=0) then Paid Search(t=10), obs=24, Loyal segment (η=0.5).
    """
    config = default_dgp_config(alpha_0=-5.0)
    obs = 24.0
    val = compute_log_intensity(
        touchpoint_channels=["Display", "Paid Search"],
        touchpoint_timestamps=[0.0, 10.0],
        observation_time=obs,
        config=config,
        segment=segment_by_name("Loyal"),
    )
    # channel effects: β_Display·f_Display(24) + β_PaidSearch·f_PaidSearch(14)
    ch = (
        0.3 * np.exp(-(obs - 0.0) / (14.0 * 24.0))
        + 1.2 * np.exp(-(obs - 10.0) / (1.0 * 24.0))
    )
    cross = 0.4 * np.exp(-(obs - 0.0) / (14.0 * 24.0))  # Display→Paid Search
    eta = 0.5  # Loyal
    expected = -5.0 + ch + cross + eta
    np.testing.assert_allclose(val, expected, rtol=1e-9)


def test_log_intensity_segment_eta_shift():
    """Identical journeys differing only in segment shift by Δη exactly."""
    config = default_dgp_config(alpha_0=-5.0)
    kwargs = dict(
        touchpoint_channels=["Email"],
        touchpoint_timestamps=[3.0],
        observation_time=12.0,
        config=config,
    )
    v_new = compute_log_intensity(segment=segment_by_name("New"), **kwargs)
    v_loyal = compute_log_intensity(segment=segment_by_name("Loyal"), **kwargs)
    # η_Loyal - η_New = 0.5 - (-0.3) = 0.8
    np.testing.assert_allclose(v_loyal - v_new, 0.8, atol=1e-9)


def test_log_intensity_clips_negative_recency_to_zero():
    """A touchpoint after the observation time uses max(0, Δt)=0 → full decay 1.0."""
    config = default_dgp_config(alpha_0=-5.0)
    # Touchpoint at t=30 but observation at t=10 → negative recency clipped to 0.
    val = compute_log_intensity(
        touchpoint_channels=["Display"],
        touchpoint_timestamps=[30.0],
        observation_time=10.0,
        config=config,
        segment=segment_by_name("Exploratory"),  # η=0.0
    )
    # α₀ + β_Display·1.0 + 0 cross + 0 η = -5.0 + 0.3
    np.testing.assert_allclose(val, -5.0 + 0.3, atol=1e-9)


def test_log_intensity_alpha0_passthrough():
    """Empty channels list → log λ == α₀ + η only (no channel/cross terms)."""
    config = default_dgp_config(alpha_0=-2.5)
    val = compute_log_intensity(
        touchpoint_channels=[],
        touchpoint_timestamps=[],
        observation_time=5.0,
        config=config,
        segment=segment_by_name("Exploratory"),  # η=0.0
    )
    np.testing.assert_allclose(val, -2.5, atol=1e-9)


# ============================================================
# 5. compute_log_intensity_vectorized — matches scalar per-user
# ============================================================

def test_vectorized_matches_scalar():
    """Batch form returns the per-user scalar values elementwise."""
    config = default_dgp_config(alpha_0=-5.0)
    channels_list = [["Display"], ["Display", "Paid Search"], ["Email"]]
    timestamps_list = [[0.0], [0.0, 10.0], [3.0]]
    obs_times = [0.0, 24.0, 12.0]
    segments = [
        segment_by_name("New"),
        segment_by_name("Loyal"),
        segment_by_name("Exploratory"),
    ]
    vec = compute_log_intensity_vectorized(
        channels_list, timestamps_list, obs_times, config, segments
    )
    expected = np.array([
        compute_log_intensity(c, t, o, config, s)
        for c, t, o, s in zip(channels_list, timestamps_list, obs_times, segments)
    ])
    np.testing.assert_allclose(vec, expected, atol=1e-9)
    assert vec.shape == (3,)


# ============================================================
# 6. decide_conversion — deterministic under fixed RNG; matches threshold
# ============================================================

def test_decide_conversion_deterministic_under_seed():
    """Same seed → identical Bernoulli decision sequence."""
    seq_a = [decide_conversion(0.0, np.random.default_rng(123)) for _ in range(1)]
    seq_b = [decide_conversion(0.0, np.random.default_rng(123)) for _ in range(1)]
    assert seq_a == seq_b

    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    draws_a = [decide_conversion(-1.0, rng_a) for _ in range(50)]
    draws_b = [decide_conversion(-1.0, rng_b) for _ in range(50)]
    assert draws_a == draws_b


def test_decide_conversion_matches_threshold_rule():
    """decide_conversion(x, rng) == (rng.random() < P(x)) with synchronized RNG."""
    log_intensity = -1.5
    prob = intensity_to_conversion_prob(log_intensity)
    rng_decide = np.random.default_rng(2024)
    rng_manual = np.random.default_rng(2024)
    for _ in range(100):
        decided = decide_conversion(log_intensity, rng_decide)
        expected = bool(rng_manual.random() < prob)
        assert decided == expected


def test_decide_conversion_returns_python_bool():
    """Return type is a built-in bool (not numpy.bool_)."""
    out = decide_conversion(0.0, np.random.default_rng(0))
    assert isinstance(out, bool)


def test_decide_conversion_frequency_tracks_probability():
    """Empirical conversion rate at fixed log-intensity ≈ P(conversion)."""
    log_intensity = -0.5
    prob = intensity_to_conversion_prob(log_intensity)
    rng = np.random.default_rng(99)
    n = 20000
    hits = sum(decide_conversion(log_intensity, rng) for _ in range(n))
    rate = hits / n
    # Monte-Carlo: within a few standard errors of the true probability.
    se = np.sqrt(prob * (1.0 - prob) / n)
    assert abs(rate - prob) < 5.0 * se
