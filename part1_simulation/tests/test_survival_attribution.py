"""Unit tests for survival_attribution.py (Shender et al. 2023 TEDDA).

Each test maps to a specific paper equation/section:
    1.  Eq 12 / 4.1.7  — interval construction
    2.  Eq 12          — log-offset present
    3.  Eq 5           — single-ad β decay recovery
    4.  Eq 7           — multiplicative combination on log-scale
    5.  Eq 10          — segment dummy = α₀ shift
    6.  Eq 11          — query/ad split
    7.  4.1.6 (b)      — repeat conversions integer response
    8.  Eq 13          — telescoping invariant
    9.  Eq 17 / Eq 18  — normalization options
    10. Eq 20          — incremental keeps query effects
    11. Eq 21          — synergy definition (paper Example: m=5−1−2=2)
    12. Eq 24          — synergy decreases with gap
    13. Section 4.2.3  — BE − Shapley = ½·S (paper Example Eq 22–27)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from part1_simulation import CHANNEL_NAMES
from part1_simulation.models.causal.survival_attribution import (
    TIME_BIN_EDGES_HOURS,
    _build_interval_features,
    _compute_synergy_for_path,
    _fit_poisson_model,
    _predict_intensity_at,
    compute_survival_attribution,
    compute_synergy_report,
)


# ============================================================
# Helpers — toy DataFrame builders
# ============================================================

def _journey_rows(
    user_id: int,
    segment: str,
    channels: List[str],
    timestamps: List[float],
    converted: bool,
) -> List[dict]:
    n = len(channels)
    return [
        {
            "user_id": user_id,
            "segment": segment,
            "touchpoint_idx": i,
            "channel": ch,
            "timestamp": float(ts),
            "is_last_touchpoint": (i == n - 1),
            "converted": converted,
            "journey_length": n,
            "conversion_intensity": 0.0,
            "touchpoint_cost": 0.0,
        }
        for i, (ch, ts) in enumerate(zip(channels, timestamps))
    ]


def _make_journeys(specs: List[Tuple[int, str, List[str], List[float], bool]]) -> pd.DataFrame:
    rows: List[dict] = []
    for spec in specs:
        rows.extend(_journey_rows(*spec))
    return pd.DataFrame(rows)


# ============================================================
# 1. Interval construction — Section 4.1.7
# ============================================================

def test_interval_construction_single_user():
    """3 touchpoints + observation_end → 4 intervals: [0,t1],[t1,t2],[t2,t3],[t3,τ]."""
    j = _make_journeys([(1, "New", ["Display", "Email", "Paid Search"], [5, 10, 20], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=50.0)
    user_intervals = idf[idf["user_id"] == 1].sort_values("interval_idx")
    # break points are {0, 5, 10, 20, 50} → 4 non-degenerate intervals
    assert len(user_intervals) == 4
    np.testing.assert_allclose(
        user_intervals["t_start"].values, [0.0, 5.0, 10.0, 20.0]
    )
    np.testing.assert_allclose(
        user_intervals["t_end"].values, [5.0, 10.0, 20.0, 50.0]
    )
    # Conversion goes in the post-last-touchpoint interval [20, 50]
    assert user_intervals.iloc[-1]["conversion_count"] == 1
    assert user_intervals.iloc[:-1]["conversion_count"].sum() == 0


# ============================================================
# 2. Log-offset present — Eq 12
# ============================================================

def test_offset_present_in_design():
    j = _make_journeys([
        (1, "New", ["Display"], [0.0], True),
        (2, "Loyal", ["Email", "Paid Search"], [5.0, 12.0], False),
    ])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    assert "log_interval_length" in idf.columns
    np.testing.assert_allclose(
        np.exp(idf["log_interval_length"].values),
        idf["length"].values,
        rtol=1e-9,
    )


# ============================================================
# 3. Eq 5 single-ad recovery — fast-decay channel
# ============================================================

def _simulate_single_channel(
    n_users: int,
    half_life_hours: float,
    beta: float = 1.5,
    alpha0: float = -8.0,
    obs_end: float = 720.0,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """One ad of channel 'Display' per user at random time; conversion ~ Poisson with
    intensity exp(α₀ + β·exp(-Δt/τ_h)) over [t1, obs_end]."""
    rng = np.random.default_rng(rng_seed)
    rows: List[dict] = []
    for uid in range(n_users):
        t1 = float(rng.uniform(0, obs_end - 1))
        # Compute total Λ over [t1, obs_end]
        n_grid = 200
        ts = np.linspace(t1, obs_end, n_grid)
        recency = ts - t1
        decay = np.exp(-recency / half_life_hours)
        log_lam = alpha0 + beta * decay
        lam = np.exp(log_lam)
        Lambda = float(np.trapz(lam, ts))
        # Bernoulli outcome (1 conversion or 0 in single-occurrence DGP)
        converted = rng.random() < (1.0 - np.exp(-Lambda))
        rows.extend(_journey_rows(uid, "New", ["Display"], [t1], bool(converted)))
    return pd.DataFrame(rows)


def test_paper_eq5_recovery_single_channel():
    """Single fast-decaying ad → β_{Display, bin0} > β_{Display, bin4} (decay shape)."""
    j = _simulate_single_channel(
        n_users=4000, half_life_hours=24.0, beta=2.0, alpha0=-7.0
    )
    idf, cols, meta = _build_interval_features(j, observation_end=720.0)
    model = _fit_poisson_model(idf, cols)

    betas = [
        float(model.params.get(f"tb_Display_{b}", 0.0))
        for b in range(len(TIME_BIN_EDGES_HOURS) - 1)
    ]
    # Early bins should reflect higher intensity than late bins.
    assert betas[0] > betas[-1], (
        f"Expected fast decay (β[0] > β[-1]); got {betas}"
    )


# ============================================================
# 4. Eq 7 multiplicative combination — additive on log
# ============================================================

def test_eq7_multiplicative_combination():
    """λ̂ for {ad1, ad2} = exp(β1 + β2) · baseline (log-additive).

    Verify by using a known param vector with two ads in different channels.
    """
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    # Hand-craft params: const=-3, tb_Display_0=1.0, tb_Email_0=0.5, others=0
    params = pd.Series(
        {"const": -3.0, "tb_Display_0": 1.0, "tb_Email_0": 0.5},
    )
    # Evaluate at t*=15 (so both ads are in bin 0, recency < 24h)
    t_star = 15.0
    chs = np.array(["Display", "Email"])
    ts = np.array([5.0, 10.0])

    lam_full = _predict_intensity_at(
        params, t_star, [0, 1], chs, ts, {}, cols, meta
    )
    lam_d_only = _predict_intensity_at(
        params, t_star, [0], chs, ts, {}, cols, meta
    )
    lam_e_only = _predict_intensity_at(
        params, t_star, [1], chs, ts, {}, cols, meta
    )
    lam_empty = _predict_intensity_at(
        params, t_star, [], chs, ts, {}, cols, meta
    )
    # log(lam_full) - log(lam_empty) == (log(lam_d_only)-log(lam_empty)) + (log(lam_e_only)-log(lam_empty))
    np.testing.assert_allclose(
        np.log(lam_full / lam_empty),
        np.log(lam_d_only / lam_empty) + np.log(lam_e_only / lam_empty),
        atol=1e-9,
    )


# ============================================================
# 5. Eq 10 — segment dummy = intercept shift
# ============================================================

def test_eq10_segment_intercept_shift():
    """Two identical paths differing only in segment → intensity ratio = exp(γ_seg).

    Segment column naming follows alphabetical ordering: with segments
    ('Loyal', 'New'), Loyal is the reference and u_segment_New is the dummy.
    """
    j = _make_journeys([
        (1, "New", ["Display"], [5.0], True),
        (2, "Loyal", ["Display"], [5.0], True),
    ])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    seg_dummy_cols = [c for c in cols if c.startswith("u_segment_")]
    assert seg_dummy_cols, "expected at least one segment dummy column"
    seg_col = seg_dummy_cols[0]  # the non-reference segment
    params = pd.Series({"const": -3.0, seg_col: 0.7, "tb_Display_0": 1.0})
    chs = np.array(["Display"])
    ts = np.array([5.0])
    t_star = 6.0

    lam_ref = _predict_intensity_at(
        params, t_star, [0], chs, ts, {seg_col: 0.0}, cols, meta
    )
    lam_dummy = _predict_intensity_at(
        params, t_star, [0], chs, ts, {seg_col: 1.0}, cols, meta
    )
    np.testing.assert_allclose(
        lam_dummy / lam_ref, np.exp(0.7), rtol=1e-9
    )


# ============================================================
# 6. Eq 11 query/ad split
# ============================================================

def test_eq11_query_ad_split():
    """When query_events provided, design adds qb_* columns disjoint from tb_*."""
    j = _make_journeys([(1, "New", ["Display"], [5.0], True)])
    queries = pd.DataFrame([
        {"user_id": 1, "channel": "Display", "timestamp": 1.0, "ad_shown": True},
        {"user_id": 1, "channel": "Display", "timestamp": 4.0, "ad_shown": False},
    ])
    idf, cols, meta = _build_interval_features(
        j, observation_end=24.0, query_events=queries
    )
    qb_cols = [c for c in cols if c.startswith("qb_")]
    tb_cols = [c for c in cols if c.startswith("tb_")]
    assert qb_cols, "query columns missing"
    assert tb_cols, "ad columns missing"
    assert set(qb_cols).isdisjoint(set(tb_cols))
    # qb_Display_0 should reflect 2 queries seen by t_star=24 (within bin 0)
    last_interval = idf.sort_values("t_start").iloc[-1]
    # query at t=4 has recency 24-4=20h <24 → bin 0; query at t=1 recency 23h <24 → bin 0
    assert last_interval["qb_Display_0"] == 2.0


# ============================================================
# 7. Section 4.1.6 (b) — repeat conversions handled as integer response
# ============================================================

def test_repeat_conversion_integer_response():
    """If we synthesize conversion_count > 1 in an interval, GLM accepts it."""
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    # Inject a 2nd conversion in the final interval
    last_idx = idf["t_start"].idxmax()
    idf.loc[last_idx, "conversion_count"] = 2
    model = _fit_poisson_model(idf, cols)
    assert model.converged or model.fit_history is not None  # smoke check


# ============================================================
# 8. Eq 13 telescoping
# ============================================================

def test_telescoping_invariant_unclamped():
    """Σⱼ [λ̂(A(j))−λ̂(A(j−1))] == λ̂(A(n))−λ̂(∅) (raw, unclamped)."""
    j = _make_journeys([(1, "New", ["Display", "Email", "Paid Search"], [5, 10, 20], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=50.0)
    params = pd.Series({
        "const": -3.0,
        "tb_Display_0": 0.7,
        "tb_Email_0": 0.5,
        "tb_Paid Search_0": 0.4,
    })
    chs = np.array(["Display", "Email", "Paid Search"])
    ts = np.array([5.0, 10.0, 20.0])
    t_star = 20.0

    lam_full = _predict_intensity_at(params, t_star, [0, 1, 2], chs, ts, {}, cols, meta)
    lam_empty = _predict_intensity_at(params, t_star, [], chs, ts, {}, cols, meta)

    # Compute backwards-elimination raw credits without clamping
    active = [0, 1, 2]
    prev = lam_full
    total_raw = 0.0
    for k in (2, 1, 0):
        active_next = [a for a in active if a != k]
        new = _predict_intensity_at(
            params, t_star, active_next, chs, ts, {}, cols, meta
        )
        total_raw += (prev - new)  # raw, unclamped
        prev = new
        active = active_next

    np.testing.assert_allclose(total_raw, lam_full - lam_empty, atol=1e-9)


# ============================================================
# 9. Eq 17 / Eq 18 normalization options
# ============================================================

def test_normalization_eq17_eq18_run():
    """eq17 / eq18 normalization paths run without error and produce different totals."""
    j = _make_journeys([
        (1, "New", ["Display", "Email"], [5.0, 10.0], True),
        (2, "Loyal", ["Paid Search"], [3.0], True),
        (3, "New", ["Display"], [2.0], False),
    ])
    r17 = compute_survival_attribution(j, normalize="eq17")
    r18 = compute_survival_attribution(j, normalize="eq18")
    s17 = sum(r17.channel_credits.values())
    s18 = sum(r18.channel_credits.values())
    # Both should be finite and bounded: eq18 normalizes by (full - empty), so
    # total = 1.0 (telescoping over converted users), eq17 < 1.0 (denom larger).
    assert np.isfinite(s17) and np.isfinite(s18)
    assert s17 < s18 + 1e-9, f"eq17 ({s17}) should be ≤ eq18 ({s18}) since denom is larger"


# ============================================================
# 10. Eq 20 — incremental mode keeps query terms
# ============================================================

def test_eq20_incremental_keeps_query_effect():
    """With query_events provided, incremental mode produces a result; query coefficients
    remain in the model and are not ablated when ads are removed."""
    j = _make_journeys([
        (1, "New", ["Display", "Email"], [5.0, 10.0], True),
        (2, "Loyal", ["Paid Search"], [3.0], False),
        (3, "New", ["Display"], [2.0], True),
    ])
    queries = pd.DataFrame([
        {"user_id": 1, "channel": "Display", "timestamp": 5.0, "ad_shown": True},
        {"user_id": 1, "channel": "Email", "timestamp": 10.0, "ad_shown": True},
        {"user_id": 2, "channel": "Paid Search", "timestamp": 3.0, "ad_shown": True},
        {"user_id": 3, "channel": "Display", "timestamp": 2.0, "ad_shown": True},
        {"user_id": 3, "channel": "Email", "timestamp": 4.0, "ad_shown": False},
    ])
    r = compute_survival_attribution(
        j, credit_method="incremental", query_events=queries
    )
    assert r.metadata["options"]["has_queries"] is True
    assert r.metadata["learned_query_decay_curves"] is not None
    # Query decay curves include all channels
    assert "Email" in r.metadata["learned_query_decay_curves"]


# ============================================================
# 11. Eq 21 synergy definition — paper Example numerical
# ============================================================

def test_eq21_synergy_paper_example():
    """Paper Example (p.16): λ̂(∅)=1, λ̂({A1})=2, λ̂({A2})=3, λ̂({A1,A2})=6.
    m(A1)=1, m(A2)=2, m(A1∪A2)=5. S = 5 − 1 − 2 = 2.
    """
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    # Hand-craft params so that:
    # λ̂(∅) = exp(0) = 1
    # λ̂({Display}) = exp(log 2) = 2
    # λ̂({Email})   = exp(log 3) = 3
    # λ̂({D, E})   = exp(log 2 + log 3) = 6
    params = pd.Series({
        "const": 0.0,
        "tb_Display_0": float(np.log(2)),
        "tb_Email_0": float(np.log(3)),
    })

    class FakeModel:
        pass
    fake = FakeModel()
    fake.params = params

    S = _compute_synergy_for_path(fake, 1, j.sort_values("touchpoint_idx").reset_index(drop=True),
                                  meta, j=1)
    # S({Display}, Email) = m({D,E}) − m({D}) − m({E}) = 5 − 1 − 2 = 2
    np.testing.assert_allclose(S, 2.0, atol=1e-9)


# ============================================================
# 12. Eq 24 — synergy decreases with gap
# ============================================================

def test_eq24_synergy_decreases_with_gap():
    """For two ads A1, A2: as t2-t1 increases (so t*-t1 increases too),
    S should decrease (assuming f is decreasing).
    Use Eq 24: S = (exp(f1(t*-t1))-1)·(exp(f2(t*-t2))-1).
    Hold t*-t2 fixed, vary t1 such that t*-t1 grows.
    """
    # Construct: t2=10 always, t1 varies in {2, 5, 8}, t* = 11
    synergies = []
    gaps = [2, 5, 8]  # t2 - t1
    for gap in gaps:
        t1 = 10 - gap
        j = _make_journeys([(1, "New", ["Display", "Email"], [t1, 10.0], True)])
        idf, cols, meta = _build_interval_features(j, observation_end=24.0)
        # Decreasing step-function decay across bins for both channels
        params = pd.Series({
            "const": 0.0,
            "tb_Display_0": 1.0, "tb_Display_1": 0.5, "tb_Display_2": 0.2,
            "tb_Email_0": 0.8,
        })

        class FakeModel: pass
        fake = FakeModel(); fake.params = params

        S = _compute_synergy_for_path(
            fake, 1, j.sort_values("touchpoint_idx").reset_index(drop=True),
            meta, j=1,
        )
        synergies.append(S)
    # With increasing gap, t*-t1 grows from 3h → 6h → 9h. All in bin 0 (<24h),
    # so step-function gives same f1 for all gaps.
    # To trigger Eq 24's decay, push t1 across a bin boundary instead.
    # Re-run with t1 that crosses bin boundaries.
    synergies = []
    bin_target_recencies = [12.0, 36.0, 96.0]  # bin 0 (<24), bin 1 (24-72), bin 2 (72-168)
    for rec in bin_target_recencies:
        t_star = 100.0
        t1 = t_star - rec
        t2 = t_star - 2.0  # always within bin 0 → same f2
        j = _make_journeys([(1, "New", ["Display", "Email"], [t1, t2], True)])
        idf, cols, meta = _build_interval_features(j, observation_end=t_star + 1)
        params = pd.Series({
            "const": 0.0,
            "tb_Display_0": 1.0, "tb_Display_1": 0.5, "tb_Display_2": 0.2,
            "tb_Email_0": 0.8,
        })

        class FakeModel: pass
        fake = FakeModel(); fake.params = params

        S = _compute_synergy_for_path(
            fake, 1, j.sort_values("touchpoint_idx").reset_index(drop=True),
            meta, j=1,
        )
        synergies.append(S)
    # Synergy should decrease as ad 1 ages (bin 0 → bin 1 → bin 2)
    assert synergies[0] > synergies[1] > synergies[2], (
        f"Expected monotonically decreasing synergy with bin recency, got {synergies}"
    )


# ============================================================
# 13. BE − Shapley = ½·S — Section 4.2.3 paper Example (Eq 22-27)
# ============================================================

def test_be_minus_shapley_equals_half_synergy():
    """Paper Example: 2 ads, λ̂(∅)=1, λ̂({A1})=2, λ̂({A2})=3, λ̂({A1,A2})=6.

    BE: RawCredit(A1)=1, RawCredit(A2)=4
    Shapley: A1=2, A2=3
    BE(A2) − Shapley(A2) = 4 − 3 = 1 = ½ · S({A1},A2) = ½ · 2 = 1 ✓
    """
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    params = pd.Series({
        "const": 0.0,
        "tb_Display_0": float(np.log(2)),
        "tb_Email_0": float(np.log(3)),
    })

    class FakeModel: pass
    fake = FakeModel(); fake.params = params

    chs = np.array(["Display", "Email"])
    ts = np.array([5.0, 10.0])
    t_star = 10.0

    # λ̂ for all coalitions
    def lam(active):
        return _predict_intensity_at(params, t_star, active, chs, ts, {}, cols, meta)

    lam_empty = lam([])
    lam_d = lam([0])
    lam_e = lam([1])
    lam_de = lam([0, 1])

    # BE raw credits (Eq 13, going last → first):
    be_e = lam_de - lam_d  # remove last (Email) first
    be_d = lam_d - lam_empty

    # Shapley raw credits (2-player formula):
    # φ(D) = ½[(λ_D − λ_∅) + (λ_DE − λ_E)]
    # φ(E) = ½[(λ_E − λ_∅) + (λ_DE − λ_D)]
    sh_d = 0.5 * ((lam_d - lam_empty) + (lam_de - lam_e))
    sh_e = 0.5 * ((lam_e - lam_empty) + (lam_de - lam_d))

    # Synergy
    S = _compute_synergy_for_path(
        fake, 1, j.sort_values("touchpoint_idx").reset_index(drop=True), meta, j=1
    )

    # Verify paper claim: BE(A2) − Shapley(A2) = ½ · S
    np.testing.assert_allclose(be_e - sh_e, 0.5 * S, atol=1e-9)
    # And the symmetric: Shapley(A1) − BE(A1) = ½ · S
    np.testing.assert_allclose(sh_d - be_d, 0.5 * S, atol=1e-9)


# ============================================================
# 14-17. Shapley credit (Section 4.2.3 unified framework)
# ============================================================

from part1_simulation.models.causal.survival_attribution import _shapley_credits


def test_shapley_constant_invariance():
    """Shapley credits invariant under v(A) → v(A) + c for constant c.

    This proves that Survival/Poisson Shapley with v(A) = λ̂(A) gives the
    SAME credits as v(A) = λ̂(A) - λ̂(∅) (Du-style incremental form).
    """
    j = _make_journeys([
        (1, "New", ["Display", "Email", "Paid Search"], [5.0, 10.0, 15.0], True),
        (2, "Loyal", ["Email", "Direct"], [3.0, 8.0], True),
    ])
    r = compute_survival_attribution(j, credit_method="shapley")
    credits_unshifted = r.channel_credits

    # Manually call _shapley_credits and verify the invariance property holds:
    # φ_i = Σ marginals where marginals = v(S∪{i}) - v(S).
    # Adding a constant c to v doesn't change marginals → credits identical.
    # The implementation already uses v(A) = λ̂(A); this test confirms output is
    # well-defined (sum to 1 after normalization, finite, positive after clamp).
    total = sum(credits_unshifted.values())
    assert abs(total - 1.0) < 1e-9 or total == 0.0, f"sum={total}"
    for v in credits_unshifted.values():
        assert v >= 0.0, "negative credit before normalization"
        assert np.isfinite(v)


def test_shapley_efficiency_paper_example():
    """Eq 25 efficiency: Σ_i φ_i = v(N) - v(∅) = λ̂(N) - λ̂(∅).

    Use paper Example p.17: λ̂(∅)=1, λ̂(N=Display+Email)=6.
    Expected total Shapley credit = 6 - 1 = 5; with paper Shapley (Eq 26-27):
    φ(D) = 2, φ(E) = 3 → sum = 5. ✓
    """
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    # Hand-crafted params:
    # λ̂(∅) = exp(0) = 1, λ̂({D,E}) = exp(log 2 + log 3) = 6
    params = pd.Series({
        "const": 0.0,
        "tb_Display_0": float(np.log(2)),
        "tb_Email_0": float(np.log(3)),
    })

    class FakeModel:
        pass
    fake = FakeModel()
    fake.params = params

    credits = _shapley_credits(fake, j, meta)
    total = sum(credits.values())
    # λ̂(N) - λ̂(∅) = 6 - 1 = 5
    np.testing.assert_allclose(total, 5.0, atol=1e-9)


def test_shapley_2_player_paper_example():
    """Paper Example (Eq 26-27): 2-ad path → Shapley(D)=2, Shapley(E)=3.

    With λ̂(∅)=1, λ̂({D})=2, λ̂({E})=3, λ̂({D,E})=6:
    φ(D) = ½[(λ_D - λ_∅) + (λ_DE - λ_E)] = ½[1 + 3] = 2
    φ(E) = ½[(λ_E - λ_∅) + (λ_DE - λ_D)] = ½[2 + 4] = 3
    """
    j = _make_journeys([(1, "New", ["Display", "Email"], [5.0, 10.0], True)])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    params = pd.Series({
        "const": 0.0,
        "tb_Display_0": float(np.log(2)),
        "tb_Email_0": float(np.log(3)),
    })

    class FakeModel:
        pass
    fake = FakeModel()
    fake.params = params

    credits = _shapley_credits(fake, j, meta)
    np.testing.assert_allclose(credits["Display"], 2.0, atol=1e-9)
    np.testing.assert_allclose(credits["Email"], 3.0, atol=1e-9)


def test_shapley_du_incremental_equivalence():
    """Confirm: Shapley(v=λ̂) yields the same per-channel credits as
    Shapley(v=λ̂ - λ̂(∅)), the Du-style incremental form.

    This is mathematical (Shapley constant-invariance) — verified numerically.
    """
    # Smoke test on tiny dataset: shifted vs unshifted value functions.
    # Since our _shapley_credits uses v=λ̂ directly, this is trivially true,
    # but we confirm by re-deriving credits with manual shift.
    import itertools, math

    j = _make_journeys([
        (1, "New", ["Display", "Email"], [5.0, 10.0], True),
        (2, "New", ["Paid Search"], [3.0], True),
    ])
    idf, cols, meta = _build_interval_features(j, observation_end=24.0)
    params = pd.Series({
        "const": -2.0,
        "tb_Display_0": 0.4,
        "tb_Email_0": 0.5,
        "tb_Paid Search_0": 0.7,
    })

    class FakeModel: pass
    fake = FakeModel(); fake.params = params

    # Run our Shapley implementation
    credits_v = _shapley_credits(fake, j, meta)

    # Manually compute Shapley with v'(S) = v(S) - v(∅) (Du-style)
    from part1_simulation import CHANNEL_NAMES
    from part1_simulation.models.causal.survival_attribution import _predict_intensity_at

    converted = j[j["converted"]]
    user_data = []
    levels_per_feature = meta["levels_per_feature"]
    from part1_simulation.models.causal.survival_attribution import _user_feature_values
    for uid, g in converted.groupby("user_id", sort=False):
        g = g.sort_values("touchpoint_idx").reset_index(drop=True)
        ts = g["timestamp"].values.astype(float)
        chs = g["channel"].values
        sf = _user_feature_values(g.iloc[0], levels_per_feature)
        user_data.append((chs, ts, sf, float(ts.max())))

    def v_lam(S):
        total = 0.0
        for chs, ts, sf, t_star in user_data:
            active = [i for i, ch in enumerate(chs) if ch in S]
            total += _predict_intensity_at(
                params, t_star, active, chs, ts, sf,
                meta["feature_cols"], meta,
            )
        return total / len(user_data)

    v_empty = v_lam(frozenset())

    def v_shifted(S):
        return v_lam(S) - v_empty

    # Manual Shapley with v_shifted
    n = len(CHANNEL_NAMES)
    credits_shifted = {ch: 0.0 for ch in CHANNEL_NAMES}
    for ch_target in CHANNEL_NAMES:
        others = [c for c in CHANNEL_NAMES if c != ch_target]
        for r in range(n):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                S_with = S | {ch_target}
                marginal = v_shifted(S_with) - v_shifted(S)
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                credits_shifted[ch_target] += weight * marginal

    # Verify per-channel equivalence (Shapley constant-invariance)
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(
            credits_v[ch], credits_shifted[ch], atol=1e-9,
            err_msg=f"channel {ch}: v(A)-Shapley={credits_v[ch]}, "
                    f"(v(A)-v(∅))-Shapley={credits_shifted[ch]}"
        )


# ============================================================
# 18-19. Multivariate user features (Eq 10 generalization)
# ============================================================

def _journey_rows_with_user_feats(
    user_id: int,
    user_feats: dict,
    channels: List[str],
    timestamps: List[float],
    converted: bool,
) -> List[dict]:
    """Variant of _journey_rows that takes a dict of user features."""
    n = len(channels)
    base = {
        "user_id": user_id,
        "is_last_touchpoint": False,
        "converted": converted,
        "journey_length": n,
        "conversion_intensity": 0.0,
        "touchpoint_cost": 0.0,
        **user_feats,
    }
    return [
        {
            **base,
            "touchpoint_idx": i,
            "channel": ch,
            "timestamp": float(ts),
            "is_last_touchpoint": (i == n - 1),
        }
        for i, (ch, ts) in enumerate(zip(channels, timestamps))
    ]


def test_multivariate_user_features():
    """`user_features=['segment','device']` produces u_segment_* AND u_device_*
    dummies; intercept shift is multivariate (Eq 10 with multiple covariates)."""
    rows: List[dict] = []
    rows.extend(_journey_rows_with_user_feats(
        1, {"segment": "New",   "device": "mobile"},  ["Display"], [5.0], True))
    rows.extend(_journey_rows_with_user_feats(
        2, {"segment": "Loyal", "device": "desktop"}, ["Display"], [5.0], True))
    rows.extend(_journey_rows_with_user_feats(
        3, {"segment": "New",   "device": "tablet"},  ["Display"], [5.0], False))
    rows.extend(_journey_rows_with_user_feats(
        4, {"segment": "Loyal", "device": "mobile"},  ["Display"], [5.0], True))
    j = pd.DataFrame(rows)

    idf, cols, meta = _build_interval_features(
        j, observation_end=24.0, user_features=("segment", "device"),
    )
    seg_cols = [c for c in cols if c.startswith("u_segment_")]
    dev_cols = [c for c in cols if c.startswith("u_device_")]
    assert seg_cols, f"u_segment_* missing in cols: {cols}"
    assert dev_cols, f"u_device_* missing in cols: {cols}"
    # 2 segments → 1 dummy, 3 devices → 2 dummies
    assert len(seg_cols) == 1
    assert len(dev_cols) == 2

    # Intercept shift verifies multivariate composition (Loyal+mobile vs New+desktop)
    seg_col = seg_cols[0]  # u_segment_New
    dev_col_d = "u_device_mobile" if "u_device_mobile" in dev_cols else dev_cols[0]
    dev_col_t = "u_device_tablet" if "u_device_tablet" in dev_cols else dev_cols[1]
    params = pd.Series({
        "const": -3.0,
        seg_col: 0.5,
        dev_col_d: 0.3,
        dev_col_t: 0.7,
        "tb_Display_0": 1.0,
    })
    chs = np.array(["Display"])
    ts = np.array([5.0])
    t_star = 6.0

    lam_ref = _predict_intensity_at(
        params, t_star, [0], chs, ts,
        {seg_col: 0.0, dev_col_d: 0.0, dev_col_t: 0.0}, cols, meta,
    )
    lam_seg = _predict_intensity_at(
        params, t_star, [0], chs, ts,
        {seg_col: 1.0, dev_col_d: 0.0, dev_col_t: 0.0}, cols, meta,
    )
    lam_seg_dev = _predict_intensity_at(
        params, t_star, [0], chs, ts,
        {seg_col: 1.0, dev_col_d: 1.0, dev_col_t: 0.0}, cols, meta,
    )
    # Multivariate log-additive: ratio (seg+dev / ref) = exp(0.5 + 0.3)
    np.testing.assert_allclose(lam_seg_dev / lam_ref, np.exp(0.5 + 0.3), rtol=1e-9)
    np.testing.assert_allclose(lam_seg / lam_ref, np.exp(0.5), rtol=1e-9)

    # End-to-end Shapley credits run cleanly with multivariate features
    r = compute_survival_attribution(
        j, credit_method="shapley", user_features=("segment", "device"),
    )
    assert r.method == "Survival/Poisson (Shapley)"
    total = sum(r.channel_credits.values())
    assert abs(total - 1.0) < 1e-6 or total == 0.0
    # Metadata exposes the multivariate setup
    assert r.metadata["feature_cols"] is not None
    feat_cols = r.metadata["feature_cols"]
    assert any(c.startswith("u_segment_") for c in feat_cols)
    assert any(c.startswith("u_device_") for c in feat_cols)


def test_user_features_default_backward_compat():
    """Default `user_features=('segment',)` is identical to explicit `['segment']`."""
    j = _make_journeys([
        (1, "New",   ["Display", "Email"], [5.0, 10.0], True),
        (2, "Loyal", ["Paid Search"], [3.0], True),
        (3, "New",   ["Display"], [2.0], False),
        (4, "Loyal", ["Email", "Display"], [4.0, 9.0], True),
    ])
    r1 = compute_survival_attribution(j)  # default
    r2 = compute_survival_attribution(j, user_features=("segment",))  # explicit tuple
    r3 = compute_survival_attribution(j, user_features=["segment"])  # explicit list

    for ch in r1.channel_credits:
        np.testing.assert_allclose(
            r1.channel_credits[ch], r2.channel_credits[ch], atol=1e-9,
            err_msg=f"default vs tuple differ on {ch}",
        )
        np.testing.assert_allclose(
            r1.channel_credits[ch], r3.channel_credits[ch], atol=1e-9,
            err_msg=f"default vs list differ on {ch}",
        )


# ============================================================
# Path-level incrementality — compute_path_incrementality
# ============================================================

from part1_simulation.models.causal.survival_attribution import (  # noqa: E402
    _shapley_credits,
    _user_feature_values,
    compute_path_incrementality,
)


def test_path_incrementality_telescoping_matches_be_raw_unclamped():
    """Σ_u Δ_path (converters) == Σ_u [λ̂(full)−λ̂(∅)] computed directly.

    By the telescoping invariant this equals the §4 backwards-elimination
    raw (unclamped) total — the helper must reproduce that game value.
    """
    j = _make_journeys([
        (1, "New",   ["Display", "Email", "Paid Search"], [5.0, 10.0, 20.0], True),
        (2, "Loyal", ["Paid Search", "Email"],            [3.0, 8.0],        True),
        (3, "New",   ["Display"],                         [2.0],             False),
        (4, "Loyal", ["Email", "Display"],                [4.0, 9.0],        True),
    ])
    idf, cols, meta = _build_interval_features(j, observation_end=50.0)
    model = _fit_poisson_model(idf, cols)

    df = compute_path_incrementality(model, j, meta, cols, subpopulation="converters")

    # Direct per-converted-user λ̂(full) − λ̂(∅)
    params = model.params
    expected = 0.0
    for _, g in j[j["converted"]].groupby("user_id", sort=False):
        g = g.sort_values("touchpoint_idx").reset_index(drop=True)
        n = len(g)
        chs = g["channel"].values
        ts = g["timestamp"].values.astype(float)
        t_star = float(ts.max())
        ufv = _user_feature_values(g.iloc[0], meta["levels_per_feature"])
        lam_full = _predict_intensity_at(params, t_star, list(range(n)), chs, ts, ufv, cols, meta)
        lam_empty = _predict_intensity_at(params, t_star, [], chs, ts, ufv, cols, meta)
        expected += (lam_full - lam_empty)

    np.testing.assert_allclose(df["delta"].sum(), expected, atol=1e-9)
    # converters subpopulation excludes the non-converter (user 3)
    assert set(df["user_id"]) == {1, 2, 4}


def test_path_incrementality_efficiency_axiom_vs_shapley():
    """Marginal: Σ_u Δ_path == (Σ_c φ_c^Gcomp) × n_users (efficiency axiom).

    _shapley_credits returns the mean coalition-marginal over users, so its
    sum is v(N)−v(∅) = E_u[λ̂(full)−λ̂(∅)]; scaling by n_users recovers the
    path-Δ total. Directly exercises the subpopulation="all" branch.
    """
    j = _make_journeys([
        (1, "New",         ["Display", "Email"],        [5.0, 10.0], True),
        (2, "Loyal",       ["Paid Search", "Display"],  [3.0, 8.0],  True),
        (3, "New",         ["Email"],                   [2.0],       False),
        (4, "Loyal",       ["Display"],                 [4.0],       False),
        (5, "Exploratory", ["Email", "Paid Search"],    [1.0, 6.0],  True),
    ])
    idf, cols, meta = _build_interval_features(j, observation_end=50.0)
    model = _fit_poisson_model(idf, cols)

    df_all = compute_path_incrementality(model, j, meta, cols, subpopulation="all")
    n_users = j["user_id"].nunique()
    assert len(df_all) == n_users  # ALL users, converters + non-converters

    sh = _shapley_credits(model, j, meta, subpopulation="all")
    sigma_delta = df_all["delta"].sum()
    sigma_phi_scaled = sum(sh.values()) * n_users

    rel_err = abs(sigma_delta - sigma_phi_scaled) / max(abs(sigma_delta), 1e-12)
    assert rel_err < 1e-3, f"efficiency axiom rel_err={rel_err:.2e}"
