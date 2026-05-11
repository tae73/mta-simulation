"""Credit assignment routines for Survival/Poisson attribution.

Section 4.2 of Shender et al. 2023 (TEDDA):
    - 4.2.1 Backwards Elimination (Eq 13): sequential ablation, raw credit
            via telescoping intensity drops.
    - 4.2.2 Incremental Attribution (Eq 19, 20): BE with query effects
            retained — separates query-driven from ad-driven incremental.
    - 4.2.3 Synergy (Eq 21, 24) and Shapley credit (Eq 25): exact Shapley
            on intensity backbone; equivalent to Du Incremental Shapley
            with Poisson response by constant-invariance.

Also includes:
    - `_aicpe_credits`: non-paper independent-removal extension.
    - `compute_synergy_report`: per-path synergy aggregation (Eq 21).
    - `_extract_learned_decay`: pull β_{c,b} step-function coefficients.

This module is internal — its functions are re-exported from
``survival_attribution`` for backward compatibility with tests.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from part1_simulation import CHANNEL_NAMES
from part1_simulation.models.causal._survival_features import (
    _N_BINS,
    _user_feature_values,
)
from part1_simulation.models.causal._survival_glm import (
    _GLMResult,
    _build_query_features,
    _predict_intensity_at,
)


# ============================================================
# Credit: Backwards Elimination (Section 4.2.1, Eq 13)
# ============================================================

def _backwards_elimination_credits(
    model: _GLMResult,
    journeys: pd.DataFrame,
    meta: Dict[str, Any],
    *,
    query_events: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Eq 13: RawCredit(j) = λ̂(t*, A(j)) - λ̂(t*, A(j-1)).

    Sum over converted users. Telescoping ensures Σⱼ RawCredit(j) = λ̂(A(n)) - λ̂(∅).
    """
    params = model.params
    feature_cols = meta["feature_cols"]
    levels_per_feature = meta["levels_per_feature"]

    channel_credits: Dict[str, float] = {ch: 0.0 for ch in CHANNEL_NAMES}

    if query_events is not None and len(query_events) > 0:
        query_by_user = {
            uid: g.sort_values("timestamp").reset_index(drop=True)
            for uid, g in query_events.groupby("user_id", sort=False)
        }
    else:
        query_by_user = {}

    for user_id, group in journeys[journeys["converted"]].groupby("user_id", sort=False):
        group = group.sort_values("touchpoint_idx").reset_index(drop=True)
        n = len(group)
        ts = group["timestamp"].values.astype(float)
        chs = group["channel"].values
        user_feat_values = _user_feature_values(group.iloc[0], levels_per_feature)

        t_star = float(ts.max())

        u_queries = query_by_user.get(user_id) if query_by_user else None
        q_feats = _build_query_features(t_star, u_queries, feature_cols)

        active = list(range(n))
        prev_lambda = _predict_intensity_at(
            params, t_star, active, chs, ts, user_feat_values, feature_cols, meta,
            query_features=q_feats,
        )

        for j in range(n - 1, -1, -1):
            active_next = active.copy()
            active_next.remove(j)
            new_lambda = _predict_intensity_at(
                params, t_star, active_next, chs, ts, user_feat_values, feature_cols, meta,
                query_features=q_feats,
            )
            channel_credits[chs[j]] += max(0.0, prev_lambda - new_lambda)
            prev_lambda = new_lambda
            active = active_next

    return channel_credits


# ============================================================
# Credit: Incremental (Section 4.2.2, Eq 19, 20)
# ============================================================

def _incremental_credits(
    model: _GLMResult,
    journeys: pd.DataFrame,
    meta: Dict[str, Any],
    query_events: pd.DataFrame,
) -> Dict[str, float]:
    """Eq 20: incremental BE — keep query effects, ablate ads only.

    For each converted user, compute Σⱼ [λ̂(A(j), Q) - λ̂(A(j-1), Q)] where Q is
    the FULL set of query events (always retained). Reduces to BE when query_events
    is None / empty.
    """
    return _backwards_elimination_credits(
        model, journeys, meta, query_events=query_events,
    )


# ============================================================
# Credit: AICPE (NON-PAPER extension)
# ============================================================

def _aicpe_credits(
    model: _GLMResult,
    interval_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Non-paper extension: independent channel removal averaged across intervals.

    For each channel, zero out its tb_* features in the design and average the
    drop in predicted intensity. NOT the paper's Eq 13 algorithm.
    """
    X_full = sm.add_constant(interval_df[feature_cols].astype(float), has_constant="add")
    offset = interval_df["log_interval_length"].values.astype(float)
    base_pred = model.predict(X_full, offset=offset)

    aicpe: Dict[str, float] = {}
    for ch in CHANNEL_NAMES:
        ch_cols = [c for c in feature_cols if c.startswith(f"tb_{ch}_")]
        X_removed = X_full.copy()
        for col in ch_cols:
            X_removed[col] = 0.0
        removed_pred = model.predict(X_removed, offset=offset)
        aicpe[ch] = float(np.mean(base_pred - removed_pred))

    return aicpe


# ============================================================
# Credit: Shapley (Shender Section 4.2.3, Eq 25)
# ============================================================

def _shapley_credits(
    model: _GLMResult,
    journeys: pd.DataFrame,
    meta: Dict[str, Any],
) -> Dict[str, float]:
    """Eq 25: Shapley credit on Survival/Poisson backbone (Section 4.2.3).

    Coalition value v(S) = E_user[λ̂(t*, A_user ∩ S)] over converted users.
    Enumerate 2^7 = 128 coalitions, exact Shapley formula.

    By Shapley constant-invariance, this is equivalent to using
    v'(S) = λ̂(S) - λ̂(∅) (Du-style incremental value function).
    Total credit distributed = λ̂(N) - λ̂(∅) per user, averaged.

    NOTE: This unifies Shender (intensity backbone) with Du
    (Shapley credit allocation). See Methodology_05 for derivation.
    """
    import itertools
    import math

    params = model.params
    feature_cols = meta["feature_cols"]
    levels_per_feature = meta["levels_per_feature"]
    channels = list(CHANNEL_NAMES)
    n = len(channels)

    # Pre-compute per-user metadata for converted users
    converted = journeys[journeys["converted"]]
    user_data: List[Tuple[Any, np.ndarray, np.ndarray, Dict[str, float], float]] = []
    for user_id, group in converted.groupby("user_id", sort=False):
        group = group.sort_values("touchpoint_idx").reset_index(drop=True)
        ts = group["timestamp"].values.astype(float)
        chs = group["channel"].values
        user_feat_values = _user_feature_values(group.iloc[0], levels_per_feature)
        t_star = float(ts.max())
        user_data.append((user_id, chs, ts, user_feat_values, t_star))

    if not user_data:
        return {ch: 0.0 for ch in CHANNEL_NAMES}

    # Coalition value cache: frozenset(channels) -> mean λ̂ across users
    cache: Dict[frozenset, float] = {}

    def coalition_value(S: frozenset) -> float:
        if S in cache:
            return cache[S]
        total = 0.0
        for _, chs, ts, user_feat_values, t_star in user_data:
            # Active ad indices: those whose channel is in S
            active = [i for i, ch in enumerate(chs) if ch in S]
            lam = _predict_intensity_at(
                params, t_star, active, chs, ts, user_feat_values,
                feature_cols, meta,
            )
            total += lam
        v = total / len(user_data)
        cache[S] = v
        return v

    # Exact Shapley over 128 coalitions
    shapley: Dict[str, float] = {ch: 0.0 for ch in channels}
    for ch_target in channels:
        others = [c for c in channels if c != ch_target]
        for r in range(n):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                S_with = S | {ch_target}
                marginal = coalition_value(S_with) - coalition_value(S)
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                shapley[ch_target] += weight * marginal

    return shapley


# ============================================================
# Synergy & Shapley comparison (Section 4.2.3, Eq 21, 24)
# ============================================================

def _compute_synergy_for_path(
    model: _GLMResult,
    user_id: Any,
    journeys_user: pd.DataFrame,
    meta: Dict[str, Any],
    j: int,
    query_events_user: Optional[pd.DataFrame] = None,
) -> float:
    """Eq 21: S(A(j-1), Aj) = m(A(j)) - m(A(j-1)) - m({Aj})
    where m(A) = λ̂(A) - λ̂(∅).
    """
    params = model.params
    feature_cols = meta["feature_cols"]
    levels_per_feature = meta["levels_per_feature"]

    n = len(journeys_user)
    if j < 0 or j >= n:
        return 0.0

    ts = journeys_user["timestamp"].values.astype(float)
    chs = journeys_user["channel"].values
    user_feat_values = _user_feature_values(journeys_user.iloc[0], levels_per_feature)
    t_star = float(ts.max())

    q_feats = _build_query_features(t_star, query_events_user, feature_cols)

    def lam(active: List[int]) -> float:
        return _predict_intensity_at(
            params, t_star, active, chs, ts, user_feat_values, feature_cols, meta,
            query_features=q_feats,
        )

    lam_empty = lam([])
    A_j = list(range(j + 1))
    A_jm1 = list(range(j))
    A_singleton = [j]

    m_Aj = lam(A_j) - lam_empty
    m_Ajm1 = lam(A_jm1) - lam_empty
    m_Aj_alone = lam(A_singleton) - lam_empty
    return m_Aj - m_Ajm1 - m_Aj_alone


def compute_synergy_report(
    journeys: pd.DataFrame,
    model: _GLMResult,
    meta: Dict[str, Any],
    *,
    gap_bin_edges_hours: Tuple[float, ...] = (0, 1, 6, 24, 72, 168, float("inf")),
) -> pd.DataFrame:
    """Synergy report aggregated by (channel_prev, channel_last, gap_bin).

    For each consecutive ad pair (j-1, j) in each user's path, compute
    S(A(j-1), Aj) per Eq 21 and aggregate.
    """
    rows: List[Dict[str, Any]] = []
    for user_id, group in journeys.groupby("user_id", sort=False):
        group = group.sort_values("touchpoint_idx").reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue
        ts = group["timestamp"].values.astype(float)
        chs = group["channel"].values

        for j in range(1, n):
            S = _compute_synergy_for_path(model, user_id, group, meta, j)
            gap = float(ts[j] - ts[j - 1])
            gap_bin = "inf"
            for b in range(len(gap_bin_edges_hours) - 1):
                if gap < gap_bin_edges_hours[b + 1]:
                    gap_bin = f"[{gap_bin_edges_hours[b]:.0f},{gap_bin_edges_hours[b+1]:.0f})"
                    break
            rows.append({
                "user_id": user_id,
                "channel_prev": chs[j - 1],
                "channel_last": chs[j],
                "gap_hours": gap,
                "gap_bin": gap_bin,
                "synergy": S,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["channel_prev", "channel_last", "gap_bin"], observed=True)
        .agg(mean_synergy=("synergy", "mean"), n_paths=("synergy", "size"))
        .reset_index()
    )


# ============================================================
# Learned decay extraction
# ============================================================

def _extract_learned_decay(
    model: _GLMResult, prefix: str = "tb"
) -> Dict[str, List[float]]:
    """Extract β_{c,b} as decay curves per channel (or per query if prefix='qb')."""
    return {
        ch: [float(model.params.get(f"{prefix}_{ch}_{b}", 0.0)) for b in range(_N_BINS)]
        for ch in CHANNEL_NAMES
    }
