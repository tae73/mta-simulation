"""Survival/Poisson-based attribution (Shender et al. 2023, TEDDA).

Implementation of "A Time To Event Framework For Multi-touch Attribution"
(JDS 2023, ArXiv 2009.08432). Section 4 methodology fully covered.

Section 4.1 — Modeling User Conversion Behavior:
    - 4.1.1 Step-function intensity (Eq 5): 5 piecewise-constant time bins
    - 4.1.2 Ad features (Eq 6): extra_ad_features hook
    - 4.1.3 Multiple ads (Eq 7) + position (Eq 8) + cross-ad (Eq 9):
            count features additive on log; include_position, include_cross_channel
    - 4.1.4 User features (Eq 10): segment dummies as α₀ shift
    - 4.1.5 Experimental data (Eq 11): query_events arg splits query/ad effects
    - 4.1.6 Refinements: include_seasonality (a), integer response (b),
            include_self_excitation (c)
    - 4.1.7 Estimation (Eq 12): interval split + Poisson regression with
            offset = log(interval_length), right-censoring via observation_end τ

Section 4.2 — Credit Assignment:
    - 4.2.1 Backwards Elimination (Eq 13): RawCredit(j) = λ̂(A(j)) - λ̂(A(j-1))
    - 4.2.2 Incremental Attribution (Eq 19, 20): mode="incremental" + query_events
    - 4.2.3 Synergy & Shapley (Eq 21, 24): _compute_synergy + compute_synergy_report
"""

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from part1_simulation import AttributionResult, CHANNEL_NAMES

_GLMResult = Any
logger = logging.getLogger(__name__)

# Piecewise-constant decay bins (hours of recency) — Eq 5 step-function
TIME_BIN_EDGES_HOURS: Tuple[float, ...] = (0, 24, 72, 168, 336, float("inf"))
_N_BINS = len(TIME_BIN_EDGES_HOURS) - 1


# ============================================================
# Helpers
# ============================================================

def _assign_bin(recency_hours: float) -> int:
    for b in range(_N_BINS):
        if recency_hours < TIME_BIN_EDGES_HOURS[b + 1]:
            return b
    return _N_BINS - 1


def _channel_bin_col_names(prefix: str) -> List[str]:
    return [f"{prefix}_{ch}_{b}" for ch in CHANNEL_NAMES for b in range(_N_BINS)]


def _user_feature_col_names(
    journeys: pd.DataFrame,
    user_features: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Generic Eq 10 user-feature dummies: one-hot with reference = first sorted level.

    Returns (column_names, levels_per_feature). For each user feature column in
    `user_features` (e.g. "segment", "device", "country"), produces one dummy
    column per non-reference level named ``u_{feature}_{level}``.
    """
    cols: List[str] = []
    levels_per_feature: Dict[str, List[str]] = {}
    for feat in user_features:
        levels = sorted(journeys[feat].astype(str).unique().tolist())
        levels_per_feature[feat] = levels
        for lvl in levels[1:]:  # drop reference level
            cols.append(f"u_{feat}_{lvl}")
    return cols, levels_per_feature


def _user_feature_values(
    row_like: Any,
    levels_per_feature: Dict[str, List[str]],
) -> Dict[str, float]:
    """Build {dummy_col: 0/1} for a single user from a row (Series/dict-like)."""
    feats: Dict[str, float] = {}
    for feat, levels in levels_per_feature.items():
        user_val = str(row_like[feat])
        for lvl in levels[1:]:
            feats[f"u_{feat}_{lvl}"] = 1.0 if user_val == lvl else 0.0
    return feats


def _seasonality_col_names() -> List[str]:
    return [f"hod_{h}" for h in range(1, 24)] + [f"dow_{d}" for d in range(1, 7)]


def _self_excitation_col_names() -> List[str]:
    return [f"prevconv_{b}" for b in range(_N_BINS)]


def _position_col_names() -> List[str]:
    return ["pos_first", "pos_last"]  # middle = reference level


def _cross_channel_col_names() -> List[str]:
    return [
        f"cross_{prev}_{curr}"
        for prev in CHANNEL_NAMES
        for curr in CHANNEL_NAMES
        if prev != curr
    ]


# ============================================================
# Feature Engineering — Interval Split (Section 4.1.7, Eq 12)
# ============================================================

def _build_interval_features(
    journeys: pd.DataFrame,
    *,
    observation_end: Optional[float] = None,
    query_events: Optional[pd.DataFrame] = None,
    include_position: bool = False,
    include_cross_channel: bool = False,
    include_seasonality: bool = False,
    include_self_excitation: bool = False,
    extra_ad_features: Optional[List[str]] = None,
    cross_channel_window_hours: float = 24.0,
    user_features: Tuple[str, ...] = ("segment",),
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Split each user path into piecewise-constant intervals (Eq 12).

    Break points per user: {0, t_1, ..., t_n, τ_user} (+ query times if provided).
    Each interval = one Poisson regression observation with offset = log(Δt).
    Conversion (1 per converted user) is placed in the final interval [t_n, τ].
    Right-censoring (Requirement #1) is naturally handled via the [t_n, τ] interval
    for non-converted users with conversion_count = 0.

    `user_features`: list of column names treated as user features (Eq 10).
    Each becomes one-hot dummies (reference = first sorted level). Default is
    `("segment",)` for backward compatibility.
    """
    if observation_end is None:
        observation_end = float(journeys["timestamp"].max() + 1.0)

    user_features = tuple(user_features)
    user_feature_cols, levels_per_feature = _user_feature_col_names(
        journeys, list(user_features)
    )

    tb_cols = _channel_bin_col_names("tb")
    feature_cols: List[str] = list(tb_cols) + list(user_feature_cols)

    if include_position:
        feature_cols += _position_col_names()
    if include_cross_channel:
        feature_cols += _cross_channel_col_names()
    if include_seasonality:
        feature_cols += _seasonality_col_names()
    if include_self_excitation:
        feature_cols += _self_excitation_col_names()

    extra_ad_features = list(extra_ad_features or [])
    extra_ad_cols: List[str] = []
    if extra_ad_features:
        # Eq 6: gₖ as step-function — bin × extra-feature value × channel
        # Simplest form: count of touchpoints per (extra_feature_value, bin)
        for feat_name in extra_ad_features:
            unique_vals = sorted(journeys[feat_name].dropna().unique().tolist())
            for val in unique_vals[1:]:  # drop reference level
                for b in range(_N_BINS):
                    extra_ad_cols.append(f"ag_{feat_name}_{val}_{b}")
        feature_cols += extra_ad_cols

    has_queries = query_events is not None and len(query_events) > 0
    query_cols: List[str] = []
    if has_queries:
        query_cols = _channel_bin_col_names("qb")
        feature_cols += query_cols

    if has_queries:
        query_by_user = {
            uid: g.sort_values("timestamp").reset_index(drop=True)
            for uid, g in query_events.groupby("user_id", sort=False)
        }
    else:
        query_by_user = {}

    rows: List[Dict[str, Any]] = []

    for user_id, group in journeys.groupby("user_id", sort=False):
        group = group.sort_values("touchpoint_idx").reset_index(drop=True)
        n = len(group)
        ts = group["timestamp"].values.astype(float)
        chs = group["channel"].values
        user_feat_values = _user_feature_values(group.iloc[0], levels_per_feature)
        converted = bool(group["converted"].iloc[0])

        u_queries = query_by_user.get(user_id) if has_queries else None
        if u_queries is not None and len(u_queries) > 0:
            q_ts = u_queries["timestamp"].values.astype(float)
            q_chs = u_queries["channel"].values
        else:
            q_ts = np.array([])
            q_chs = np.array([])

        # Build break points: 0, every ad time, every query time, τ
        bp_set = {0.0, observation_end}
        bp_set.update(ts.tolist())
        if len(q_ts) > 0:
            bp_set.update(q_ts.tolist())
        bp = sorted(bp_set)

        for i in range(len(bp) - 1):
            t_start = bp[i]
            t_end = bp[i + 1]
            length = t_end - t_start
            if length <= 0:
                continue

            row: Dict[str, Any] = {
                "user_id": user_id,
                "interval_idx": i,
                "t_start": t_start,
                "t_end": t_end,
                "length": length,
                "log_interval_length": float(np.log(length)),
                "conversion_count": 0,
            }

            # Channel-bin features (ads with t_j <= t_start)
            tb_feat = {col: 0.0 for col in tb_cols}
            active_ad_idx: List[int] = []
            for j in range(n):
                if ts[j] <= t_start + 1e-12:
                    recency = t_start - ts[j]
                    b = _assign_bin(max(0.0, recency))
                    tb_feat[f"tb_{chs[j]}_{b}"] += 1.0
                    active_ad_idx.append(j)
            row.update(tb_feat)
            row.update(user_feat_values)

            # Conversion in final interval [t_n, τ] for converted users
            if converted and n > 0 and t_start >= ts[-1] - 1e-9:
                row["conversion_count"] = 1

            if include_position:
                if active_ad_idx:
                    j_last = active_ad_idx[-1]
                    row["pos_first"] = 1.0 if j_last == 0 else 0.0
                    row["pos_last"] = 1.0 if j_last == n - 1 else 0.0
                else:
                    row["pos_first"] = 0.0
                    row["pos_last"] = 0.0

            if include_cross_channel:
                cross_feat = {col: 0.0 for col in _cross_channel_col_names()}
                for k in range(1, len(active_ad_idx)):
                    prev_j = active_ad_idx[k - 1]
                    curr_j = active_ad_idx[k]
                    if ts[curr_j] - ts[prev_j] <= cross_channel_window_hours:
                        if chs[prev_j] != chs[curr_j]:
                            cross_feat[f"cross_{chs[prev_j]}_{chs[curr_j]}"] += 1.0
                row.update(cross_feat)

            if include_seasonality:
                seas_feat = {col: 0.0 for col in _seasonality_col_names()}
                hod = int(t_start) % 24
                dow = (int(t_start) // 24) % 7
                if hod > 0:
                    seas_feat[f"hod_{hod}"] = 1.0
                if dow > 0:
                    seas_feat[f"dow_{dow}"] = 1.0
                row.update(seas_feat)

            if include_self_excitation:
                # Reference level (no prior conversions in DGP).
                row.update({col: 0.0 for col in _self_excitation_col_names()})

            if extra_ad_cols:
                ag_feat = {col: 0.0 for col in extra_ad_cols}
                for j in active_ad_idx:
                    recency = t_start - ts[j]
                    b = _assign_bin(max(0.0, recency))
                    for feat_name in extra_ad_features:
                        val = group[feat_name].iloc[j]
                        col = f"ag_{feat_name}_{val}_{b}"
                        if col in ag_feat:
                            ag_feat[col] += 1.0
                row.update(ag_feat)

            if has_queries and len(q_ts) > 0:
                q_feat = {col: 0.0 for col in query_cols}
                for k in range(len(q_ts)):
                    if q_ts[k] <= t_start + 1e-12:
                        recency = t_start - q_ts[k]
                        b = _assign_bin(max(0.0, recency))
                        q_feat[f"qb_{q_chs[k]}_{b}"] += 1.0
                row.update(q_feat)

            rows.append(row)

    interval_df = pd.DataFrame(rows)

    # Drop all-zero columns for stability (no information)
    nonzero_cols = [c for c in feature_cols if c in interval_df.columns and (interval_df[c] != 0).any()]
    feature_cols = nonzero_cols

    meta = {
        "user_features": list(user_features),
        "levels_per_feature": levels_per_feature,
        "user_feature_cols": [c for c in user_feature_cols if c in feature_cols],
        "observation_end": observation_end,
        "feature_cols": feature_cols,
        "tb_cols": [c for c in tb_cols if c in feature_cols],
        "query_cols": [c for c in query_cols if c in feature_cols],
        "extra_ad_cols": [c for c in extra_ad_cols if c in feature_cols],
        "extra_ad_features": extra_ad_features,
        "has_queries": has_queries,
        "include_position": include_position,
        "include_cross_channel": include_cross_channel,
        "include_seasonality": include_seasonality,
        "include_self_excitation": include_self_excitation,
        "cross_channel_window_hours": cross_channel_window_hours,
    }
    return interval_df, feature_cols, meta


# ============================================================
# Poisson GLM with offset (Eq 12)
# ============================================================

def _fit_poisson_model(
    interval_df: pd.DataFrame,
    feature_cols: List[str],
) -> _GLMResult:
    """Fit Poisson GLM: log(E[y]) = α₀ + Σ β·x + log(Δt) — Eq 12."""
    X = sm.add_constant(interval_df[feature_cols].astype(float), has_constant="add")
    y = interval_df["conversion_count"].values.astype(int)
    offset = interval_df["log_interval_length"].values.astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(
            y, X,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=offset,
        )
        return model.fit(disp=False, maxiter=200)


# ============================================================
# Intensity prediction at fixed observation time t*
# ============================================================

def _predict_intensity_at(
    params: pd.Series,
    t_star: float,
    active_ad_indices: List[int],
    all_channels: np.ndarray,
    all_timestamps: np.ndarray,
    user_feature_values: Dict[str, float],
    feature_cols: List[str],
    meta: Dict[str, Any],
    query_features: Optional[Dict[str, float]] = None,
    extra_ad_values: Optional[Dict[int, Dict[str, Any]]] = None,
) -> float:
    """Compute λ̂(t*, A) for ad subset A = active_ad_indices.

    `user_feature_values`: dict of {dummy_col: 0/1} for ALL user features
    (segment, device, country, ...), keyed by `u_{feature}_{level}`. Generic
    Eq 10 — any number of user features supported.
    """
    feat: Dict[str, float] = {col: 0.0 for col in feature_cols}

    n_total = len(all_channels)

    # Channel-bin features
    for j in active_ad_indices:
        recency = t_star - float(all_timestamps[j])
        if recency < 0:
            continue
        b = _assign_bin(max(0.0, recency))
        col = f"tb_{all_channels[j]}_{b}"
        if col in feat:
            feat[col] += 1.0

    # User-feature dummies (Eq 10) — supports multivariate
    for col, val in user_feature_values.items():
        if col in feat:
            feat[col] = val

    if meta.get("include_position", False):
        sorted_active = sorted(active_ad_indices, key=lambda j: float(all_timestamps[j]))
        if sorted_active:
            j_last = sorted_active[-1]
            if "pos_first" in feat:
                feat["pos_first"] = 1.0 if j_last == 0 else 0.0
            if "pos_last" in feat:
                feat["pos_last"] = 1.0 if j_last == n_total - 1 else 0.0

    if meta.get("include_cross_channel", False):
        sorted_active = sorted(active_ad_indices, key=lambda j: float(all_timestamps[j]))
        win = meta.get("cross_channel_window_hours", 24.0)
        for k in range(1, len(sorted_active)):
            prev_j = sorted_active[k - 1]
            curr_j = sorted_active[k]
            if float(all_timestamps[curr_j]) - float(all_timestamps[prev_j]) <= win:
                prev_ch = all_channels[prev_j]
                curr_ch = all_channels[curr_j]
                if prev_ch != curr_ch:
                    col = f"cross_{prev_ch}_{curr_ch}"
                    if col in feat:
                        feat[col] += 1.0

    if meta.get("include_seasonality", False):
        hod = int(t_star) % 24
        dow = (int(t_star) // 24) % 7
        if hod > 0 and f"hod_{hod}" in feat:
            feat[f"hod_{hod}"] = 1.0
        if dow > 0 and f"dow_{dow}" in feat:
            feat[f"dow_{dow}"] = 1.0

    # Self-excitation: reference level (no prior conversions in our DGP)

    if extra_ad_values and meta.get("extra_ad_features"):
        for j in active_ad_indices:
            if j not in extra_ad_values:
                continue
            recency = t_star - float(all_timestamps[j])
            b = _assign_bin(max(0.0, recency))
            for feat_name in meta["extra_ad_features"]:
                val = extra_ad_values[j].get(feat_name)
                col = f"ag_{feat_name}_{val}_{b}"
                if col in feat:
                    feat[col] += 1.0

    if query_features:
        for col, val in query_features.items():
            if col in feat:
                feat[col] = val

    # Linear predictor
    lp = float(params.get("const", 0.0))
    for col in feature_cols:
        coef = params.get(col, 0.0)
        if coef != 0.0 and feat[col] != 0.0:
            lp += float(coef) * feat[col]

    return float(np.exp(min(lp, 10.0)))


def _build_query_features(
    t_star: float,
    user_queries: Optional[pd.DataFrame],
    feature_cols: List[str],
) -> Dict[str, float]:
    """Build query-bin features (qb_*) at t_star — for Eq 11 / Eq 20 incremental."""
    q_feat: Dict[str, float] = {}
    qb_cols = [c for c in feature_cols if c.startswith("qb_")]
    for col in qb_cols:
        q_feat[col] = 0.0
    if user_queries is None or len(user_queries) == 0:
        return q_feat
    for _, q in user_queries.iterrows():
        qt = float(q["timestamp"])
        if qt > t_star + 1e-12:
            continue
        recency = t_star - qt
        b = _assign_bin(max(0.0, recency))
        col = f"qb_{q['channel']}_{b}"
        if col in q_feat:
            q_feat[col] += 1.0
    return q_feat


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


# ============================================================
# Public API
# ============================================================

def compute_survival_attribution(
    journeys: pd.DataFrame,
    credit_method: Literal["backelim", "aicpe", "incremental", "shapley"] = "backelim",
    *,
    query_events: Optional[pd.DataFrame] = None,
    observation_end: Optional[float] = None,
    include_position: bool = False,
    include_cross_channel: bool = False,
    include_seasonality: bool = False,
    include_self_excitation: bool = False,
    extra_ad_features: Optional[List[str]] = None,
    cross_channel_window_hours: float = 24.0,
    normalize: Literal["sum_to_one", "eq17", "eq18"] = "sum_to_one",
    user_features: Tuple[str, ...] = ("segment",),
) -> AttributionResult:
    """Survival/Poisson attribution — Shender et al. 2023 TEDDA.

    Args:
        journeys: long-format journey DataFrame (per JOURNEY_SCHEMA).
        credit_method:
            - "backelim" (Eq 13, default, paper primary): sequential ablation, synergy → last ad
            - "shapley" (Eq 25, Section 4.2.3): exact Shapley on intensity, synergy split equally;
              equivalent to Du Incremental Shapley with Poisson response (constant-invariance)
            - "incremental" (Eq 20): query/ad split incremental — requires query_events
            - "aicpe" (non-paper extension): independent channel removal averaged
        query_events: optional DataFrame (user_id, channel, timestamp[, ad_shown])
            for Eq 11 query/ad split. When provided + credit_method="incremental",
            yields Eq 20 incremental attribution.
        observation_end: right-censoring time τ; defaults to max timestamp + 1h.
        include_position: add Eq 8 position dummies (first/last).
        include_cross_channel: add Eq 9 cross-channel interaction indicators.
        include_seasonality: add 4.1.6(a) hour-of-day / day-of-week dummies.
        include_self_excitation: add 4.1.6(c) prior-conversion-recency bin (no-op
            in single-conversion DGP — feature stays at reference).
        extra_ad_features: list of additional ad-level columns for Eq 6 gₖ terms.
        normalize: "sum_to_one" (default), "eq17" (Eq 17), or "eq18" (Eq 18).
        user_features: tuple of column names treated as user features (Eq 10).
            Default is ``("segment",)`` (backward-compat). Pass any subset of
            user-level columns (e.g. ``("segment", "device", "country")``);
            each becomes one-hot dummies (reference = first sorted level).
    """
    logger.info("  Building interval features (Section 4.1.7, Eq 12)...")
    interval_df, feature_cols, meta = _build_interval_features(
        journeys,
        observation_end=observation_end,
        query_events=query_events,
        include_position=include_position,
        include_cross_channel=include_cross_channel,
        include_seasonality=include_seasonality,
        include_self_excitation=include_self_excitation,
        extra_ad_features=extra_ad_features,
        cross_channel_window_hours=cross_channel_window_hours,
        user_features=user_features,
    )

    logger.info(
        "  Fitting Poisson GLM with offset (n_intervals=%d, n_features=%d)...",
        len(interval_df), len(feature_cols),
    )
    model = _fit_poisson_model(interval_df, feature_cols)

    logger.info("  Credit assignment: %s", credit_method)
    if credit_method == "backelim":
        raw_credits = _backwards_elimination_credits(
            model, journeys, meta, query_events=None,
        )
        method_name = "Survival/Poisson (BackElim)"
    elif credit_method == "incremental":
        if query_events is None:
            logger.info(
                "  No query_events provided — incremental falls back to BE."
            )
        raw_credits = _incremental_credits(
            model, journeys, meta,
            query_events=query_events if query_events is not None else pd.DataFrame(),
        )
        method_name = "Survival/Poisson (Incremental)"
    elif credit_method == "aicpe":
        raw_credits = _aicpe_credits(model, interval_df, feature_cols)
        method_name = "Survival/Poisson (AICPE)"
    elif credit_method == "shapley":
        raw_credits = _shapley_credits(model, journeys, meta)
        method_name = "Survival/Poisson (Shapley)"
    else:
        raise ValueError(f"Unknown credit_method: {credit_method!r}")

    # Normalization (Eq 17, Eq 18, or sum-to-one)
    if normalize == "sum_to_one":
        clamped = {k: max(0.0, v) for k, v in raw_credits.items()}
        total = sum(clamped.values())
        normalized = (
            {k: v / total for k, v in clamped.items()}
            if total > 0
            else {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}
        )
    elif normalize in ("eq17", "eq18"):
        # Compute λ̂(A(n)) and λ̂(∅) summed over converted users for the denominators.
        params = model.params
        feature_cols = meta["feature_cols"]
        levels_per_feature = meta["levels_per_feature"]
        sum_lambda_full = 0.0
        sum_lambda_empty = 0.0
        for user_id, group in journeys[journeys["converted"]].groupby("user_id", sort=False):
            group = group.sort_values("touchpoint_idx").reset_index(drop=True)
            ts = group["timestamp"].values.astype(float)
            chs = group["channel"].values
            user_feat_values = _user_feature_values(group.iloc[0], levels_per_feature)
            t_star = float(ts.max())
            sum_lambda_full += _predict_intensity_at(
                params, t_star, list(range(len(group))), chs, ts, user_feat_values,
                feature_cols, meta,
            )
            sum_lambda_empty += _predict_intensity_at(
                params, t_star, [], chs, ts, user_feat_values, feature_cols, meta,
            )
        if normalize == "eq17":
            denom = sum_lambda_full
        else:  # eq18
            denom = sum_lambda_full - sum_lambda_empty
        normalized = (
            {k: v / denom for k, v in raw_credits.items()}
            if denom > 0
            else {k: 0.0 for k in CHANNEL_NAMES}
        )
    else:
        raise ValueError(f"Unknown normalize: {normalize!r}")

    decay_curves = _extract_learned_decay(model, "tb")
    query_decay = _extract_learned_decay(model, "qb") if meta.get("has_queries") else None

    return AttributionResult(
        method=method_name,
        channel_credits=normalized,
        channel_credits_raw=raw_credits,
        metadata={
            "credit_method": credit_method,
            "normalize": normalize,
            "learned_decay_curves": decay_curves,
            "learned_query_decay_curves": query_decay,
            "time_bins_hours": list(TIME_BIN_EDGES_HOURS),
            "model_aic": float(model.aic),
            "model_deviance": float(model.deviance),
            "n_intervals": int(len(interval_df)),
            "n_features": int(len(feature_cols)),
            "estimated_betas": {ch: sum(decay_curves[ch]) for ch in CHANNEL_NAMES},
            "feature_cols": feature_cols,
            "options": {
                "include_position": include_position,
                "include_cross_channel": include_cross_channel,
                "include_seasonality": include_seasonality,
                "include_self_excitation": include_self_excitation,
                "extra_ad_features": list(extra_ad_features or []),
                "has_queries": bool(meta.get("has_queries")),
            },
        },
    )


def compute_backwards_elimination_attribution(
    journeys: pd.DataFrame,
    **kwargs,
) -> AttributionResult:
    """Alias for compute_survival_attribution(credit_method='backelim', **kwargs)."""
    kwargs.setdefault("credit_method", "backelim")
    return compute_survival_attribution(journeys, **kwargs)


def compute_aicpe_attribution(
    journeys: pd.DataFrame,
    config=None,
) -> AttributionResult:
    """DEPRECATED: use compute_survival_attribution(credit_method='aicpe').

    AICPE is a non-paper extension; the paper-faithful method is BackElim (Eq 13).
    """
    warnings.warn(
        "compute_aicpe_attribution is deprecated; use compute_survival_attribution"
        " with credit_method='aicpe' (note: AICPE is a non-paper extension).",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_survival_attribution(journeys, credit_method="aicpe")


# ============================================================
# Future Work — Survival × IPW Hybrid (Debiased Survival)
# ============================================================

def compute_survival_propensity_attribution(
    journeys: pd.DataFrame,
    user_features: Tuple[str, ...],
    credit_method: Literal["backelim", "shapley"] = "backelim",
    *,
    propensity_strategy: Literal["per_channel_logistic"] = "per_channel_logistic",
    stabilize_weights: bool = True,
    **kwargs,
) -> AttributionResult:
    """[FUTURE WORK] Doubly robust Survival/Poisson via IPW weighting.

    Combines two causal-inference primitives:
        - Outcome model (Survival/Poisson Eq 12) — current ``compute_survival_attribution``
        - Propensity model (per-channel exposure ~ user features) — NEW

    Pipeline:
        1. Build interval features (existing ``_build_interval_features``).
        2. For each channel c:
           - Fit logistic regression: ``P(channel c ever exposed | W) = e_c(W)``
             at user level using `user_features` as covariates W.
           - Compute IPW weights ``w_i = 1 / e_c(W_i)`` (or stabilized
             ``w_i = P(c) / e_c(W_i)``) per user, broadcast to that user's
             intervals.
        3. Weighted Poisson GLM (statsmodels supports ``freq_weights``):
           ``sm.GLM(y, X, family=Poisson(Log()), offset=log(Δt),
                    freq_weights=w)``.
        4. BackElim/Shapley credit on the weighted model — same algorithms
           as in this file, applied to the propensity-corrected fit.

    Doubly robust property: consistent if EITHER the outcome model
    (Eq 12 + user feature dummies, Eq 10) OR the propensity model
    (per-channel logistic) is correctly specified.

    Status: NOT IMPLEMENTED. See ``docs/Methodology_05_Causal_Attribution_Frameworks.md``
    Section 8.1 for full design rationale, propensity strategy options
    (per-channel vs multinomial), and known caveats (multi-channel exposure,
    weight stabilization, time-varying confounding).

    Args:
        journeys: long-format journey DataFrame.
        user_features: pre-treatment user-level columns to use as W in
            propensity estimation. Should be DAG-justified backdoor adjustment
            set (see Methodology_05 § 5.2 multivariate guidelines).
        credit_method: ``"backelim"`` (Eq 13) or ``"shapley"`` (Eq 25).
        propensity_strategy: ``"per_channel_logistic"`` (currently the only
            documented option) or future alternatives (e.g., multinomial).
        stabilize_weights: if True, use stabilized weights ``P(c)/e_c(W)``
            instead of raw ``1/e_c(W)``.
        **kwargs: forwarded to ``compute_survival_attribution`` for shared
            options (observation_end, include_position, ...).

    Raises:
        NotImplementedError: always. This is a stub for future work.
    """
    raise NotImplementedError(
        "compute_survival_propensity_attribution is future work — see "
        "docs/Methodology_05_Causal_Attribution_Frameworks.md Section 8.1 "
        "for the full Survival × IPW hybrid design (5-step pipeline)."
    )
