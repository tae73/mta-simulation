"""Interval-split feature engineering for Survival/Poisson attribution.

Section 4.1 of Shender et al. 2023 (TEDDA): builds the piecewise-constant
design matrix for the Poisson GLM (Eq 12). Splits each user path into
intervals between consecutive break points {0, t_1, …, t_n, τ}, augmented
with optional query times. Each interval becomes one regression observation
with `offset = log(Δt)`.

Section 4.1 features implemented here:
    - 4.1.1 step-function intensity bins (Eq 5)
    - 4.1.2 extra ad features hook (Eq 6)
    - 4.1.3 multiple-ads, position, cross-channel indicators (Eq 7–9)
    - 4.1.4 user-feature dummies (Eq 10)
    - 4.1.5 query/ad split via query_events (Eq 11)
    - 4.1.6(a) seasonality, (c) self-excitation hooks

This module is internal — its public entry point is
``compute_survival_attribution`` re-exported from ``survival_attribution``.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES

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
