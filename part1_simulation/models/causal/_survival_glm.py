"""Poisson GLM fit + intensity prediction for Survival/Poisson attribution.

Wraps the statsmodels Poisson GLM with log-Δt offset (Shender Eq 12) and
provides `_predict_intensity_at` for evaluating λ̂(t*, A) at arbitrary
observation time and active-ad subset — the building block reused by every
credit-assignment routine.

This module is internal — its functions are re-exported from
``survival_attribution`` for backward compatibility with tests.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from part1_simulation.models.causal._survival_features import _assign_bin

_GLMResult = Any


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
