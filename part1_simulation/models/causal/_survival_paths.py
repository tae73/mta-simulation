"""Path-level incremental intensity for Survival/Poisson attribution.

Companion to ``_survival_credits`` (per-channel aggregation). This module
returns the per-user / per-path game value

    Δ_path(u) = λ̂(t*_u, all ads of u) − λ̂(t*_u, ∅)

i.e. the path-level Incremental Shapley value (Shender et al. 2023 §4.2.3
intensity backbone). By the Shapley efficiency axiom, Σ_u Δ_path equals the
§4 per-channel credit total over the same subpopulation — channel-level and
path-level views are the same game aggregated differently (Methodology 05
§3.4 Channel↔Path Duality, notebook 02 (Main) §7.5/§10).

This module is internal — ``compute_path_incrementality`` is re-exported
from ``survival_attribution`` (it is public; see ``__all__`` there).
"""

from typing import Any, Dict, List, Literal

import pandas as pd

from part1_simulation.models.causal._survival_features import _user_feature_values
from part1_simulation.models.causal._survival_glm import (
    _GLMResult,
    _predict_intensity_at,
)


def compute_path_incrementality(
    model: _GLMResult,
    journeys: pd.DataFrame,
    meta: Dict[str, Any],
    feature_cols: List[str],
    *,
    subpopulation: Literal["converters", "all"] = "converters",
) -> pd.DataFrame:
    """Per-user path-level Δ = λ̂(t*, all ads) − λ̂(t*, ∅).

    Reuses the SAME fitted ``model`` (never refits) — pass the single fit
    shared across the notebook. ``delta`` is UNCLAMPED so Σ delta telescopes
    exactly to the §4 backwards-elimination raw total.

    ``subpopulation``: ``"converters"`` (default, paper-faithful conditional
    estimand — Shender §4.2; notebook §7.5) iterates converted users only;
    ``"all"`` iterates ALL users for the G-computation marginal estimand
    (notebook §10). Same user-source convention as
    ``_backwards_elimination_credits`` / ``_shapley_credits``.

    Returns one row per user, columns:
        user_id      : user identifier (original dtype preserved)
        template     : tuple[str, ...]  ordered channel sequence
        path_length  : int              len(template)
        delta        : float            λ̂(full) − λ̂(∅), unclamped
    """
    if subpopulation not in ("converters", "all"):
        raise ValueError(
            f"subpopulation must be 'converters' or 'all', got {subpopulation!r}"
        )

    params = model.params
    levels_per_feature = meta["levels_per_feature"]

    user_source = (
        journeys[journeys["converted"]]
        if subpopulation == "converters"
        else journeys
    )

    records: List[Dict[str, Any]] = []
    for user_id, group in user_source.groupby("user_id", sort=False):
        group = group.sort_values("touchpoint_idx").reset_index(drop=True)
        n = len(group)
        channels = group["channel"].values
        timestamps = group["timestamp"].values.astype(float)
        t_star = float(timestamps.max())
        ufv = _user_feature_values(group.iloc[0], levels_per_feature)

        lam_full = _predict_intensity_at(
            params, t_star, list(range(n)),
            channels, timestamps, ufv, feature_cols, meta,
        )
        lam_empty = _predict_intensity_at(
            params, t_star, [],
            channels, timestamps, ufv, feature_cols, meta,
        )
        records.append({
            "user_id": user_id,
            "template": tuple(channels.tolist()),
            "path_length": n,
            "delta": float(lam_full - lam_empty),
        })

    return pd.DataFrame(records)
