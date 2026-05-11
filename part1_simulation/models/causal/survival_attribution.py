"""Survival/Poisson-based attribution (Shender et al. 2023, TEDDA).

Implementation of "A Time To Event Framework For Multi-touch Attribution"
(JDS 2023, ArXiv 2009.08432). Section 4 methodology fully covered.

Section 4.1 — Modeling User Conversion Behavior (see ``_survival_features``):
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

Section 4.2 — Credit Assignment (see ``_survival_credits``):
    - 4.2.1 Backwards Elimination (Eq 13): RawCredit(j) = λ̂(A(j)) - λ̂(A(j-1))
    - 4.2.2 Incremental Attribution (Eq 19, 20): mode="incremental" + query_events
    - 4.2.3 Synergy & Shapley (Eq 21, 24): _compute_synergy + compute_synergy_report

Module layout:
    - ``_survival_features``: interval-split design matrix (Section 4.1)
    - ``_survival_glm``:      Poisson GLM fit + λ̂(t*, A) prediction
    - ``_survival_credits``:  4 credit methods + synergy + decay extraction
    - this file:              public ``compute_survival_attribution`` orchestrator
                              + re-exports of internal helpers for tests
"""

import logging
import warnings
from typing import List, Literal, Optional, Tuple

import pandas as pd

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.causal._survival_credits import (
    _aicpe_credits,
    _backwards_elimination_credits,
    _compute_synergy_for_path,
    _extract_learned_decay,
    _incremental_credits,
    _shapley_credits,
    compute_synergy_report,
)
from part1_simulation.models.causal._survival_features import (
    TIME_BIN_EDGES_HOURS,
    _build_interval_features,
    _user_feature_values,
)
from part1_simulation.models.causal._survival_glm import (
    _build_query_features,
    _fit_poisson_model,
    _predict_intensity_at,
)

# Backward-compat re-exports — tests import these directly from this module.
__all__ = [
    "TIME_BIN_EDGES_HOURS",
    "compute_survival_attribution",
    "compute_backwards_elimination_attribution",
    "compute_aicpe_attribution",
    "compute_survival_propensity_attribution",
    "compute_synergy_report",
    # Internal helpers re-exported for test access:
    "_build_interval_features",
    "_user_feature_values",
    "_fit_poisson_model",
    "_predict_intensity_at",
    "_build_query_features",
    "_backwards_elimination_credits",
    "_incremental_credits",
    "_aicpe_credits",
    "_shapley_credits",
    "_compute_synergy_for_path",
    "_extract_learned_decay",
]

logger = logging.getLogger(__name__)


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
