"""Du et al. 2019 Incremental Shapley with LSTM + Attention response model.

Faithful re-implementation of the original paper's architecture. The LR-based
version (`incremental_shapley.py`) is preserved for fast baseline.

Pipeline:
    1. Train LSTM(64) + Bahdanau-style attention on (sequence, label) pairs.
    2. For each coalition S of channels (128 total):
       - Mask sequence: zero-out one-hot of channels not in S; keep timing/position.
       - Forward pass → mean P(conv | S) over users → coalition value v(S).
    3. Apply exact Shapley formula (128 coalitions × 7 channels).

Difference from existing `lstm_attention.py`:
    - That model produces per-touchpoint attribution (attention weights / LOO).
    - This module reuses the same architecture but the attribution is computed
      via game-theoretic coalition over CHANNELS (not touchpoints) — directly
      matching Du's two-step pipeline.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.lstm_attention import (
    JourneyDataset,
    LSTMAttentionModel,
    train_lstm_model,
)

logger = logging.getLogger(__name__)


# ============================================================
# Coalition value computation via channel masking
# ============================================================

def _mask_features_for_coalition(
    features: np.ndarray,
    coalition: FrozenSet[str],
) -> np.ndarray:
    """Zero-out one-hot of channels NOT in coalition; keep timing/position.

    features: (N_users, max_len, n_channels + 2)
    coalition: set of channel names to KEEP active

    Returns masked copy.
    """
    masked = features.copy()
    n_ch = len(CHANNEL_NAMES)
    for i, ch in enumerate(CHANNEL_NAMES):
        if ch not in coalition:
            masked[:, :, i] = 0.0
    return masked


def _coalition_value(
    model: LSTMAttentionModel,
    features: np.ndarray,
    lengths: np.ndarray,
    coalition: FrozenSet[str],
    device: torch.device,
    batch_size: int = 512,
) -> float:
    """Compute v(S) = mean P(conv | only channels in S active) over users."""
    masked = _mask_features_for_coalition(features, coalition)
    n = len(masked)

    model.eval()
    probs_list = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(masked[start:end]).to(device)
            lens = torch.from_numpy(lengths[start:end]).to(device)
            logits, _ = model(x, lens)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            probs_list.append(probs)

    return float(np.mean(np.concatenate(probs_list)))


# ============================================================
# Exact Shapley over 128 coalitions
# ============================================================

def _exact_shapley_over_coalitions(
    coalition_value_fn,
    channels: Tuple[str, ...] = CHANNEL_NAMES,
) -> Dict[str, float]:
    """Standard Shapley formula. coalition_value_fn(S: frozenset) -> float."""
    n = len(channels)
    cache: Dict[FrozenSet[str], float] = {}

    def v(S: FrozenSet[str]) -> float:
        if S not in cache:
            cache[S] = coalition_value_fn(S)
        return cache[S]

    shapley: Dict[str, float] = {ch: 0.0 for ch in channels}
    for ch_target in channels:
        others = [c for c in channels if c != ch_target]
        for r in range(n):
            for S_tuple in itertools.combinations(others, r):
                S = frozenset(S_tuple)
                S_with = S | {ch_target}
                marginal = v(S_with) - v(S)
                weight = (
                    math.factorial(len(S))
                    * math.factorial(n - len(S) - 1)
                    / math.factorial(n)
                )
                shapley[ch_target] += weight * marginal
    return shapley


# ============================================================
# Public API
# ============================================================

def compute_incremental_shapley_lstm(
    journeys: pd.DataFrame,
    n_epochs: int = 25,
    batch_size: int = 256,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    sample_users: Optional[int] = None,
    device: Optional[str] = None,
    random_seed: int = 42,
) -> AttributionResult:
    """Du Incremental Shapley with LSTM + Attention response model.

    Args:
        journeys: long-format journey DataFrame.
        n_epochs: LSTM training epochs.
        batch_size: training batch size.
        sample_users: if set, evaluate coalition values on a random subsample
            (for speed). 128 coalitions × N_users forwards is the bottleneck.
        device: 'cpu' / 'cuda' / None (auto).

    Returns:
        AttributionResult with method="Incremental Shapley (LSTM)".
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Train LSTM + Attention (uses internal dataset/split)
    logger.info(f"  Training LSTM (epochs={n_epochs}, hidden={hidden_dim})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, info = train_lstm_model(
            journeys=journeys,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            epochs=n_epochs,
            patience=5,
            device=str(device),
        )

    # Build a unified dataset for coalition value computation (over all users)
    dataset = JourneyDataset(journeys, max_length=20)

    logger.info(
        f"  LSTM trained: test_AUC={info.get('test_auc', float('nan')):.4f}"
    )

    # Subsample users for coalition value computation
    features_all = dataset.features
    lengths_all = dataset.lengths
    n_total = len(features_all)
    if sample_users is not None and sample_users < n_total:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n_total, size=sample_users, replace=False)
        features_eval = features_all[idx]
        lengths_eval = lengths_all[idx]
        logger.info(f"  Subsampling {sample_users}/{n_total} users for coalition values")
    else:
        features_eval = features_all
        lengths_eval = lengths_all

    logger.info(f"  Computing 128 coalition values via masked forward passes...")

    def coalition_value_fn(S: FrozenSet[str]) -> float:
        return _coalition_value(
            model, features_eval, lengths_eval, S, device, batch_size=batch_size,
        )

    raw_credits = _exact_shapley_over_coalitions(coalition_value_fn)

    # Normalize: clamp negatives, sum-to-one
    clamped = {k: max(0.0, v) for k, v in raw_credits.items()}
    total = sum(clamped.values())
    normalized = (
        {k: v / total for k, v in clamped.items()}
        if total > 0
        else {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}
    )

    return AttributionResult(
        method="Incremental Shapley (LSTM)",
        channel_credits=normalized,
        channel_credits_raw=raw_credits,
        metadata={
            "response_model": "LSTM+Attention",
            "n_epochs": n_epochs,
            "hidden_dim": hidden_dim,
            "test_auc": float(info.get("test_auc", float("nan"))),
            "n_eval_users": int(len(features_eval)),
            "framework": "Du 2019",
        },
    )
