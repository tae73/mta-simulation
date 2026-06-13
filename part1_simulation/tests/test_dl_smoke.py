"""Smoke tests for the deep-learning attribution models.

Covers four PyTorch-based methods that share the same ``JourneyDataset`` and
9-dim feature representation:

    1. LSTM + Attention      (``lstm_attention.py``)
    2. Transformer           (``transformer.py``)
    3. CAMTA                  (``causal/camta.py``)
    4. Incremental Shapley(LSTM) (``causal/incremental_shapley_lstm.py``)

These are *structural* smoke tests — never exact numeric values (DL output is
nondeterministic across BLAS/threading). Every test asserts only invariants:
training runs a couple of tiny epochs without raising / NaN-Inf, and the
produced attribution is a valid simplex over ``CHANNEL_NAMES`` (finite,
non-negative, sums to ≈1.0) with predicted probabilities inside [0, 1].

All tests are marked ``slow`` (they train a model end-to-end).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest
import torch

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.models.lstm_attention import (
    JourneyDataset,
    LSTMAttentionModel,
    compute_lstm_attention_attribution,
    train_lstm_model,
)
from part1_simulation.models.transformer import (
    TransformerAttributionModel,
    compute_transformer_attribution,
)
from part1_simulation.models.causal.camta import (
    _compute_loo_targets,
    compute_camta_attribution,
)
from part1_simulation.models.causal.incremental_shapley_lstm import (
    _exact_shapley_over_coalitions,
    _mask_features_for_coalition,
    compute_incremental_shapley_lstm,
)
from part1_simulation.tests._journey_factory import default_dgp_config


pytestmark = pytest.mark.slow

SEED = 0


# ============================================================
# Shared fixtures-as-functions (no pytest fixtures, per house style)
# ============================================================

def _tiny_journeys():
    """Small synthetic journey set with a healthy converted minority.

    ``alpha_0=-2.0`` lifts the conversion rate well above the calibrated 2.5%
    so a 400-user sample yields ~200 converted journeys — enough for the
    train/val/test split and for converted-only attribution passes.
    """
    cfg = default_dgp_config(n_users=400, alpha_0=-2.0)
    journeys, _ = generate_all_journeys(cfg, calibrate=False)
    return journeys


def _seed_all(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _assert_valid_credits(credits: Dict[str, float], atol: float = 1e-3) -> None:
    """A valid attribution: exact channel keys, finite, non-negative, sums≈1."""
    assert set(credits.keys()) == set(CHANNEL_NAMES)
    values = np.array(list(credits.values()), dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values >= -atol)  # allow tiny negative FP dust
    np.testing.assert_allclose(values.sum(), 1.0, atol=atol)


def _assert_valid_result(result: AttributionResult, atol: float = 1e-3) -> None:
    assert isinstance(result, AttributionResult)
    assert isinstance(result.method, str) and result.method
    _assert_valid_credits(result.channel_credits, atol=atol)


# ============================================================
# JourneyDataset — shared 9-dim feature representation
# ============================================================

def test_journey_dataset_shapes_and_bounds():
    """Dataset packs each user into a (max_len, n_channels+2) float32 tensor."""
    journeys = _tiny_journeys()
    max_length = 20
    ds = JourneyDataset(journeys, max_length=max_length)

    n_users = journeys["user_id"].nunique()
    assert len(ds) == n_users

    feat, length, label = ds[0]
    assert feat.shape == (max_length, len(CHANNEL_NAMES) + 2)
    assert feat.dtype == torch.float32
    assert torch.all(torch.isfinite(feat))
    # One-hot block is binary; position column lives in [0, 1].
    one_hot_block = feat[:, : len(CHANNEL_NAMES)]
    assert torch.all((one_hot_block == 0.0) | (one_hot_block == 1.0))
    pos_col = feat[:, len(CHANNEL_NAMES) + 1]
    assert torch.all(pos_col >= 0.0) and torch.all(pos_col <= 1.0)

    assert 1 <= int(length) <= max_length
    assert float(label) in (0.0, 1.0)


# ============================================================
# 1. LSTM + Attention
# ============================================================

def test_lstm_train_runs_finite_loss():
    """train_lstm_model returns a model + info with a finite best val loss."""
    _seed_all()
    journeys = _tiny_journeys()
    model, info = train_lstm_model(
        journeys, hidden_dim=8, batch_size=64, epochs=2, patience=5,
    )
    assert isinstance(model, LSTMAttentionModel)
    assert math.isfinite(info["best_val_loss"])
    assert 0.0 <= info["test_auc"] <= 1.0
    assert info["epochs_trained"] >= 1


def test_lstm_forward_probs_in_unit_interval():
    """A forward pass through the trained model yields probabilities in [0,1]."""
    _seed_all()
    journeys = _tiny_journeys()
    model, _ = train_lstm_model(journeys, hidden_dim=8, batch_size=64, epochs=1)

    ds = JourneyDataset(journeys, max_length=20)
    feats = torch.stack([ds[i][0] for i in range(min(16, len(ds)))])
    lengths = torch.tensor([ds[i][1] for i in range(min(16, len(ds)))])

    model.eval()
    with torch.no_grad():
        logits, attn = model(feats, lengths)
        probs = torch.sigmoid(logits.squeeze(-1))
    assert torch.all(torch.isfinite(probs))
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    # Attention rows over valid positions form a (sub-)probability vector.
    assert torch.all(torch.isfinite(attn))
    assert torch.all(attn >= 0.0)
    row_sums = attn.sum(dim=1)
    assert torch.all(row_sums <= 1.0 + 1e-4)


def test_lstm_attention_attribution_valid():
    """Attention-weight extraction produces a valid channel simplex."""
    _seed_all()
    journeys = _tiny_journeys()
    result, model, info = compute_lstm_attention_attribution(
        journeys, method="attention", hidden_dim=8, epochs=2,
    )
    _assert_valid_result(result)
    assert isinstance(model, LSTMAttentionModel)
    assert result.metadata["extraction_method"] == "attention"


def test_lstm_loo_attribution_valid():
    """Leave-One-Out extraction (mask each touchpoint) is also a valid simplex."""
    _seed_all()
    journeys = _tiny_journeys()
    # Reuse a pre-trained model so we exercise only the LOO extraction branch.
    model, info = train_lstm_model(journeys, hidden_dim=8, batch_size=64, epochs=2)
    result, _, _ = compute_lstm_attention_attribution(
        journeys, method="loo", model=model, training_info=info,
    )
    _assert_valid_result(result)
    assert result.metadata["extraction_method"] == "loo"


def test_lstm_unknown_method_raises():
    """An unsupported extraction method is rejected before any heavy work."""
    _seed_all()
    journeys = _tiny_journeys()
    model, info = train_lstm_model(journeys, hidden_dim=8, batch_size=64, epochs=1)
    with pytest.raises(ValueError):
        compute_lstm_attention_attribution(
            journeys, method="nope", model=model, training_info=info,
        )


# ============================================================
# 2. Transformer (encoder-only, CLS token)
# ============================================================

def test_transformer_forward_probs_in_unit_interval():
    """Transformer forward emits finite logits and a valid attention map."""
    _seed_all()
    journeys = _tiny_journeys()
    model = TransformerAttributionModel(
        input_dim=9, d_model=8, n_heads=2, n_layers=1, max_length=20,
    )
    ds = JourneyDataset(journeys, max_length=20)
    feats = torch.stack([ds[i][0] for i in range(min(16, len(ds)))])
    lengths = torch.tensor([ds[i][1] for i in range(min(16, len(ds)))])

    model.eval()
    with torch.no_grad():
        logits, attn = model(feats, lengths)
        probs = torch.sigmoid(logits.squeeze(-1))
    assert torch.all(torch.isfinite(probs))
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    assert torch.all(torch.isfinite(attn))
    assert torch.all(attn >= 0.0)
    row_sums = attn.sum(dim=1)
    assert torch.all(row_sums <= 1.0 + 1e-4)


def test_transformer_attribution_valid():
    """End-to-end Transformer training + attention attribution is a valid simplex."""
    _seed_all()
    journeys = _tiny_journeys()
    result, model, info = compute_transformer_attribution(
        journeys, d_model=8, n_heads=2, n_layers=1, epochs=2,
    )
    _assert_valid_result(result)
    assert isinstance(model, TransformerAttributionModel)
    assert math.isfinite(info["best_val_loss"])
    assert 0.0 <= info["test_auc"] <= 1.0
    assert result.metadata["n_layers"] == 1


# ============================================================
# 3. CAMTA (causal-regularized attention)
# ============================================================

def test_camta_loo_targets_are_per_sample_simplex():
    """LOO targets are non-negative and sum to 1 per sample with active touchpoints."""
    _seed_all()
    journeys = _tiny_journeys()
    model = LSTMAttentionModel(input_dim=9, hidden_dim=8)

    ds = JourneyDataset(journeys, max_length=20)
    feats = torch.stack([ds[i][0] for i in range(min(8, len(ds)))])
    lengths = torch.tensor([ds[i][1] for i in range(min(8, len(ds)))])

    targets = _compute_loo_targets(model, feats, lengths)
    assert targets.shape == feats.shape[:2]
    targets_np = targets.cpu().numpy()
    assert np.all(np.isfinite(targets_np))
    assert np.all(targets_np >= 0.0)
    # Per-sample sum is either ~1 (some LOO effect) or ~0 (no measurable drop).
    row_sums = targets_np.sum(axis=1)
    for s in row_sums:
        assert (abs(s - 1.0) < 1e-4) or (s < 1e-4)


def test_camta_attribution_valid():
    """CAMTA training (BCE warmup → causal phase) yields a valid simplex."""
    _seed_all()
    journeys = _tiny_journeys()
    # epochs=3 > causal_warmup default(5) would skip the causal phase, so lower
    # warmup implicitly by training enough to cross it: use 6 epochs but keep the
    # model tiny so it stays fast.
    result = compute_camta_attribution(
        journeys, hidden_dim=8, epochs=6, lambda_causal=0.5,
    )
    _assert_valid_result(result)
    assert result.method == "CAMTA (Causal Attention)"
    assert 0.0 <= result.metadata["test_auc"] <= 1.0
    np.testing.assert_allclose(result.metadata["lambda_causal"], 0.5, atol=1e-9)


# ============================================================
# 4. Incremental Shapley (LSTM response model)
# ============================================================

def test_mask_features_keeps_only_coalition_channels():
    """Masking zeros the one-hot of out-of-coalition channels, keeps timing/pos."""
    journeys = _tiny_journeys()
    ds = JourneyDataset(journeys, max_length=20)
    features = ds.features  # (N, max_len, n_channels+2)
    keep = frozenset({CHANNEL_NAMES[0], CHANNEL_NAMES[3]})

    masked = _mask_features_for_coalition(features, keep)
    assert masked.shape == features.shape
    n_ch = len(CHANNEL_NAMES)
    for i, ch in enumerate(CHANNEL_NAMES):
        if ch in keep:
            np.testing.assert_allclose(masked[:, :, i], features[:, :, i], atol=1e-9)
        else:
            np.testing.assert_allclose(masked[:, :, i], 0.0, atol=1e-9)
    # Timing + position columns (last two) are untouched.
    np.testing.assert_allclose(masked[:, :, n_ch:], features[:, :, n_ch:], atol=1e-9)
    # Original array not mutated.
    assert not np.shares_memory(masked, features)


def test_exact_shapley_efficiency_full_minus_empty():
    """Shapley values sum to v(grand) - v(empty) (efficiency axiom)."""
    # Closed-form additive game: v(S) = sum of fixed per-channel contributions.
    contrib = {ch: float(i + 1) for i, ch in enumerate(CHANNEL_NAMES)}

    def value_fn(S):
        return sum(contrib[c] for c in S)

    shapley = _exact_shapley_over_coalitions(value_fn)
    # For an additive game each player's Shapley value equals its own contribution.
    for ch in CHANNEL_NAMES:
        np.testing.assert_allclose(shapley[ch], contrib[ch], atol=1e-9)
    total = value_fn(frozenset(CHANNEL_NAMES)) - value_fn(frozenset())
    np.testing.assert_allclose(sum(shapley.values()), total, atol=1e-9)


def test_incremental_shapley_lstm_attribution_valid():
    """Du-style Incremental Shapley over the LSTM response model is a valid simplex."""
    _seed_all()
    journeys = _tiny_journeys()
    result = compute_incremental_shapley_lstm(
        journeys,
        n_epochs=2,
        batch_size=64,
        hidden_dim=8,
        sample_users=80,
        device="cpu",
        random_seed=SEED,
    )
    _assert_valid_result(result)
    assert result.method == "Incremental Shapley (LSTM)"
    assert result.metadata["framework"] == "Du 2019"
    assert result.metadata["n_eval_users"] == 80
    # Raw (pre-clamp) credits must at least be finite.
    raw = np.array(list(result.channel_credits_raw.values()), dtype=float)
    assert np.all(np.isfinite(raw))
