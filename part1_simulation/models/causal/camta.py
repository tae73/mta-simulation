"""CAMTA: Causal Attention Model for Multi-Touch Attribution.

Variant of LSTM + Attention with causal regularization:
auxiliary loss forces attention weights to align with Leave-One-Out
counterfactual effects rather than just predictive importance.

L_total = L_bce + λ_causal × MSE(attention_weights, normalized_LOO_effects)

Based on Kumar et al. (ICDM Workshop 2020) ideas.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.lstm_attention import (
    JourneyDataset,
    LSTMAttentionModel,
)

logger = logging.getLogger(__name__)


def _compute_loo_targets(
    model: LSTMAttentionModel,
    features: torch.Tensor,
    lengths: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute normalized LOO effects as causal attention targets.

    For each touchpoint, zero it out and measure the drop in prediction.
    Returns normalized LOO weights (sum=1 per sample).
    """
    batch_size, max_len, feat_dim = features.shape
    model.eval()

    with torch.no_grad():
        base_logits, _ = model(features, lengths)
        base_probs = torch.sigmoid(base_logits.squeeze(-1))

    loo_effects = torch.zeros(batch_size, max_len, device=device)

    with torch.no_grad():
        for j in range(max_len):
            masked = features.clone()
            masked[:, j, :] = 0.0
            masked_logits, _ = model(masked, lengths)
            masked_probs = torch.sigmoid(masked_logits.squeeze(-1))
            loo_effects[:, j] = F.relu(base_probs - masked_probs)

    # Mask padding positions
    mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    loo_effects = loo_effects.masked_fill(mask, 0.0)

    # Normalize per sample
    loo_sums = loo_effects.sum(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = loo_effects / loo_sums

    return normalized.detach()


def train_camta_model(
    journeys: pd.DataFrame,
    hidden_dim: int = 64,
    max_length: int = 20,
    batch_size: int = 256,
    lr: float = 0.001,
    epochs: int = 50,
    patience: int = 7,
    lambda_causal: float = 0.5,
    causal_warmup: int = 5,
    device: str = "cpu",
) -> Tuple[LSTMAttentionModel, dict]:
    """Train CAMTA model with causal regularization.

    Phase 1 (epochs 1..causal_warmup): Train with BCE only
    Phase 2 (epochs causal_warmup+1..): Add causal auxiliary loss

    Args:
        lambda_causal: weight for the causal alignment loss.
        causal_warmup: number of epochs with BCE-only before adding causal loss.
    """
    dataset = JourneyDataset(journeys, max_length=max_length)

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator,
    )

    train_labels = dataset.labels[train_ds.indices]
    pos_weight = (1 - train_labels.mean()) / max(train_labels.mean(), 1e-6)
    sample_weights = np.where(train_labels == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMAttentionModel(input_dim=9, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )
    mse_criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        use_causal = epoch >= causal_warmup

        for features, lengths, labels in train_loader:
            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits, attn_weights = model(features, lengths)
            loss_bce = bce_criterion(logits.squeeze(-1), labels)

            if use_causal:
                # Compute LOO targets (detached — no gradient through LOO)
                loo_targets = _compute_loo_targets(model, features, lengths, device)
                # Mask padding in both
                mask = torch.arange(features.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
                attn_masked = attn_weights.masked_fill(mask, 0.0)
                loss_causal = mse_criterion(attn_masked, loo_targets)
                loss = loss_bce + lambda_causal * loss_causal
            else:
                loss = loss_bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, lengths, labels in val_loader:
                features = features.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                logits, _ = model(features, lengths)
                loss = bce_criterion(logits.squeeze(-1), labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            phase = "causal" if use_causal else "warmup"
            logger.info(
                f"  Epoch {epoch+1:3d} [{phase}]: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test AUC
    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, lengths, labels in test_loader:
            features = features.to(device)
            lengths = lengths.to(device)
            logits, _ = model(features, lengths)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    from sklearn.metrics import roc_auc_score
    try:
        test_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        test_auc = 0.5

    logger.info(f"  Test AUC: {test_auc:.4f}")

    return model, {
        "test_auc": test_auc,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "lambda_causal": lambda_causal,
    }


def _get_camta_attributions(
    model: LSTMAttentionModel,
    journeys: pd.DataFrame,
    max_length: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Extract attribution from CAMTA model's causally-regularized attention."""
    converted = journeys.loc[journeys["converted"]]
    dataset = JourneyDataset(converted, max_length=max_length)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    channel_credits = {ch: 0.0 for ch in CHANNEL_NAMES}
    converted_groups = list(converted.groupby("user_id", sort=False))
    user_idx = 0

    model.eval()
    with torch.no_grad():
        for features, lengths, labels in loader:
            features = features.to(device)
            lengths = lengths.to(device)
            _, attn_weights = model(features, lengths)
            attn_np = attn_weights.cpu().numpy()

            for i in range(len(labels)):
                if user_idx >= len(converted_groups):
                    break
                _, group = converted_groups[user_idx]
                channels = group.sort_values("touchpoint_idx")["channel"].tolist()
                seq_len = min(len(channels), max_length)

                for j in range(seq_len):
                    channel_credits[channels[j]] += attn_np[i, j]
                user_idx += 1

    total = sum(channel_credits.values())
    if total > 0:
        channel_credits = {k: v / total for k, v in channel_credits.items()}

    return channel_credits


def compute_camta_attribution(
    journeys: pd.DataFrame,
    hidden_dim: int = 64,
    max_length: int = 20,
    epochs: int = 40,
    lambda_causal: float = 0.5,
    device: str = "cpu",
) -> AttributionResult:
    """Train CAMTA and extract causally-regularized attention attribution.

    Returns:
        AttributionResult with causal attention credits.
    """
    logger.info("Training CAMTA model (causal attention)...")
    model, info = train_camta_model(
        journeys,
        hidden_dim=hidden_dim,
        max_length=max_length,
        epochs=epochs,
        lambda_causal=lambda_causal,
        device=device,
    )

    credits = _get_camta_attributions(model, journeys, max_length, device)

    return AttributionResult(
        method="CAMTA (Causal Attention)",
        channel_credits=credits,
        channel_credits_raw=credits,
        metadata={
            "test_auc": info["test_auc"],
            "lambda_causal": lambda_causal,
            "epochs_trained": info["epochs_trained"],
        },
    )
