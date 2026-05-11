"""Encoder-only Transformer for conversion prediction and attribution.

Architecture: input(9-dim) → Linear projection(d_model=64) → Positional Encoding
    → TransformerEncoder(1-2 layers, 2 heads) → CLS token → Dense(1) → Sigmoid

Uses learned CLS token prepended to the sequence. Time-delta features are part
of the input (not separate positional encoding).

Attribution: Attention weight aggregation from multi-head self-attention.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.models.lstm_attention import JourneyDataset

logger = logging.getLogger(__name__)


class TransformerAttributionModel(nn.Module):
    """Encoder-only Transformer with CLS token for binary classification."""

    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_length: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learned positional encoding (max_length + 1 for CLS)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_length + 1, d_model) * 0.02,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

        # Store attention weights for attribution
        self._attention_weights = None
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hook to capture attention weights."""
        def hook_fn(module, input, output):
            # TransformerEncoderLayer internally uses MultiheadAttention
            pass

        # We'll extract attention via a different approach (explicit computation)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, max_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, 1)
            attn_weights: (batch, max_len) attention-like scores for attribution
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Project input
        x_proj = self.input_proj(x)  # (batch, max_len, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_tokens, x_proj], dim=1)  # (batch, max_len+1, d_model)

        # Add positional encoding
        x_with_cls = x_with_cls + self.pos_encoding[:, :seq_len + 1, :]

        # Create padding mask: True = ignore
        # CLS token is never masked (position 0)
        padding_mask = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            length = lengths[i].item()
            if length < seq_len:
                padding_mask[i, length + 1:] = True  # +1 for CLS offset

        # Transformer forward
        encoded = self.transformer(x_with_cls, src_key_padding_mask=padding_mask)

        # CLS token output → classification
        cls_output = encoded[:, 0, :]  # (batch, d_model)
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)  # (batch, 1)

        # Compute attention-like scores for attribution:
        # Use dot-product between CLS output and each position's encoding
        # as a proxy for attention-based attribution
        position_outputs = encoded[:, 1:, :]  # (batch, max_len, d_model)
        attn_scores = torch.bmm(
            cls_output.unsqueeze(1),  # (batch, 1, d_model)
            position_outputs.transpose(1, 2),  # (batch, d_model, max_len)
        ).squeeze(1) / math.sqrt(self.d_model)  # (batch, max_len)

        # Mask padding
        pad_mask_positions = padding_mask[:, 1:]  # remove CLS column
        attn_scores = attn_scores.masked_fill(pad_mask_positions, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(pad_mask_positions, 0.0)

        return logits, attn_weights


def train_transformer_model(
    journeys: pd.DataFrame,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    max_length: int = 20,
    batch_size: int = 256,
    lr: float = 0.0005,
    epochs: int = 50,
    patience: int = 7,
    device: str = "cpu",
) -> Tuple[TransformerAttributionModel, dict]:
    """Train Transformer model on journey data."""
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

    model = TransformerAttributionModel(
        input_dim=9, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_length=max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, lengths, labels in train_loader:
            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits, _ = model(features, lengths)
            loss = criterion(logits.squeeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, lengths, labels in val_loader:
                features = features.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                logits, _ = model(features, lengths)
                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

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
    }


def _get_transformer_attention_attributions(
    model: TransformerAttributionModel,
    journeys: pd.DataFrame,
    max_length: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Extract attribution from Transformer CLS-to-position attention scores."""
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


def compute_transformer_attribution(
    journeys: pd.DataFrame,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    max_length: int = 20,
    epochs: int = 50,
    device: str = "cpu",
    model: Optional[TransformerAttributionModel] = None,
    training_info: Optional[dict] = None,
) -> Tuple[AttributionResult, TransformerAttributionModel, dict]:
    """Train Transformer and extract attention-based attribution.

    Returns:
        (AttributionResult, trained_model, training_info).
    """
    if model is None:
        logger.info("Training Transformer model...")
        model, training_info = train_transformer_model(
            journeys, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            max_length=max_length, epochs=epochs, device=device,
        )

    credits = _get_transformer_attention_attributions(model, journeys, max_length, device)

    result = AttributionResult(
        method=f"Transformer ({n_layers}L/{n_heads}H)",
        channel_credits=credits,
        channel_credits_raw=credits,
        metadata={
            "test_auc": training_info.get("test_auc") if training_info else None,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
        },
    )

    return result, model, training_info
