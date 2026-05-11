"""LSTM + Attention sequence model for conversion prediction and attribution.

Architecture: channel_one_hot(7) + time_delta(1) + position(1) = 9-dim input
    → LSTM(hidden=64) → Dot-product Attention → Dense(64→1) → Sigmoid

Attribution extraction (3 methods):
    1. Attention Weight: attention scores aggregated by channel
    2. Leave-One-Out: mask each touchpoint, measure prediction drop
    3. (SHAP DeepExplainer: optional, requires shap library)

Training: BCE loss with class weights, Adam, early stopping.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from part1_simulation import AttributionResult, CHANNEL_NAMES

logger = logging.getLogger(__name__)


# ============================================================
# Data Preparation
# ============================================================

class JourneyDataset(Dataset):
    """Convert journey DataFrame to padded tensors for LSTM/Transformer.

    Feature per touchpoint: [channel_one_hot(7), time_delta(1), position(1)] = 9-dim.
    """

    def __init__(self, journeys: pd.DataFrame, max_length: int = 20):
        self.max_length = max_length
        channel_to_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
        n_channels = len(CHANNEL_NAMES)

        user_groups = journeys.groupby("user_id", sort=False)
        features_list = []
        lengths_list = []
        labels_list = []

        for user_id, group in user_groups:
            group = group.sort_values("touchpoint_idx")
            seq_len = min(len(group), max_length)

            feat = np.zeros((max_length, n_channels + 2), dtype=np.float32)
            for i, (_, row) in enumerate(group.iterrows()):
                if i >= max_length:
                    break
                ch_idx = channel_to_idx.get(row["channel"])
                if ch_idx is not None:
                    feat[i, ch_idx] = 1.0  # one-hot
                # Time delta (hours, normalized by /100 for scale)
                if i == 0:
                    feat[i, n_channels] = 0.0
                else:
                    prev_ts = group.iloc[i - 1]["timestamp"]
                    feat[i, n_channels] = (row["timestamp"] - prev_ts) / 100.0
                # Position in journey [0, 1]
                jlen = row["journey_length"]
                feat[i, n_channels + 1] = i / max(1, jlen - 1) if jlen > 1 else 0.5

            features_list.append(feat)
            lengths_list.append(seq_len)
            labels_list.append(float(group["converted"].iloc[0]))

        self.features = np.stack(features_list)
        self.lengths = np.array(lengths_list, dtype=np.int64)
        self.labels = np.array(labels_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            self.lengths[idx],
            self.labels[idx],
        )


# ============================================================
# Model Architecture
# ============================================================

class LSTMAttentionModel(nn.Module):
    """LSTM(64) → Dot-product Attention → Dense → Sigmoid."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=0.0,
        )
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, attention_weights).

        Args:
            x: (batch, max_len, input_dim)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, 1) raw logits before sigmoid
            attn_weights: (batch, max_len) attention scores
        """
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        lstm_out, (h_n, _) = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x.size(1),
        )

        # Dot-product attention: query = last hidden state
        query = self.attention_query(h_n[-1]).unsqueeze(1)  # (batch, 1, hidden)
        scores = torch.bmm(query, lstm_out.transpose(1, 2)).squeeze(1)  # (batch, max_len)

        # Mask padding positions
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, max_len)
        attn_weights = attn_weights.masked_fill(mask, 0.0)

        # Context vector
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        context = self.dropout(context)

        logits = self.fc(context)  # (batch, 1)
        return logits, attn_weights


# ============================================================
# Training
# ============================================================

def train_lstm_model(
    journeys: pd.DataFrame,
    hidden_dim: int = 64,
    max_length: int = 20,
    batch_size: int = 256,
    lr: float = 0.001,
    epochs: int = 50,
    patience: int = 5,
    device: str = "cpu",
) -> Tuple[LSTMAttentionModel, dict]:
    """Train LSTM + Attention model on journey data.

    Returns:
        (trained model, training_info dict with metrics).
    """
    dataset = JourneyDataset(journeys, max_length=max_length)

    # Train/val/test split (70/15/15)
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator,
    )

    # Class-weighted sampling for imbalance
    train_labels = dataset.labels[train_ds.indices]
    pos_weight = (1 - train_labels.mean()) / max(train_labels.mean(), 1e-6)
    sample_weights = np.where(train_labels == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMAttentionModel(input_dim=9, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
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
                loss = criterion(logits.squeeze(-1), labels)
                val_loss += loss.item() * len(labels)
        val_loss /= len(val_ds)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test AUC
    all_probs = []
    all_labels = []
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

    training_info = {
        "test_auc": test_auc,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }

    return model, training_info


# ============================================================
# Attribution Extraction
# ============================================================

def _get_attention_attributions(
    model: LSTMAttentionModel,
    journeys: pd.DataFrame,
    max_length: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """Extract attribution from attention weights, aggregated by channel.

    For each converted journey, get attention weights per touchpoint,
    then sum by channel across all converted users.
    """
    converted = journeys.loc[journeys["converted"]]
    dataset = JourneyDataset(converted, max_length=max_length)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    channel_to_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
    channel_credits = {ch: 0.0 for ch in CHANNEL_NAMES}

    model.eval()
    user_idx = 0
    converted_groups = list(converted.groupby("user_id", sort=False))

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
                    ch = channels[j]
                    channel_credits[ch] += attn_np[i, j]

                user_idx += 1

    # Normalize
    total = sum(channel_credits.values())
    if total > 0:
        channel_credits = {k: v / total for k, v in channel_credits.items()}

    return channel_credits


def _get_loo_attributions(
    model: LSTMAttentionModel,
    journeys: pd.DataFrame,
    max_length: int = 20,
    device: str = "cpu",
    n_sample: int = 1000,
) -> Dict[str, float]:
    """Leave-One-Out attribution: mask each touchpoint, measure prediction drop.

    For each converted journey (subsampled for speed), zero out each
    touchpoint's features and measure the drop in conversion probability.
    """
    converted = journeys.loc[journeys["converted"]]
    dataset = JourneyDataset(converted, max_length=max_length)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_sample, len(dataset)), replace=False)

    channel_credits = {ch: 0.0 for ch in CHANNEL_NAMES}
    converted_groups = list(converted.groupby("user_id", sort=False))

    model.eval()
    with torch.no_grad():
        for idx in indices:
            features, length, label = dataset[idx]
            features = features.unsqueeze(0).to(device)
            length_t = torch.tensor([length], device=device)

            # Base prediction
            logits_base, _ = model(features, length_t)
            prob_base = torch.sigmoid(logits_base).item()

            if idx >= len(converted_groups):
                continue
            _, group = converted_groups[idx]
            channels = group.sort_values("touchpoint_idx")["channel"].tolist()
            seq_len = min(len(channels), max_length)

            for j in range(seq_len):
                # Zero out touchpoint j
                features_masked = features.clone()
                features_masked[0, j, :] = 0.0

                logits_masked, _ = model(features_masked, length_t)
                prob_masked = torch.sigmoid(logits_masked).item()

                drop = max(0.0, prob_base - prob_masked)
                channel_credits[channels[j]] += drop

    # Normalize
    total = sum(channel_credits.values())
    if total > 0:
        channel_credits = {k: v / total for k, v in channel_credits.items()}

    return channel_credits


def compute_lstm_attention_attribution(
    journeys: pd.DataFrame,
    method: str = "attention",
    hidden_dim: int = 64,
    max_length: int = 20,
    epochs: int = 50,
    device: str = "cpu",
    model: Optional[LSTMAttentionModel] = None,
    training_info: Optional[dict] = None,
) -> Tuple[AttributionResult, LSTMAttentionModel, dict]:
    """Train LSTM + Attention and extract attribution.

    Args:
        journeys: long-format journey DataFrame.
        method: "attention" or "loo" (Leave-One-Out).
        hidden_dim: LSTM hidden dimension.
        max_length: max sequence length for padding.
        epochs: training epochs.
        device: torch device.
        model: pre-trained model (skip training if provided).
        training_info: training info from previous run.

    Returns:
        (AttributionResult, trained_model, training_info).
    """
    if model is None:
        logger.info("Training LSTM + Attention model...")
        model, training_info = train_lstm_model(
            journeys, hidden_dim=hidden_dim, max_length=max_length,
            epochs=epochs, device=device,
        )

    if method == "attention":
        credits = _get_attention_attributions(model, journeys, max_length, device)
        method_name = "LSTM+Attention (attn weights)"
    elif method == "loo":
        credits = _get_loo_attributions(model, journeys, max_length, device)
        method_name = "LSTM+Attention (LOO)"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'attention' or 'loo'.")

    result = AttributionResult(
        method=method_name,
        channel_credits=credits,
        channel_credits_raw=credits,
        metadata={
            "extraction_method": method,
            "test_auc": training_info.get("test_auc", None) if training_info else None,
            "hidden_dim": hidden_dim,
        },
    )

    return result, model, training_info
