"""Evaluation metrics for comparing attribution results against ground truth.

All functions take normalized channel credit dicts (channel_name → float, sum=1.0)
and return scalar or per-channel metrics.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats


def compute_mae(predicted: Dict[str, float], truth: Dict[str, float]) -> float:
    """Mean Absolute Error across channels.

    MAE = (1/n) * Σ |predicted_k - truth_k|
    """
    channels = sorted(truth.keys())
    errors = [abs(predicted.get(ch, 0.0) - truth[ch]) for ch in channels]
    return float(np.mean(errors))


def compute_rmse(predicted: Dict[str, float], truth: Dict[str, float]) -> float:
    """Root Mean Squared Error across channels.

    RMSE = sqrt((1/n) * Σ (predicted_k - truth_k)²)
    """
    channels = sorted(truth.keys())
    sq_errors = [(predicted.get(ch, 0.0) - truth[ch]) ** 2 for ch in channels]
    return float(np.sqrt(np.mean(sq_errors)))


def compute_kendall_tau(
    predicted: Dict[str, float],
    truth: Dict[str, float],
) -> float:
    """Kendall's Tau rank correlation of channel orderings.

    Returns tau in [-1, 1]. 1.0 = perfect rank agreement, -1.0 = reversed.
    """
    channels = sorted(truth.keys())
    pred_values = [predicted.get(ch, 0.0) for ch in channels]
    truth_values = [truth[ch] for ch in channels]

    tau, _ = scipy_stats.kendalltau(pred_values, truth_values)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_channel_bias(
    predicted: Dict[str, float],
    truth: Dict[str, float],
) -> Dict[str, float]:
    """Per-channel bias: predicted - truth.

    Positive = overestimation, negative = underestimation.
    """
    return {ch: predicted.get(ch, 0.0) - truth[ch] for ch in truth}


def compute_top_k_accuracy(
    predicted: Dict[str, float],
    truth: Dict[str, float],
    k: int = 3,
) -> float:
    """Fraction of top-k channels in ground truth that appear in predicted top-k."""
    truth_ranking = sorted(truth, key=truth.get, reverse=True)[:k]
    pred_ranking = sorted(predicted, key=predicted.get, reverse=True)[:k]
    overlap = len(set(truth_ranking) & set(pred_ranking))
    return overlap / k


def compute_all_metrics(
    predicted: Dict[str, float],
    truth: Dict[str, float],
) -> Dict[str, float]:
    """Compute all metrics at once.

    Returns flat dict with keys: mae, rmse, kendall_tau, top3_accuracy,
    and bias_{channel_name} for each channel.
    """
    result = {
        "mae": compute_mae(predicted, truth),
        "rmse": compute_rmse(predicted, truth),
        "kendall_tau": compute_kendall_tau(predicted, truth),
        "top3_accuracy": compute_top_k_accuracy(predicted, truth, k=3),
    }

    bias = compute_channel_bias(predicted, truth)
    for ch, b in bias.items():
        result[f"bias_{ch}"] = b

    return result


# ============================================================
# Budget Optimization Metrics
# ============================================================

def compute_allocation_mae(
    predicted_alloc: Dict[str, float],
    truth_alloc: Dict[str, float],
) -> float:
    """MAE of budget allocation fractions vs ground truth optimal.

    Only compares channels present in truth_alloc (paid channels).
    """
    channels = sorted(truth_alloc.keys())
    errors = [abs(predicted_alloc.get(ch, 0.0) - truth_alloc[ch]) for ch in channels]
    return float(np.mean(errors))


def compute_allocation_kendall_tau(
    predicted_alloc: Dict[str, float],
    truth_alloc: Dict[str, float],
) -> float:
    """Kendall's Tau of budget allocation rankings (paid channels only)."""
    channels = sorted(truth_alloc.keys())
    pred_vals = [predicted_alloc.get(ch, 0.0) for ch in channels]
    truth_vals = [truth_alloc[ch] for ch in channels]
    tau, _ = scipy_stats.kendalltau(pred_vals, truth_vals)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_channel_roas(
    channel_credits: Dict[str, float],
    channel_costs: Dict[str, float],
    total_conversions: int,
    revenue_per_conversion: float,
) -> Dict[str, float]:
    """ROAS = attributed_revenue / cost per paid channel.

    Skips zero-cost channels.
    """
    roas: Dict[str, float] = {}
    for ch, credit in channel_credits.items():
        cost = channel_costs.get(ch, 0.0)
        if cost <= 0.0:
            continue
        attributed_revenue = credit * total_conversions * revenue_per_conversion
        roas[ch] = attributed_revenue / cost if cost > 0 else 0.0
    return roas


def compute_channel_cpa(
    channel_credits: Dict[str, float],
    channel_costs: Dict[str, float],
    total_conversions: int,
) -> Dict[str, float]:
    """CPA = cost / attributed_conversions per paid channel.

    Skips zero-cost channels.
    """
    cpa: Dict[str, float] = {}
    for ch, credit in channel_credits.items():
        cost = channel_costs.get(ch, 0.0)
        if cost <= 0.0:
            continue
        attributed_conv = credit * total_conversions
        cpa[ch] = cost / attributed_conv if attributed_conv > 0 else 0.0
    return cpa
