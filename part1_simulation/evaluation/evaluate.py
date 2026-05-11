"""Unified evaluation runner: compare all attribution methods against ground truth.

Takes a list of AttributionResult objects and a ground truth dict,
computes all metrics, and returns a comparison DataFrame.
"""

from typing import Dict, List, Optional

import pandas as pd

from part1_simulation import AttributionResult, BudgetConfig, CHANNEL_NAMES, EvaluationResult
from part1_simulation.evaluation.metrics import (
    compute_all_metrics,
    compute_allocation_kendall_tau,
    compute_allocation_mae,
    compute_channel_bias,
    compute_kendall_tau,
    compute_mae,
    compute_rmse,
    compute_top_k_accuracy,
)
from part1_simulation.optimization.budget_optimizer import optimize_budget


def evaluate_method(
    result: AttributionResult,
    ground_truth: Dict[str, float],
) -> EvaluationResult:
    """Evaluate a single attribution method against ground truth.

    Args:
        result: attribution result from any model.
        ground_truth: normalized channel credits dict (sum=1.0).

    Returns:
        EvaluationResult NamedTuple.
    """
    return EvaluationResult(
        method=result.method,
        mae=compute_mae(result.channel_credits, ground_truth),
        kendall_tau=compute_kendall_tau(result.channel_credits, ground_truth),
        rmse=compute_rmse(result.channel_credits, ground_truth),
        channel_bias=compute_channel_bias(result.channel_credits, ground_truth),
    )


def evaluate_all_methods(
    results: List[AttributionResult],
    ground_truth: Dict[str, float],
) -> pd.DataFrame:
    """Compare all methods against ground truth.

    Args:
        results: list of AttributionResult from different methods.
        ground_truth: normalized channel credits dict.

    Returns:
        DataFrame with columns: method, mae, kendall_tau, rmse, top3_accuracy,
        and bias_{channel} for each channel. Sorted by MAE ascending.
    """
    rows = []
    for result in results:
        metrics = compute_all_metrics(result.channel_credits, ground_truth)
        metrics["method"] = result.method
        rows.append(metrics)

    df = pd.DataFrame(rows)

    # Reorder columns: method first, then main metrics, then bias
    main_cols = ["method", "mae", "rmse", "kendall_tau", "top3_accuracy"]
    bias_cols = [c for c in df.columns if c.startswith("bias_")]
    df = df[main_cols + sorted(bias_cols)]

    return df.sort_values("mae").reset_index(drop=True)


def print_evaluation_summary(
    eval_df: pd.DataFrame,
    ground_truth: Dict[str, float],
    ground_truth_name: str = "Ground Truth A",
) -> None:
    """Print a formatted evaluation summary table.

    Args:
        eval_df: DataFrame from evaluate_all_methods.
        ground_truth: the ground truth credits used for comparison.
        ground_truth_name: label for display.
    """
    print(f"\n{'='*80}")
    print(f"Attribution Method Evaluation vs {ground_truth_name}")
    print(f"{'='*80}")

    # Ground truth reference
    print(f"\n{ground_truth_name} (reference):")
    for ch in sorted(ground_truth, key=ground_truth.get, reverse=True):
        print(f"  {ch:20s}: {ground_truth[ch]:.4f}")

    # Main metrics table
    print(f"\n{'Method':<30s} {'MAE':>8s} {'RMSE':>8s} {'Tau':>8s} {'Top-3':>8s}")
    print("-" * 66)
    for _, row in eval_df.iterrows():
        print(
            f"{row['method']:<30s} "
            f"{row['mae']:>8.4f} "
            f"{row['rmse']:>8.4f} "
            f"{row['kendall_tau']:>8.4f} "
            f"{row['top3_accuracy']:>8.2f}"
        )

    # Best method
    best = eval_df.iloc[0]
    print(f"\nBest method (lowest MAE): {best['method']} "
          f"(MAE={best['mae']:.4f}, Tau={best['kendall_tau']:.4f})")


# ============================================================
# Budget Allocation Evaluation
# ============================================================

def evaluate_budget_allocation(
    results: List[AttributionResult],
    budget_config: BudgetConfig,
    gt_optimal: Dict[str, float],
    total_conversions: int,
) -> pd.DataFrame:
    """Compare budget allocations from each attribution method against GT optimal.

    For each method:
    1. Derive budget allocation from attribution credits (Linear Response)
    2. Compare against ground truth optimal allocation fractions

    Args:
        results: list of AttributionResult from different methods.
        budget_config: cost and budget configuration.
        gt_optimal: ground truth optimal_allocation_fraction (paid channels only).
        total_conversions: observed number of conversions.

    Returns:
        DataFrame with columns: method, allocation_mae, allocation_tau,
        and alloc_{channel} for each paid channel. Sorted by allocation_mae.
    """
    rows = []
    for result in results:
        opt = optimize_budget(result, budget_config, total_conversions)
        predicted_alloc = opt["allocation_fraction"]

        row = {
            "method": result.method,
            "allocation_mae": compute_allocation_mae(predicted_alloc, gt_optimal),
            "allocation_tau": compute_allocation_kendall_tau(predicted_alloc, gt_optimal),
        }

        # Per-channel allocation fractions
        for ch in sorted(gt_optimal.keys()):
            row[f"alloc_{ch}"] = predicted_alloc.get(ch, 0.0)
            row[f"gt_alloc_{ch}"] = gt_optimal[ch]

        rows.append(row)

    df = pd.DataFrame(rows)
    main_cols = ["method", "allocation_mae", "allocation_tau"]
    other_cols = [c for c in df.columns if c not in main_cols]
    df = df[main_cols + sorted(other_cols)]

    return df.sort_values("allocation_mae").reset_index(drop=True)
