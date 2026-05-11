"""Experiment 07 — Budget Optimization Evaluation.

Hypothesis: Causal methods (Survival/Poisson, Incremental Shapley) that accurately
recover true DGP betas produce near-optimal budget allocations, while rule-based
methods (Last Click) over-invest in Paid Search due to lower-funnel bias.

Setup: Reconstruct channel_credits from Experiment 01 bias values (no re-run needed).
For each method, derive budget allocation via Linear Response, compare to GT optimal.

Metrics: Allocation MAE, Allocation Kendall's Tau (paid channels only).
"""

import json
import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation.config_loader import load_budget_config
from part1_simulation.evaluation.evaluate import evaluate_budget_allocation
from part1_simulation.experiments._common import (
    CATEGORY_COLORS,
    METHOD_CATEGORIES,
    prepare_output_dir,
    reconstruct_credits_from_eval,
    setup_experiment_logging,
)

logger = logging.getLogger(__name__)

CHANNEL_COLORS = {
    "Email": "#3498DB", "Display": "#2ECC71",
    "Social": "#F39C12", "Paid Search": "#E74C3C",
}


def plot_allocation_comparison(
    budget_df: pd.DataFrame,
    gt_optimal: Dict[str, float],
    output_dir: str,
) -> None:
    """Grouped bar chart: allocation by method for each paid channel."""
    paid_channels = sorted(gt_optimal.keys())
    methods = budget_df["method"].tolist()
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_methods)
    width = 0.18
    offsets = np.arange(len(paid_channels)) - (len(paid_channels) - 1) / 2

    for i, ch in enumerate(paid_channels):
        col = f"alloc_{ch}"
        if col in budget_df.columns:
            vals = budget_df[col].values
            color = CHANNEL_COLORS.get(ch, "#999")
            ax.bar(x + offsets[i] * width, vals, width, label=ch,
                   color=color, edgecolor="white", alpha=0.8)

    # GT optimal lines
    for i, ch in enumerate(paid_channels):
        gt_val = gt_optimal[ch]
        color = CHANNEL_COLORS.get(ch, "#999")
        ax.axhline(y=gt_val, color=color, linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Budget Allocation Fraction")
    ax.set_title("Experiment 07: Budget Allocation by Attribution Method\n"
                 "(dashed lines = GT optimal)", fontsize=14)
    ax.legend(title="Channel", loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_budget_allocation.png", dpi=150)
    plt.close()


def plot_allocation_mae_scatter(
    budget_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Scatter: allocation_mae vs allocation_tau, colored by category."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in budget_df.iterrows():
        cat = METHOD_CATEGORIES.get(row["method"], "")
        color = CATEGORY_COLORS.get(cat, "#999")
        ax.scatter(row["allocation_mae"], row["allocation_tau"],
                   c=color, s=120, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(row["method"],
                    (row["allocation_mae"], row["allocation_tau"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7.5)

    ax.set_xlabel("Allocation MAE (lower = better)", fontsize=12)
    ax.set_ylabel("Allocation Kendall's Tau (higher = better)", fontsize=12)
    ax.set_title("Budget Allocation: Accuracy vs Ranking Quality", fontsize=14)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_allocation_mae_tau.png", dpi=150)
    plt.close()


def run_experiment_07(
    data_dir: str = "data/simulation",
    output_dir: str = "results/part1",
) -> pd.DataFrame:
    """Run Experiment 07: budget allocation evaluation for all 18 methods."""
    output_path = prepare_output_dir(output_dir)

    # Load ground truth
    with open(f"{data_dir}/ground_truth.json") as f:
        gt = json.load(f)
    gt_a = gt["ground_truth_A"]["channel_credits"]
    gt_budget = gt["ground_truth_budget"]
    gt_optimal = gt_budget["optimal_allocation_fraction"]

    # Load Experiment 01 results
    eval_01 = pd.read_csv(f"{output_dir}/01_method_accuracy.csv")
    logger.info(f"Loaded {len(eval_01)} methods from Experiment 01")

    # Reconstruct channel credits from bias
    attr_results = reconstruct_credits_from_eval(eval_01, gt_a)

    # Load budget config
    budget_config = load_budget_config()
    n_converters = gt["data_statistics"]["n_converters"]

    # Evaluate budget allocation for all methods
    budget_df = evaluate_budget_allocation(
        results=attr_results,
        budget_config=budget_config,
        gt_optimal=gt_optimal,
        total_conversions=n_converters,
    )

    # Add method categories
    budget_df["category"] = budget_df["method"].map(METHOD_CATEGORIES)

    # Save
    budget_df.to_csv(output_path / "07_budget_optimization.csv", index=False)

    # Visualize
    plot_allocation_comparison(budget_df, gt_optimal, str(output_path))
    plot_allocation_mae_scatter(budget_df, str(output_path))

    # Print summary
    print(f"\n{'='*80}")
    print("Experiment 07: Budget Optimization Evaluation")
    print(f"{'='*80}")

    print(f"\nGT Optimal Allocation (paid channels):")
    for ch in sorted(gt_optimal, key=gt_optimal.get, reverse=True):
        print(f"  {ch:20s}: {gt_optimal[ch]:.4f}")
    print(f"  Efficiency ranking: {gt_budget['efficiency_ranking']}")

    print(f"\n{'Method':<35s} {'Alloc MAE':>10s} {'Alloc Tau':>10s}")
    print("-" * 57)
    for _, row in budget_df.iterrows():
        print(f"{row['method']:<35s} {row['allocation_mae']:>10.4f} {row['allocation_tau']:>10.4f}")

    # Category summary
    cat_summary = (
        budget_df.groupby("category")
        .agg(
            mean_alloc_mae=("allocation_mae", "mean"),
            best_alloc_mae=("allocation_mae", "min"),
            mean_alloc_tau=("allocation_tau", "mean"),
            n_methods=("method", "count"),
        )
        .sort_values("mean_alloc_mae")
    )
    print(f"\n=== Category Summary ===")
    print(cat_summary.to_string(float_format="%.4f"))

    best = budget_df.iloc[0]
    print(f"\nBest method (lowest Allocation MAE): {best['method']} "
          f"(MAE={best['allocation_mae']:.4f}, Tau={best['allocation_tau']:.4f})")

    return budget_df


if __name__ == "__main__":
    setup_experiment_logging()
    run_experiment_07()
