"""Experiment 01 — Method Accuracy Comparison (Main Experiment).

Hypothesis: Causal methods (Incremental Shapley, Survival/Poisson, DML)
achieve lower MAE than correlational methods (Shapley, LSTM, Markov)
when confounding exists in the DGP.

Setup: Default DGP config with 100K users. Run all 18 methods.
Compare MAE and Kendall's Tau vs Ground Truth A (intensity decomposition).

Metrics: MAE, RMSE, Kendall's Tau, Top-3 accuracy per method.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from part1_simulation.config_loader import load_dgp_config
from part1_simulation.evaluation.evaluate import evaluate_all_methods, print_evaluation_summary
from part1_simulation.experiments._common import (
    CATEGORY_COLORS,
    METHOD_CATEGORIES,
    load_journeys_and_gt,
    prepare_output_dir,
    setup_experiment_logging,
)
from part1_simulation.models.causal.camta import compute_camta_attribution
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.propensity import (
    compute_doubly_robust_attribution,
    compute_ipw_attribution,
)
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.lstm_attention import compute_lstm_attention_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import run_all_rule_based
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.transformer import compute_transformer_attribution

logger = logging.getLogger(__name__)


def run_all_methods(journeys, config):
    """Run all attribution methods and collect results."""
    results = []

    logger.info("Running Rule-based (5 methods)...")
    results.extend(run_all_rule_based(journeys))

    logger.info("Running Markov Chain...")
    results.append(compute_markov_attribution(journeys, order=1))
    results.append(compute_markov_attribution(journeys, order=2))

    logger.info("Running Shapley Value (model-based)...")
    results.append(compute_shapley_model_based(journeys))

    logger.info("Running LSTM + Attention...")
    lstm_attn, model, info = compute_lstm_attention_attribution(
        journeys, method="attention", epochs=30,
    )
    results.append(lstm_attn)
    lstm_loo, _, _ = compute_lstm_attention_attribution(
        journeys, method="loo", model=model, training_info=info,
    )
    results.append(lstm_loo)

    logger.info("Running Transformer...")
    tf_result, _, _ = compute_transformer_attribution(journeys, epochs=30)
    results.append(tf_result)

    logger.info("Running Incremental Shapley...")
    results.append(compute_incremental_shapley(journeys, sample_users=3000))

    logger.info("Running Survival/Poisson...")
    results.append(compute_survival_attribution(journeys))

    logger.info("Running IPW...")
    results.append(compute_ipw_attribution(journeys))

    logger.info("Running Doubly Robust...")
    results.append(compute_doubly_robust_attribution(journeys))

    logger.info("Running DML...")
    results.append(compute_dml_attribution(journeys))

    logger.info("Running CAMTA...")
    results.append(compute_camta_attribution(journeys, epochs=25))

    return results


def plot_mae_comparison(eval_df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart: MAE by method, colored by category."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = [
        CATEGORY_COLORS.get(METHOD_CATEGORIES.get(m, ""), "#999999")
        for m in eval_df["method"]
    ]

    bars = ax.barh(range(len(eval_df)), eval_df["mae"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(eval_df)))
    ax.set_yticklabels(eval_df["method"], fontsize=10)
    ax.set_xlabel("MAE vs Ground Truth", fontsize=12)
    ax.set_title("Experiment 01: Attribution Method Accuracy Comparison", fontsize=14)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_mae_comparison.png", dpi=150)
    plt.close()


def plot_mae_vs_tau(eval_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: MAE vs Kendall's Tau, annotated by method name."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in eval_df.iterrows():
        cat = METHOD_CATEGORIES.get(row["method"], "")
        color = CATEGORY_COLORS.get(cat, "#999999")
        ax.scatter(row["mae"], row["kendall_tau"], c=color, s=100, edgecolor="black", zorder=3)
        ax.annotate(
            row["method"], (row["mae"], row["kendall_tau"]),
            textcoords="offset points", xytext=(5, 5), fontsize=7,
        )

    ax.set_xlabel("MAE (lower is better)", fontsize=12)
    ax.set_ylabel("Kendall's Tau (higher is better)", fontsize=12)
    ax.set_title("MAE vs Ranking Accuracy", fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_mae_vs_tau.png", dpi=150)
    plt.close()


def run_experiment_01(
    data_dir: str = "data/simulation",
    output_dir: str = "results/part1",
) -> pd.DataFrame:
    """Run Experiment 01: full method accuracy comparison."""
    output_path = prepare_output_dir(output_dir)

    journeys, _gt, gt_a = load_journeys_and_gt(data_dir)
    config = load_dgp_config(overrides=["alpha_0=-5.625"])

    results = run_all_methods(journeys, config)
    eval_df = evaluate_all_methods(results, gt_a)

    # Save
    eval_df.to_csv(output_path / "01_method_accuracy.csv", index=False)

    # Visualize
    plot_mae_comparison(eval_df, str(output_path))
    plot_mae_vs_tau(eval_df, str(output_path))

    # Print
    print_evaluation_summary(eval_df, gt_a)

    # Category summary
    eval_df["category"] = eval_df["method"].map(METHOD_CATEGORIES)
    cat_summary = eval_df.groupby("category").agg(
        mean_mae=("mae", "mean"),
        best_mae=("mae", "min"),
        mean_tau=("kendall_tau", "mean"),
    ).sort_values("mean_mae")

    print("\n=== Category Summary ===")
    print(cat_summary.to_string())

    return eval_df


if __name__ == "__main__":
    setup_experiment_logging()
    run_experiment_01()
