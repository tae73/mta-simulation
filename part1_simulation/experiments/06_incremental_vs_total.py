"""Experiment 06 — Incremental vs Total Shapley.

Hypothesis: When base conversion rate is high (many natural conversions),
Incremental Shapley and Traditional Shapley diverge significantly.
At 0% base rate (all ad-driven), they should agree.

Setup: Vary alpha_0 to create different base conversion rates:
    - Very low base (~0%): alpha_0 = -10 → almost all conversions are ad-driven
    - Low base (~5%): alpha_0 adjusted
    - Medium base (~10%): alpha_0 adjusted
    - High base (~20%): alpha_0 adjusted higher

Metric: Per-channel credit difference between Incremental and Total Shapley.
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.ground_truth import compute_ground_truth_intensity
from part1_simulation.evaluation.metrics import compute_mae
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley

logger = logging.getLogger(__name__)

N_USERS = 15000

# alpha_0 values that produce different base conversion rates
# base rate = P(conv | no ads) = 1 - exp(-exp(alpha_0 + eta))
# With eta ∈ {-0.3, 0, 0.5}, need enough total conversions for Shapley to work
ALPHA_0_LEVELS = {
    "Very low base": -8.0,     # base ≈ 0%, total conv ~1-2%
    "Low base (~3%)": -3.5,    # base ≈ 3%, total conv ~19%
    "Medium base (~8%)": -2.5, # base ≈ 8%, total conv ~35%
    "High base (~20%)": -1.8,  # base ≈ 20%, total conv ~55%
}


def run_experiment_06(output_dir: str = "results/part1") -> pd.DataFrame:
    """Run Experiment 06: Incremental vs Total Shapley at different base rates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for level_name, alpha_0_base in ALPHA_0_LEVELS.items():
        logger.info(f"\n=== {level_name} (alpha_0_base={alpha_0_base}) ===")

        # We need total conversion ~5-10% for both Shapley methods to work
        # Adjust the DGP alpha_0 to account for the higher base
        config = load_dgp_config(overrides=[
            f"n_users={N_USERS}",
            f"alpha_0={alpha_0_base}",
        ])

        journeys, stats = generate_all_journeys(config, calibrate=False)
        actual_conv_rate = stats["conversion_rate"]

        if stats["n_converted"] < 50:
            logger.warning(f"  Too few converters ({stats['n_converted']}), skipping")
            continue

        # Total Shapley
        total_shap = compute_shapley_model_based(journeys)

        # Incremental Shapley (learns response model from data)
        inc_shap = compute_incremental_shapley(journeys, sample_users=3000)
        actual_base_rate = inc_shap.metadata["base_conversion_rate"]

        logger.info(f"  Base rate (no ads, learned): {actual_base_rate:.4f}")
        logger.info(f"  Actual conversion rate: {actual_conv_rate:.4f}")
        logger.info(f"  Incremental fraction: {max(0, actual_conv_rate - actual_base_rate) / max(actual_conv_rate, 1e-6):.2%}")

        for ch in CHANNEL_NAMES:
            all_rows.append({
                "base_rate_level": level_name,
                "alpha_0": alpha_0_base,
                "base_rate": actual_base_rate,
                "conversion_rate": actual_conv_rate,
                "channel": ch,
                "total_shapley": total_shap.channel_credits.get(ch, 0),
                "incremental_shapley": inc_shap.channel_credits.get(ch, 0),
                "difference": (
                    total_shap.channel_credits.get(ch, 0)
                    - inc_shap.channel_credits.get(ch, 0)
                ),
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(output_path / "06_incremental_vs_total.csv", index=False)

    # Print summary
    print("\n=== Experiment 06: Incremental vs Total Shapley ===")
    for level in result_df["base_rate_level"].unique():
        subset = result_df[result_df["base_rate_level"] == level]
        base_rate = subset["base_rate"].iloc[0]
        conv_rate = subset["conversion_rate"].iloc[0]
        mae_diff = subset["difference"].abs().mean()
        print(f"\n{level} (base={base_rate:.4f}, conv={conv_rate:.4f}):")
        print(f"  Avg |Total - Incremental|: {mae_diff:.4f}")
        for _, row in subset.iterrows():
            print(f"  {row['channel']:20s}: Total={row['total_shapley']:.4f}, "
                  f"Incr={row['incremental_shapley']:.4f}, Δ={row['difference']:+.4f}")

    # Plot
    levels = result_df["base_rate_level"].unique()
    if len(levels) > 0:
        fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 6), sharey=True)
        if len(levels) == 1:
            axes = [axes]

        for ax, level in zip(axes, levels):
            subset = result_df[result_df["base_rate_level"] == level]
            channels = subset["channel"].values
            x = np.arange(len(channels))
            width = 0.35

            ax.bar(x - width / 2, subset["total_shapley"], width,
                   label="Total Shapley", color="#45B7D1")
            ax.bar(x + width / 2, subset["incremental_shapley"], width,
                   label="Incremental Shapley", color="#DDA0DD")

            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
            base_rate = subset["base_rate"].iloc[0]
            ax.set_title(f"{level}\n(base={base_rate:.3f})", fontsize=10)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Channel Credit")
        plt.suptitle("Experiment 06: Incremental vs Total Shapley", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_incremental_vs_total.png", dpi=150)
        plt.close()

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")
    run_experiment_06()
