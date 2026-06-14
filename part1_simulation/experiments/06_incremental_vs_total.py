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

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.experiments._common import (
    prepare_output_dir,
    setup_experiment_logging,
)
from part1_simulation.evaluation.ground_truth import (
    compute_ground_truth_intensity,
    compute_ground_truth_shapley,
)
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.shapley import compute_shapley_model_based

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

# Series shown in the figure: ground truth (GT_B, the counterfactual incremental
# truth) plus three credit operators. All normalized to channel-credit share (Σ=1)
# so the four are directly comparable on a single axis. (col, label, color, edge)
SERIES = [
    ("ground_truth_a", "GT_A — intensity decomp (conditional / total)", "#555555", "black"),
    ("ground_truth_b", "GT_B — counterfactual (marginal / incremental)", "#2E8B57", "black"),
    ("incremental_shapley", "Incremental Shapley", "#DDA0DD", "none"),
    ("total_shapley", "Total Shapley", "#45B7D1", "none"),
    ("backelim", "BackElim (survival, last-touch)", "#FF8C00", "none"),
]


def _normalize_share(credits: dict) -> dict:
    """Clamp negatives and normalize to a channel-credit share summing to 1."""
    pos = {ch: max(0.0, credits.get(ch, 0.0)) for ch in CHANNEL_NAMES}
    total = sum(pos.values())
    return {ch: (v / total if total > 0 else 0.0) for ch, v in pos.items()}


def run_experiment_06(output_dir: str = "results/part1") -> pd.DataFrame:
    """Run Experiment 06: Incremental vs Total Shapley at different base rates."""
    output_path = prepare_output_dir(output_dir)

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

        # Total Shapley (model-based logistic; includes baseline → over-credits)
        total_shap = compute_shapley_model_based(journeys)

        # Incremental Shapley (Du; learns response model, subtracts baseline)
        inc_shap = compute_incremental_shapley(journeys, sample_users=3000)
        actual_base_rate = inc_shap.metadata["base_conversion_rate"]

        # Two ground truths (both normalized to a channel-credit share):
        #   GT_A — intensity decomposition over converters = the CONDITIONAL / total
        #          truth ("who was present in the conversions"; retrospective audit;
        #          includes baseline-correlated structure).
        #   GT_B — counterfactual Shapley = the MARGINAL / incremental truth ("what is
        #          lost if a channel is removed" — the do-effect an A/B test measures).
        #          Incremental Shapley targets GT_B.
        gt_a_credits = compute_ground_truth_intensity(journeys, config)
        gt_b_credits = compute_ground_truth_shapley(journeys, config, sample_users=3000)

        # Survival/Poisson BackElim — last-touch credit operator on the IPP backbone.
        be_credits = compute_survival_attribution(
            journeys, credit_method="backelim"
        ).channel_credits

        # Normalize every series to a channel-credit share (Σ=1) for comparability.
        gt_a_n = _normalize_share(gt_a_credits)
        gt_b_n = _normalize_share(gt_b_credits)
        inc_n = _normalize_share(inc_shap.channel_credits)
        tot_n = _normalize_share(total_shap.channel_credits)
        be_n = _normalize_share(be_credits)

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
                "ground_truth_a": gt_a_n.get(ch, 0),
                "ground_truth_b": gt_b_n.get(ch, 0),
                "incremental_shapley": inc_n.get(ch, 0),
                "total_shapley": tot_n.get(ch, 0),
                "backelim": be_n.get(ch, 0),
                "difference": tot_n.get(ch, 0) - inc_n.get(ch, 0),
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
            print(f"  {row['channel']:20s}: GT_A={row['ground_truth_a']:.3f}  "
                  f"GT_B={row['ground_truth_b']:.3f}  Incr={row['incremental_shapley']:.3f}  "
                  f"Total={row['total_shapley']:.3f}  BE={row['backelim']:.3f}")

    # Plot — 4 series (GT_B + three credit operators), grouped bars per channel,
    # faceted by base rate. As base rate rises, Total Shapley peels off the ground
    # truth (collapses for upper-funnel) while Incremental Shapley stays glued to GT.
    levels = result_df["base_rate_level"].unique()
    if len(levels) > 0:
        fig, axes = plt.subplots(1, len(levels), figsize=(5.2 * len(levels), 6), sharey=True)
        if len(levels) == 1:
            axes = [axes]

        width = 0.16
        offsets = [-2 * width, -1 * width, 0.0, 1 * width, 2 * width]

        for ax, level in zip(axes, levels):
            subset = result_df[result_df["base_rate_level"] == level]
            channels = subset["channel"].values
            x = np.arange(len(channels))
            for (col, label, color, edge), off in zip(SERIES, offsets):
                ax.bar(x + off, subset[col], width, label=label, color=color,
                       edgecolor=edge, linewidth=(1.1 if edge != "none" else 0.0))
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
            base_rate = subset["base_rate"].iloc[0]
            ax.set_title(f"{level}\n(base={base_rate:.3f})", fontsize=10)

        axes[0].set_ylabel("Channel credit share (Σ=1)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(SERIES),
                   fontsize=8, frameon=False)
        plt.suptitle(
            "Experiment 06 — Two ground truths (GT_A conditional · GT_B incremental) vs credit operators\n"
            "Incremental Shapley tracks GT_B; Total Shapley drifts past GT_A, then collapses for upper-funnel at high base",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.06, 1, 0.93])
        plt.savefig(f"{output_dir}/06_incremental_vs_total.png", dpi=150, bbox_inches="tight")
        plt.close()

    return result_df


if __name__ == "__main__":
    setup_experiment_logging()
    run_experiment_06()
