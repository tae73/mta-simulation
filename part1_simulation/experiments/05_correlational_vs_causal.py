"""Experiment 05 — Correlational vs Causal Attribution.

Hypothesis: As confounding strength increases, correlational methods degrade
while causal methods remain stable. This demonstrates when causal correction
becomes essential.

Setup: Vary confounding by adjusting segment-channel coupling:
    Weak: segments have mild channel preferences (overlapping start_channels)
    Medium: default config
    Strong: segments have extreme preferences (Loyal→only Email/Direct, New→only Display/Social)

The confounding mechanism: segment → channel exposure AND segment → conversion,
creating a spurious correlation between channels and outcomes.

Metric: MAE gap between correlational vs causal methods at each level.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.ground_truth import compute_ground_truth_intensity
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.experiments._common import (
    prepare_output_dir,
    setup_experiment_logging,
)
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.propensity import compute_ipw_attribution
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import compute_last_click, compute_time_decay
from part1_simulation.models.shapley import compute_shapley_model_based

logger = logging.getLogger(__name__)

N_USERS = 20000

# Confounding levels via segment eta differences
# Higher eta spread = stronger confounding
CONFOUNDING_LEVELS = {
    "Weak (η spread=0.2)": [
        "segments.0.eta=-0.1",  # New
        "segments.1.eta=0.0",   # Exploratory
        "segments.2.eta=0.1",   # Loyal
    ],
    "Medium (η spread=0.8, default)": [
        "segments.0.eta=-0.3",
        "segments.1.eta=0.0",
        "segments.2.eta=0.5",
    ],
    "Strong (η spread=2.0)": [
        "segments.0.eta=-0.8",
        "segments.1.eta=0.0",
        "segments.2.eta=1.2",
    ],
}

CORRELATIONAL_METHODS = ["Last Click", "Time Decay", "Markov (2nd)", "Shapley (model)"]
CAUSAL_METHODS = ["Survival/Poisson", "IPW", "DML"]


def _run_methods(journeys, config):
    """Run correlational and causal methods."""
    return {
        "Last Click": compute_last_click(journeys),
        "Time Decay": compute_time_decay(journeys),
        "Markov (2nd)": compute_markov_attribution(journeys, order=2),
        "Shapley (model)": compute_shapley_model_based(journeys),
        "Survival/Poisson": compute_survival_attribution(journeys),
        "IPW": compute_ipw_attribution(journeys),
        "DML": compute_dml_attribution(journeys),
    }


def run_experiment_05(output_dir: str = "results/part1") -> pd.DataFrame:
    """Run Experiment 05: correlational vs causal under varying confounding."""
    output_path = prepare_output_dir(output_dir)

    all_rows = []

    for level_name, overrides in CONFOUNDING_LEVELS.items():
        logger.info(f"\n=== Confounding: {level_name} ===")
        full_overrides = [f"n_users={N_USERS}", "alpha_0=-5.625"] + overrides
        config = load_dgp_config(overrides=full_overrides)

        journeys, stats = generate_all_journeys(config, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, config)

        # Measure confounding strength: correlation between segment and channel exposure
        user_seg = journeys.groupby("user_id")["segment"].first()
        seg_rates = journeys.groupby("user_id").agg(
            converted=("converted", "first"), segment=("segment", "first"),
        ).groupby("segment", observed=True)["converted"].mean()
        logger.info(f"  Conversion rate by segment: {seg_rates.to_dict()}")

        methods = _run_methods(journeys, config)
        for method_name, result in methods.items():
            mae = compute_mae(result.channel_credits, gt)
            tau = compute_kendall_tau(result.channel_credits, gt)
            category = "Causal" if method_name in CAUSAL_METHODS else "Correlational"
            all_rows.append({
                "confounding_level": level_name,
                "method": method_name,
                "category": category,
                "mae": mae,
                "kendall_tau": tau,
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(output_path / "05_correlational_vs_causal.csv", index=False)

    # Summary: avg MAE by category × confounding level
    summary = result_df.groupby(["confounding_level", "category"]).agg(
        mean_mae=("mae", "mean"),
        mean_tau=("kendall_tau", "mean"),
    ).reset_index()

    print("\n=== Experiment 05: Correlational vs Causal ===")
    print("\nAverage MAE by Category:")
    pivot = summary.pivot(index="confounding_level", columns="category", values="mean_mae")
    print(pivot.to_string(float_format="%.4f"))

    gap = pivot.get("Correlational", 0) - pivot.get("Causal", 0)
    print(f"\nMAE Gap (Correlational - Causal):")
    for level, g in gap.items():
        print(f"  {level}: {g:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = list(CONFOUNDING_LEVELS.keys())
    x = np.arange(len(levels))

    for category in ["Correlational", "Causal"]:
        subset = summary[summary["category"] == category]
        maes = [subset[subset["confounding_level"] == l]["mean_mae"].values[0] for l in levels]
        ax.plot(x, maes, "o-", label=category, linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Average MAE")
    ax.set_title("Experiment 05: Correlational vs Causal under Varying Confounding")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_correlational_vs_causal.png", dpi=150)
    plt.close()

    return result_df


if __name__ == "__main__":
    setup_experiment_logging()
    run_experiment_05()
