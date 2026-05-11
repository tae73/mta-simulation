"""Experiment 04 — DGP Assumption Sensitivity.

Hypothesis: When interactions are removed, sequence models (Markov, LSTM) lose
advantage over flat methods. When heterogeneity is removed, causal methods
lose advantage because confounding disappears.

Setup: Four DGP variants on 20K users:
    1. Full (baseline)
    2. No interactions (δ=0)
    3. No decay (half_life=1000d → effectively no decay)
    4. No heterogeneity (η=0 for all segments)

Metric: MAE per method per variant.
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

from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.ground_truth import compute_ground_truth_intensity
from part1_simulation.evaluation.metrics import compute_mae, compute_kendall_tau
from part1_simulation.models.rule_based import compute_last_click, compute_linear, compute_time_decay
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.causal.propensity import compute_ipw_attribution
from part1_simulation.models.causal.dml import compute_dml_attribution

logger = logging.getLogger(__name__)

N_USERS = 20000

# DGP variants as Hydra overrides
VARIANTS = {
    "Full (baseline)": [],
    "No interactions (δ=0)": ["cross_influences=[]"],
    "No decay (half_life=1000d)": [
        "channels.0.decay_half_life_days=1000",
        "channels.1.decay_half_life_days=1000",
        "channels.2.decay_half_life_days=1000",
        "channels.3.decay_half_life_days=1000",
        "channels.4.decay_half_life_days=1000",
        "channels.5.decay_half_life_days=1000",
        "channels.6.decay_half_life_days=1000",
    ],
    "No heterogeneity (η=0)": [
        "segments.0.eta=0.0",
        "segments.1.eta=0.0",
        "segments.2.eta=0.0",
    ],
}


def _run_methods(journeys, config):
    """Run representative subset of methods."""
    return {
        "Last Click": compute_last_click(journeys),
        "Linear": compute_linear(journeys),
        "Time Decay": compute_time_decay(journeys),
        "Markov (2nd)": compute_markov_attribution(journeys, order=2),
        "Shapley (model)": compute_shapley_model_based(journeys),
        "Survival/Poisson": compute_survival_attribution(journeys),
        "IPW": compute_ipw_attribution(journeys),
        "DML": compute_dml_attribution(journeys),
    }


def run_experiment_04(output_dir: str = "results/part1") -> pd.DataFrame:
    """Run Experiment 04: DGP assumption sensitivity."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for variant_name, overrides in VARIANTS.items():
        logger.info(f"\n=== Variant: {variant_name} ===")
        full_overrides = [f"n_users={N_USERS}", "alpha_0=-5.625"] + overrides
        config = load_dgp_config(overrides=full_overrides)

        journeys, stats = generate_all_journeys(config, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, config)

        logger.info(f"  Converted: {stats['n_converted']} ({stats['conversion_rate']:.4f})")

        methods = _run_methods(journeys, config)
        for method_name, result in methods.items():
            mae = compute_mae(result.channel_credits, gt)
            tau = compute_kendall_tau(result.channel_credits, gt)
            all_rows.append({
                "variant": variant_name,
                "method": method_name,
                "mae": mae,
                "kendall_tau": tau,
            })

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(output_path / "04_dgp_sensitivity.csv", index=False)

    # Pivot and print
    pivot_mae = result_df.pivot(index="method", columns="variant", values="mae")
    print("\n=== Experiment 04: DGP Assumption Sensitivity (MAE) ===")
    print(pivot_mae.to_string(float_format="%.4f"))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot_mae.values, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(pivot_mae.columns)))
    ax.set_xticklabels(pivot_mae.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_mae.index)))
    ax.set_yticklabels(pivot_mae.index, fontsize=10)

    for i in range(len(pivot_mae.index)):
        for j in range(len(pivot_mae.columns)):
            ax.text(j, i, f"{pivot_mae.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=9)

    plt.colorbar(im, label="MAE")
    ax.set_title("Experiment 04: DGP Assumption Sensitivity", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_dgp_sensitivity.png", dpi=150)
    plt.close()

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")
    run_experiment_04()
