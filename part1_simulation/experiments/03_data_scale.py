"""Experiment 03 — Data Scale Sensitivity (Learning Curve).

Hypothesis: DL methods require more data than statistical methods.
Markov/Shapley stabilize at ~5K users; LSTM needs ~10K+.

Setup: Generate data at n_users = {1000, 5000, 10000, 50000, 100000}.
Run representative methods at each scale. Track MAE.

Output: Learning curve plot (x=n_users, y=MAE, one line per method).
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
from part1_simulation.models.lstm_attention import compute_lstm_attention_attribution
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.causal.dml import compute_dml_attribution

logger = logging.getLogger(__name__)

SCALE_LEVELS = [1000, 5000, 10000, 50000, 100000]
DL_MIN_USERS = 5000  # Minimum users for DL methods


def _run_methods_at_scale(journeys, config, n_users):
    """Run methods appropriate for the given data scale."""
    results = {}

    results["Last Click"] = compute_last_click(journeys)
    results["Linear"] = compute_linear(journeys)
    results["Time Decay"] = compute_time_decay(journeys)
    results["Markov (1st)"] = compute_markov_attribution(journeys, order=1)
    results["Shapley (model)"] = compute_shapley_model_based(journeys)
    results["Survival/Poisson"] = compute_survival_attribution(journeys)
    results["DML"] = compute_dml_attribution(journeys)

    # DL methods only if enough data
    if n_users >= DL_MIN_USERS:
        try:
            lstm_result, _, _ = compute_lstm_attention_attribution(
                journeys, method="attention", epochs=20,
            )
            results["LSTM+Attention"] = lstm_result
        except Exception as e:
            logger.warning(f"LSTM failed at n={n_users}: {e}")

    return results


def run_experiment_03(
    output_dir: str = "results/part1",
) -> pd.DataFrame:
    """Run Experiment 03: learning curve across data scales."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for n_users in SCALE_LEVELS:
        logger.info(f"\n=== Scale: {n_users:,} users ===")
        config = load_dgp_config(overrides=[
            f"n_users={n_users}",
            "alpha_0=-5.625",
            f"random_seed={42 + n_users}",
        ])

        journeys, stats = generate_all_journeys(config, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, config)
        n_converted = stats["n_converted"]

        logger.info(f"  Converted: {n_converted} ({stats['conversion_rate']:.4f})")

        methods = _run_methods_at_scale(journeys, config, n_users)

        for method_name, result in methods.items():
            mae = compute_mae(result.channel_credits, gt)
            tau = compute_kendall_tau(result.channel_credits, gt)
            all_rows.append({
                "n_users": n_users,
                "n_converted": n_converted,
                "method": method_name,
                "mae": mae,
                "kendall_tau": tau,
            })
            logger.info(f"  {method_name}: MAE={mae:.4f}, Tau={tau:.4f}")

    result_df = pd.DataFrame(all_rows)

    # Save
    result_df.to_csv(output_path / "03_data_scale.csv", index=False)

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    methods = result_df["method"].unique()
    for method in methods:
        subset = result_df[result_df["method"] == method]
        ax1.plot(subset["n_users"], subset["mae"], "o-", label=method, markersize=5)
        ax2.plot(subset["n_users"], subset["kendall_tau"], "o-", label=method, markersize=5)

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Users (log scale)")
    ax1.set_ylabel("MAE vs Ground Truth")
    ax1.set_title("Learning Curve: MAE")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Users (log scale)")
    ax2.set_ylabel("Kendall's Tau")
    ax2.set_title("Learning Curve: Ranking Accuracy")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 03: Data Scale Sensitivity", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_data_scale.png", dpi=150)
    plt.close()

    # Print summary
    print("\n=== Experiment 03: Data Scale Sensitivity ===")
    pivot = result_df.pivot(index="method", columns="n_users", values="mae")
    print("\nMAE by Scale:")
    print(pivot.to_string(float_format="%.4f"))

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")
    run_experiment_03()
