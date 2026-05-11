"""Experiment 02 — Interaction Effect (Cross-Influence) Capture.

Hypothesis: Sequence-aware methods (Markov, LSTM) and causal methods detect
Display→PaidSearch synergy better than flat methods (Linear, Position-Based).

Setup:
    Condition A: Default DGP (δ_display→paid_search=0.4, δ_social→email=0.3, δ_organic→direct=0.2)
    Condition B: No interactions (all δ=0)

Metric: For each method, compute the "synergy detection score" = change in
attributed credit to synergy pairs between Condition A and B.
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
from part1_simulation.dgp.generate_data import generate_all_journeys, save_generated_data
from part1_simulation.evaluation.ground_truth import compute_ground_truth_intensity
from part1_simulation.evaluation.metrics import compute_mae, compute_kendall_tau
from part1_simulation.models.rule_based import (
    compute_last_click, compute_linear, compute_time_decay, compute_position_based,
)
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution

logger = logging.getLogger(__name__)

# Synergy pairs from DGP
SYNERGY_PAIRS = [
    ("Display", "Paid Search", 0.4),
    ("Social", "Email", 0.3),
    ("Organic Search", "Direct", 0.2),
]


def _run_subset_methods(journeys, config):
    """Run a representative subset of methods (faster than all 18)."""
    methods = {}
    methods["Last Click"] = compute_last_click(journeys)
    methods["Linear"] = compute_linear(journeys)
    methods["Time Decay"] = compute_time_decay(journeys)
    methods["Position-Based"] = compute_position_based(journeys)
    methods["Markov (1st)"] = compute_markov_attribution(journeys, order=1)
    methods["Markov (2nd)"] = compute_markov_attribution(journeys, order=2)
    methods["Shapley (model)"] = compute_shapley_model_based(journeys)
    methods["Survival/Poisson"] = compute_survival_attribution(journeys)
    return methods


def compute_synergy_score(credits: dict, pair: tuple) -> float:
    """Synergy detection score = combined credit of synergy pair channels."""
    source, target, _ = pair
    return credits.get(source, 0.0) + credits.get(target, 0.0)


def run_experiment_02(
    output_dir: str = "results/part1",
    n_users: int = 20000,
) -> pd.DataFrame:
    """Run Experiment 02: compare synergy detection across methods."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Condition A: Default DGP (with cross-influences)
    logger.info("=== Condition A: Default DGP (with cross-influences) ===")
    config_a = load_dgp_config(overrides=[f"n_users={n_users}", "alpha_0=-5.625"])
    journeys_a, _ = generate_all_journeys(config_a, calibrate=False)
    gt_a = compute_ground_truth_intensity(journeys_a, config_a)
    methods_a = _run_subset_methods(journeys_a, config_a)

    # Condition B: No interactions (δ=0)
    logger.info("\n=== Condition B: No interactions (δ=0) ===")
    config_b = load_dgp_config(overrides=[
        f"n_users={n_users}", "alpha_0=-5.625",
        "cross_influences=[]",
    ])
    journeys_b, _ = generate_all_journeys(config_b, calibrate=False)
    gt_b = compute_ground_truth_intensity(journeys_b, config_b)
    methods_b = _run_subset_methods(journeys_b, config_b)

    # Analyze synergy detection
    rows = []
    for method_name in methods_a:
        credits_a = methods_a[method_name].channel_credits
        credits_b = methods_b[method_name].channel_credits

        row = {"method": method_name}

        for pair in SYNERGY_PAIRS:
            source, target, delta = pair
            pair_name = f"{source}→{target}"
            score_a = compute_synergy_score(credits_a, pair)
            score_b = compute_synergy_score(credits_b, pair)
            row[f"synergy_{pair_name}_with"] = score_a
            row[f"synergy_{pair_name}_without"] = score_b
            row[f"synergy_{pair_name}_delta"] = score_a - score_b

        row["mae_with"] = compute_mae(credits_a, gt_a)
        row["mae_without"] = compute_mae(credits_b, gt_b)
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Print results
    print("\n=== Experiment 02: Interaction Effect Detection ===")
    print(f"\nGround Truth A (with δ): {gt_a}")
    print(f"Ground Truth B (δ=0):    {gt_b}")

    delta_cols = [c for c in result_df.columns if c.endswith("_delta")]
    print(f"\n{'Method':<20s} ", end="")
    for col in delta_cols:
        pair_name = col.replace("synergy_", "").replace("_delta", "")
        print(f"{pair_name:>20s} ", end="")
    print(f"{'MAE_Δ':>10s}")
    print("-" * 90)

    for _, row in result_df.iterrows():
        print(f"{row['method']:<20s} ", end="")
        for col in delta_cols:
            val = row[col]
            print(f"{val:>20.4f} ", end="")
        mae_delta = row["mae_with"] - row["mae_without"]
        print(f"{mae_delta:>10.4f}")

    # Save
    result_df.to_csv(output_path / "02_interaction_effects.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = result_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.25

    for i, pair in enumerate(SYNERGY_PAIRS):
        source, target, delta = pair
        pair_name = f"{source}→{target}"
        col = f"synergy_{pair_name}_delta"
        ax.bar(x + i * width, result_df[col], width, label=f"{pair_name} (δ={delta})")

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Credit change (with δ - without δ)")
    ax.set_title("Experiment 02: Synergy Detection by Method")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_interaction_effects.png", dpi=150)
    plt.close()

    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")
    run_experiment_02()
