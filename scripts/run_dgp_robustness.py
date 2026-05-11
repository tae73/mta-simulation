"""Cross-DGP robustness evaluation: 4 alternative DGPs × all methods.

For each of {logistic, markov, cox, hawkes} DGPs:
  - Generate 20K user data
  - Run all attribution methods + Survival/Poisson Shapley variant
  - Compute MAE, Kendall τ, top-3 accuracy vs GT
Output: results/part1/dgp_robustness.csv (long-format) + heatmap PNG.

Usage:
    PYTHONPATH=. python scripts/run_dgp_robustness.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.dgp.alternatives.cox_dgp import generate_dgp_cox
from part1_simulation.dgp.alternatives.hawkes_dgp import generate_dgp_hawkes
from part1_simulation.dgp.alternatives.logistic_dgp import generate_dgp_logistic
from part1_simulation.dgp.alternatives.markov_dgp import generate_dgp_markov
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.propensity import (
    compute_doubly_robust_attribution,
    compute_ipw_attribution,
)
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import (
    compute_first_click,
    compute_last_click,
    compute_linear,
    compute_position_based,
    compute_time_decay,
)
from part1_simulation.models.shapley import compute_shapley_model_based

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DGP_GENERATORS = {
    "logistic": generate_dgp_logistic,
    "markov": generate_dgp_markov,
    "cox": generate_dgp_cox,
    "hawkes": generate_dgp_hawkes,
}

N_USERS = 20_000


def _top3(credits: Dict[str, float]) -> set:
    return {c for c, _ in sorted(credits.items(), key=lambda x: -x[1])[:3]}


def run_methods_on_dgp(journeys: pd.DataFrame, dgp_name: str) -> List[AttributionResult]:
    """Run all attribution methods (excluding heavy DL) on a journeys DataFrame.

    DL methods (LSTM, Transformer, CAMTA) are excluded for speed — handled
    in a separate sweep if needed (LSTM IncShap is in Phase 3).
    """
    results: List[AttributionResult] = []

    logger.info(f"  [{dgp_name}] Rule-based (5)...")
    results.append(compute_last_click(journeys))
    results.append(compute_first_click(journeys))
    results.append(compute_linear(journeys))
    results.append(compute_time_decay(journeys))
    results.append(compute_position_based(journeys))

    logger.info(f"  [{dgp_name}] Markov (1st, 2nd)...")
    results.append(compute_markov_attribution(journeys, order=1))
    results.append(compute_markov_attribution(journeys, order=2))

    logger.info(f"  [{dgp_name}] Shapley (model-based)...")
    results.append(compute_shapley_model_based(journeys))

    logger.info(f"  [{dgp_name}] Incremental Shapley (LR)...")
    results.append(compute_incremental_shapley(journeys, sample_users=2000))

    logger.info(f"  [{dgp_name}] Survival/Poisson (BackElim, AICPE, Shapley)...")
    results.append(compute_survival_attribution(journeys, credit_method="backelim"))
    results.append(compute_survival_attribution(journeys, credit_method="aicpe"))
    results.append(compute_survival_attribution(journeys, credit_method="shapley"))

    logger.info(f"  [{dgp_name}] IPW / DR / DML...")
    results.append(compute_ipw_attribution(journeys))
    results.append(compute_doubly_robust_attribution(journeys))
    results.append(compute_dml_attribution(journeys))

    return results


def evaluate(results: List[AttributionResult], gt: Dict[str, float]) -> pd.DataFrame:
    rows = []
    truth_v = np.array([gt[c] for c in CHANNEL_NAMES])
    top3_gt = _top3(gt)
    for r in results:
        pred_v = np.array([r.channel_credits[c] for c in CHANNEL_NAMES])
        rows.append({
            "method": r.method,
            "mae": float(np.mean(np.abs(pred_v - truth_v))),
            "kendall_tau": compute_kendall_tau(r.channel_credits, gt),
            "top3_accuracy": len(_top3(r.channel_credits) & top3_gt) / 3.0,
        })
    return pd.DataFrame(rows)


def run_all_dgps(output_dir: str = "results/part1") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dgp_name, generator in DGP_GENERATORS.items():
        logger.info(f"\n=== DGP: {dgp_name} (n_users={N_USERS}) ===")
        if dgp_name == "hawkes":
            # Hawkes simulation is heavy; reduce
            j, gt, meta = generator(n_users=10_000, seed=42)
        elif dgp_name == "cox":
            j, gt, meta = generator(n_users=10_000, seed=42)
        else:
            j, gt, meta = generator(n_users=N_USERS, seed=42)
        logger.info(
            f"  generated: {j.shape}, conv={meta['conversion_rate']:.4f}"
        )
        logger.info(f"  GT: {gt}")

        method_results = run_methods_on_dgp(j, dgp_name)
        eval_df = evaluate(method_results, gt)
        eval_df["dgp"] = dgp_name
        eval_df["dgp_n_users"] = meta["n_users"]
        eval_df["dgp_conversion_rate"] = meta["conversion_rate"]
        all_rows.append(eval_df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(out / "dgp_robustness.csv", index=False)
    logger.info(f"Wrote {out / 'dgp_robustness.csv'}: {len(full)} rows")

    # Heatmap: method × DGP, MAE
    pivot_mae = full.pivot(index="method", columns="dgp", values="mae")
    # Sort methods by mean MAE across DGPs
    pivot_mae = pivot_mae.loc[pivot_mae.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(pivot_mae.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.15)
    ax.set_xticks(range(len(pivot_mae.columns)))
    ax.set_xticklabels(pivot_mae.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot_mae.index)))
    ax.set_yticklabels(pivot_mae.index, fontsize=8)
    for i in range(len(pivot_mae.index)):
        for j in range(len(pivot_mae.columns)):
            v = pivot_mae.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, label="MAE")
    ax.set_title("Cross-DGP Robustness: MAE per (method, DGP)\n"
                 "(rows sorted by mean MAE; lower=greener=better)")
    plt.tight_layout()
    plt.savefig(out / "dgp_robustness_heatmap.png", dpi=140)
    plt.close()
    logger.info(f"Wrote {out / 'dgp_robustness_heatmap.png'}")

    # Print summary
    print("\n=== Cross-DGP Mean MAE Ranking ===")
    rank = pivot_mae.mean(axis=1).sort_values()
    for m, v in rank.items():
        print(f"  {m:35s} mean MAE = {v:.4f}")

    return full


if __name__ == "__main__":
    run_all_dgps()
