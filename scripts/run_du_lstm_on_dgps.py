"""Run Du Incremental Shapley (both LR and LSTM versions) on 4 alternative DGPs.

Patches dgp_robustness.csv with two new rows per DGP:
  - "Incremental Shapley (LSTM)" — Du paper-faithful
  - (LR version already in CSV from run_dgp_robustness.py)

Compares LR vs LSTM response model fairly under each DGP structure.

Usage:
    PYTHONPATH=. python scripts/run_du_lstm_on_dgps.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.alternatives.cox_dgp import generate_dgp_cox
from part1_simulation.dgp.alternatives.hawkes_dgp import generate_dgp_hawkes
from part1_simulation.dgp.alternatives.logistic_dgp import generate_dgp_logistic
from part1_simulation.dgp.alternatives.markov_dgp import generate_dgp_markov
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.models.causal.incremental_shapley_lstm import (
    compute_incremental_shapley_lstm,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

DGP_GENERATORS = {
    "logistic": (generate_dgp_logistic, 20_000),
    "markov":   (generate_dgp_markov, 20_000),
    "cox":      (generate_dgp_cox, 10_000),
    "hawkes":   (generate_dgp_hawkes, 10_000),
}


def _top3(credits):
    return {c for c, _ in sorted(credits.items(), key=lambda x: -x[1])[:3]}


def main():
    csv = Path("results/part1/dgp_robustness.csv")
    df = pd.read_csv(csv)

    new_rows = []
    for dgp_name, (generator, n_users) in DGP_GENERATORS.items():
        logger.info(f"\n=== DGP: {dgp_name} (n_users={n_users}, LSTM IncShap) ===")
        j, gt, meta = generator(n_users=n_users, seed=42)
        logger.info(f"  generated: {j.shape}, conv={meta['conversion_rate']:.4f}")

        r = compute_incremental_shapley_lstm(
            j, n_epochs=20, sample_users=min(5000, n_users),
        )

        truth_v = np.array([gt[c] for c in CHANNEL_NAMES])
        pred_v = np.array([r.channel_credits[c] for c in CHANNEL_NAMES])
        mae = float(np.mean(np.abs(pred_v - truth_v)))
        tau = compute_kendall_tau(r.channel_credits, gt)
        top3_acc = len(_top3(r.channel_credits) & _top3(gt)) / 3.0

        new_rows.append({
            "method": r.method,
            "mae": mae,
            "kendall_tau": tau,
            "top3_accuracy": top3_acc,
            "dgp": dgp_name,
            "dgp_n_users": meta["n_users"],
            "dgp_conversion_rate": meta["conversion_rate"],
        })
        logger.info(f"  MAE={mae:.4f}, τ={tau:.3f}, Top3={top3_acc:.0%}")

    # Append + dedupe
    df = df[df["method"] != "Incremental Shapley (LSTM)"]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    logger.info(f"\nUpdated {csv}: {len(df)} rows total ({len(new_rows)} new LSTM rows)")


if __name__ == "__main__":
    main()
