"""Patch dgp_robustness.csv with RMSE column by re-running methods on each DGP."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.alternatives.cox_dgp import generate_dgp_cox
from part1_simulation.dgp.alternatives.hawkes_dgp import generate_dgp_hawkes
from part1_simulation.dgp.alternatives.logistic_dgp import generate_dgp_logistic
from part1_simulation.dgp.alternatives.markov_dgp import generate_dgp_markov
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.propensity import (
    compute_doubly_robust_attribution, compute_ipw_attribution,
)
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import (
    compute_first_click, compute_last_click, compute_linear,
    compute_position_based, compute_time_decay,
)
from part1_simulation.models.shapley import compute_shapley_model_based

warnings.filterwarnings("ignore")

DGPS = {
    "logistic": (generate_dgp_logistic, 20_000),
    "markov":   (generate_dgp_markov, 20_000),
    "cox":      (generate_dgp_cox, 10_000),
    "hawkes":   (generate_dgp_hawkes, 10_000),
}


def run_methods(j):
    out = []
    out.append(compute_last_click(j))
    out.append(compute_first_click(j))
    out.append(compute_linear(j))
    out.append(compute_time_decay(j))
    out.append(compute_position_based(j))
    out.append(compute_markov_attribution(j, order=1))
    out.append(compute_markov_attribution(j, order=2))
    out.append(compute_shapley_model_based(j))
    out.append(compute_incremental_shapley(j, sample_users=2000))
    out.append(compute_survival_attribution(j, credit_method="backelim"))
    out.append(compute_survival_attribution(j, credit_method="aicpe"))
    out.append(compute_survival_attribution(j, credit_method="shapley"))
    out.append(compute_ipw_attribution(j))
    out.append(compute_doubly_robust_attribution(j))
    out.append(compute_dml_attribution(j))
    return out


def metrics(r, gt):
    truth = np.array([gt[c] for c in CHANNEL_NAMES])
    pred = np.array([r.channel_credits[c] for c in CHANNEL_NAMES])
    err = pred - truth
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), compute_kendall_tau(r.channel_credits, gt)


rows = []
for dgp_name, (gen, n_users) in DGPS.items():
    print(f"=== {dgp_name} (n_users={n_users}) ===")
    j, gt, meta = gen(n_users=n_users, seed=42)
    print(f"  conv={meta['conversion_rate']:.4f}, generating methods...")
    results = run_methods(j)
    for r in results:
        mae, rmse, tau = metrics(r, gt)
        rows.append({
            "method": r.method,
            "mae": mae,
            "rmse": rmse,
            "kendall_tau": tau,
            "dgp": dgp_name,
            "dgp_n_users": meta["n_users"],
            "dgp_conversion_rate": meta["conversion_rate"],
        })
    print(f"  done — {len(results)} methods")

df_new = pd.DataFrame(rows)

# Merge with existing (preserve LSTM IncShap rows already there)
existing = pd.read_csv("results/part1/dgp_robustness.csv")
# Keep LSTM rows from existing (we don't recompute LSTM)
lstm_rows = existing[existing["method"].str.contains("LSTM", na=False)].copy()
if "rmse" not in lstm_rows.columns:
    # Compute approximate RMSE for LSTM rows by re-running... skip, just pad NaN
    lstm_rows["rmse"] = np.nan
final = pd.concat([df_new, lstm_rows[df_new.columns.intersection(lstm_rows.columns)]], ignore_index=True)

final.to_csv("results/part1/dgp_robustness.csv", index=False)
print(f"\nSaved {len(final)} rows with RMSE column.")

# Print summary
piv_mae = final.pivot(index="method", columns="dgp", values="mae")
piv_rmse = final.pivot(index="method", columns="dgp", values="rmse")
piv_mae["mean"] = piv_mae.mean(axis=1)
piv_rmse["mean"] = piv_rmse.mean(axis=1)
piv_mae = piv_mae.sort_values("mean")
print("\n=== Cross-DGP Mean MAE Ranking ===")
for m in piv_mae.index:
    print(f"  {m:35s} mean MAE = {piv_mae.loc[m, 'mean']:.4f}")
print("\n=== Cross-DGP Mean RMSE Ranking ===")
piv_rmse_sorted = piv_rmse.sort_values("mean")
for m in piv_rmse_sorted.index:
    print(f"  {m:35s} mean RMSE = {piv_rmse_sorted.loc[m, 'mean']:.4f}")
