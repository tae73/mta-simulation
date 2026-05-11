"""Multivariate user-feature recovery experiment.

Hypothesis (H_recovery): On a DGP where two user features (segment + device)
both shift conversion baseline, multivariate Survival/Poisson
(`user_features=("segment", "device")`) achieves lower MAE vs ground truth
than univariate (`user_features=("segment",)`) — and lower than the
omit-one configurations.

Pipeline:
    1. Generate multivariate DGP at multiple seeds (default 5).
    2. For each seed, run 8 attribution calls:
       4 user_feature configs × 2 credit methods (shapley, backelim).
    3. Compute MAE, RMSE, Kendall τ, top-3 accuracy vs DGP ground truth.
    4. Save results/part1/multivariate_recovery.csv (40 rows for default sweep).
    5. Print mean ± std summary.

Usage:
    PYTHONPATH=. python scripts/run_multivariate_recovery.py
    PYTHONPATH=. python scripts/run_multivariate_recovery.py --seeds 42 --n-users 5000
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.alternatives.multivariate_dgp import generate_dgp_multivariate
from part1_simulation.evaluation.metrics import compute_kendall_tau
from part1_simulation.models.causal.survival_attribution import (
    compute_survival_attribution,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


USER_FEATURE_CONFIGS: List[Tuple[str, Tuple[str, ...]]] = [
    ("multivariate (segment+device)", ("segment", "device")),
    ("univariate (segment)",          ("segment",)),
    ("device only",                   ("device",)),
    ("no user feature",               ()),
]
CREDIT_METHODS = ("shapley", "backelim")


def _top3(d):
    return {c for c, _ in sorted(d.items(), key=lambda x: -x[1])[:3]}


def _metrics(pred_credits, gt_credits) -> Tuple[float, float, float, float]:
    truth = np.array([gt_credits[c] for c in CHANNEL_NAMES])
    pred = np.array([pred_credits[c] for c in CHANNEL_NAMES])
    err = pred - truth
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    tau = compute_kendall_tau(pred_credits, gt_credits)
    top3 = len(_top3(pred_credits) & _top3(gt_credits)) / 3.0
    return mae, rmse, tau, top3


def run_single_seed(seed: int, n_users: int) -> List[dict]:
    """Generate DGP at one seed, run all 8 method calls, return rows."""
    logger.info(f"=== seed={seed}, n_users={n_users} ===")
    j, gt, meta = generate_dgp_multivariate(n_users=n_users, seed=seed)
    logger.info(f"  conv={meta['conversion_rate']:.4f}, alpha0={meta['alpha0']:.3f}")

    rows: List[dict] = []
    for credit in CREDIT_METHODS:
        for label, ufs in USER_FEATURE_CONFIGS:
            r = compute_survival_attribution(
                j, credit_method=credit, user_features=ufs,
            )
            mae, rmse, tau, top3 = _metrics(r.channel_credits, gt)
            rows.append({
                "seed": seed,
                "n_users": int(meta["n_users"]),
                "conv_rate": float(meta["conversion_rate"]),
                "credit": credit,
                "user_features_label": label,
                "user_features": "+".join(ufs) if ufs else "(none)",
                "mae": mae,
                "rmse": rmse,
                "kendall_tau": tau,
                "top3_acc": top3,
            })
            logger.info(
                f"  {credit:9s} | {label:30s} MAE={mae:.4f} τ={tau:+.3f} top3={top3:.0%}"
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds, grouped by (credit, user_features_label)."""
    grp = df.groupby(["credit", "user_features_label"], sort=False)
    summary = grp.agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        tau_mean=("kendall_tau", "mean"),
        tau_std=("kendall_tau", "std"),
        top3_mean=("top3_acc", "mean"),
        top3_std=("top3_acc", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return summary


def hypothesis_test(df: pd.DataFrame) -> dict:
    """Per-credit, per-seed comparison: is multivariate MAE < univariate MAE?"""
    out = {}
    for credit in CREDIT_METHODS:
        sub = df[df["credit"] == credit]
        # Pivot seed × config -> mae
        piv = sub.pivot(index="seed", columns="user_features_label", values="mae")
        if "multivariate (segment+device)" not in piv.columns or "univariate (segment)" not in piv.columns:
            continue
        delta = piv["multivariate (segment+device)"] - piv["univariate (segment)"]
        n_seeds = len(delta)
        n_better = int((delta < 0).sum())
        mean_delta = float(delta.mean())
        out[credit] = {
            "n_seeds": int(n_seeds),
            "n_seeds_multivariate_better": n_better,
            "mean_mae_delta_multi_minus_uni": mean_delta,
            "fraction_better": float(n_better / n_seeds) if n_seeds > 0 else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=str, default="42,1,7,13,100",
        help="Comma-separated seed list (default: 42,1,7,13,100)",
    )
    parser.add_argument("--n-users", type=int, default=20_000)
    parser.add_argument(
        "--output", type=str, default="results/part1/multivariate_recovery.csv",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    logger.info(f"Running {len(seeds)} seeds × 8 method calls = {len(seeds)*8} runs")

    all_rows: List[dict] = []
    for seed in seeds:
        all_rows.extend(run_single_seed(seed, args.n_users))

    df = pd.DataFrame(all_rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(df)} rows to {out_path}")

    # Summary
    summary = summarize(df)
    print("\n=== Summary (mean ± std across seeds) ===")
    for credit in CREDIT_METHODS:
        sub = summary[summary["credit"] == credit]
        print(f"\n  credit_method = {credit}")
        for _, row in sub.iterrows():
            print(
                f"    {row['user_features_label']:32s} "
                f"MAE = {row['mae_mean']:.4f} ± {row['mae_std']:.4f}  "
                f"τ = {row['tau_mean']:+.3f} ± {row['tau_std']:.3f}  "
                f"top3 = {row['top3_mean']:.0%}"
            )

    # Hypothesis test
    htest = hypothesis_test(df)
    print("\n=== H_recovery: multivariate MAE < univariate MAE? ===")
    for credit, stats in htest.items():
        print(
            f"  {credit:9s}: {stats['n_seeds_multivariate_better']}/{stats['n_seeds']} "
            f"seeds better (Δ MAE = {stats['mean_mae_delta_multi_minus_uni']:+.4f}, "
            f"{stats['fraction_better']:.0%})"
        )


if __name__ == "__main__":
    main()
