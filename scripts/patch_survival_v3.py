"""Surgical patch: replace Survival/Poisson rows in Exp 01-10 CSVs with v3 numbers.

Re-runs ONLY compute_survival_attribution on the same data slices each experiment
uses. All other methods' rows are preserved untouched (deterministic regen would
produce identical numbers anyway).

PNGs are NOT regenerated here — they depend on all method results. Re-render
manually via experiment scripts after this patch if needed (or via notebook
re-execution which will pick up the updated CSVs).

Usage:
    PYTHONPATH=. python scripts/patch_survival_v3.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from part1_simulation import CHANNEL_NAMES
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.dgp.generate_data import generate_all_journeys
from part1_simulation.evaluation.ground_truth import compute_ground_truth_intensity
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

RESULTS = Path("results/part1")
DATA = Path("data/simulation")


def _top3(credits: Dict[str, float]) -> set:
    return {c for c, _ in sorted(credits.items(), key=lambda x: -x[1])[:3]}


# ============================================================
# Exp 01 — full 100K, BackElim row only
# ============================================================

def patch_exp_01() -> Dict[str, float]:
    logger.info("=== Exp 01 — full 100K ===")
    journeys = pd.read_parquet(DATA / "journeys.parquet")
    gt_a = json.load(open(DATA / "ground_truth.json"))["ground_truth_A"]["channel_credits"]

    r = compute_survival_attribution(journeys)
    truth_v = np.array([gt_a[c] for c in CHANNEL_NAMES])
    pred_v = np.array([r.channel_credits[c] for c in CHANNEL_NAMES])
    mae = float(np.mean(np.abs(pred_v - truth_v)))
    rmse = float(np.sqrt(np.mean((pred_v - truth_v) ** 2)))
    tau = compute_kendall_tau(r.channel_credits, gt_a)
    top3_acc = len(_top3(r.channel_credits) & _top3(gt_a)) / 3.0

    new_row = {
        "method": "Survival/Poisson (BackElim)",
        "mae": mae, "rmse": rmse, "kendall_tau": tau, "top3_accuracy": top3_acc,
    }
    for c in CHANNEL_NAMES:
        new_row[f"bias_{c}"] = float(r.channel_credits[c] - gt_a[c])

    csv = RESULTS / "01_method_accuracy.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv, index=False)
    logger.info(f"  Patched: MAE={mae:.4f}, τ={tau:.3f}")
    return new_row


# ============================================================
# Exp 02 — synergy detection (with vs without δ), 20K users
# ============================================================

SYNERGY_PAIRS = [
    ("Display", "Paid Search"),
    ("Social", "Email"),
    ("Organic Search", "Direct"),
]


def _synergy_score(credits: Dict[str, float], pair: Tuple[str, str]) -> float:
    s, t = pair
    return credits.get(s, 0.0) + credits.get(t, 0.0)


def patch_exp_02(n_users: int = 20000) -> Dict[str, float]:
    logger.info("=== Exp 02 — synergy detection (2 × 20K DGPs) ===")

    cfg_a = load_dgp_config(overrides=[f"n_users={n_users}", "alpha_0=-5.625"])
    j_a, _ = generate_all_journeys(cfg_a, calibrate=False)
    gt_a = compute_ground_truth_intensity(j_a, cfg_a)
    r_a = compute_survival_attribution(j_a)

    cfg_b = load_dgp_config(overrides=[f"n_users={n_users}", "alpha_0=-5.625", "cross_influences=[]"])
    j_b, _ = generate_all_journeys(cfg_b, calibrate=False)
    gt_b = compute_ground_truth_intensity(j_b, cfg_b)
    r_b = compute_survival_attribution(j_b)

    row = {"method": "Survival/Poisson"}
    for s, t in SYNERGY_PAIRS:
        pair_name = f"{s}→{t}"
        score_a = _synergy_score(r_a.channel_credits, (s, t))
        score_b = _synergy_score(r_b.channel_credits, (s, t))
        row[f"synergy_{pair_name}_with"] = score_a
        row[f"synergy_{pair_name}_without"] = score_b
        row[f"synergy_{pair_name}_delta"] = score_a - score_b
    row["mae_with"] = compute_mae(r_a.channel_credits, gt_a)
    row["mae_without"] = compute_mae(r_b.channel_credits, gt_b)

    csv = RESULTS / "02_interaction_effects.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)
    logger.info(f"  Patched: MAE_with={row['mae_with']:.4f}, MAE_without={row['mae_without']:.4f}")
    return row


# ============================================================
# Exp 03 — data scale curve, 5 scales
# ============================================================

SCALE_LEVELS = [1000, 5000, 10000, 50000, 100000]


def patch_exp_03() -> List[Dict[str, float]]:
    logger.info("=== Exp 03 — data scale (5 levels) ===")
    new_rows = []
    for n in SCALE_LEVELS:
        cfg = load_dgp_config(overrides=[
            f"n_users={n}", "alpha_0=-5.625", f"random_seed={42 + n}",
        ])
        journeys, stats = generate_all_journeys(cfg, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, cfg)
        r = compute_survival_attribution(journeys)
        mae = compute_mae(r.channel_credits, gt)
        tau = compute_kendall_tau(r.channel_credits, gt)
        new_rows.append({
            "n_users": n,
            "n_converted": stats["n_converted"],
            "method": "Survival/Poisson",
            "mae": mae,
            "kendall_tau": tau,
        })
        logger.info(f"  n={n}: conv={stats['n_converted']}, MAE={mae:.4f}, τ={tau:.3f}")

    csv = RESULTS / "03_data_scale.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    return new_rows


# ============================================================
# Exp 04 — DGP sensitivity, 4 variants × 20K
# ============================================================

DGP_VARIANTS = {
    "Full (baseline)": [],
    "No interactions (δ=0)": ["cross_influences=[]"],
    "No decay (half_life=1000d)": [
        f"channels.{i}.decay_half_life_days=1000" for i in range(7)
    ],
    "No heterogeneity (η=0)": [f"segments.{i}.eta=0.0" for i in range(3)],
}


def patch_exp_04(n_users: int = 20000) -> List[Dict[str, float]]:
    logger.info("=== Exp 04 — DGP sensitivity (4 variants) ===")
    new_rows = []
    for variant_name, overrides in DGP_VARIANTS.items():
        cfg = load_dgp_config(overrides=[f"n_users={n_users}", "alpha_0=-5.625"] + overrides)
        journeys, stats = generate_all_journeys(cfg, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, cfg)
        r = compute_survival_attribution(journeys)
        mae = compute_mae(r.channel_credits, gt)
        tau = compute_kendall_tau(r.channel_credits, gt)
        new_rows.append({
            "variant": variant_name,
            "method": "Survival/Poisson",
            "mae": mae,
            "kendall_tau": tau,
        })
        logger.info(f"  {variant_name}: MAE={mae:.4f}, τ={tau:.3f}")

    csv = RESULTS / "04_dgp_sensitivity.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    return new_rows


# ============================================================
# Exp 05 — correlational vs causal, 3 confounding levels
# ============================================================

CONFOUNDING_LEVELS = {
    "Weak (η spread=0.2)": ["segments.0.eta=-0.1", "segments.1.eta=0.0", "segments.2.eta=0.1"],
    "Medium (η spread=0.8, default)": ["segments.0.eta=-0.3", "segments.1.eta=0.0", "segments.2.eta=0.5"],
    "Strong (η spread=2.0)": ["segments.0.eta=-0.8", "segments.1.eta=0.0", "segments.2.eta=1.2"],
}


def patch_exp_05(n_users: int = 20000) -> List[Dict[str, float]]:
    logger.info("=== Exp 05 — confounding levels (3) ===")
    new_rows = []
    for level_name, overrides in CONFOUNDING_LEVELS.items():
        cfg = load_dgp_config(overrides=[f"n_users={n_users}", "alpha_0=-5.625"] + overrides)
        journeys, stats = generate_all_journeys(cfg, calibrate=False)
        gt = compute_ground_truth_intensity(journeys, cfg)
        r = compute_survival_attribution(journeys)
        mae = compute_mae(r.channel_credits, gt)
        tau = compute_kendall_tau(r.channel_credits, gt)
        new_rows.append({
            "confounding_level": level_name,
            "method": "Survival/Poisson",
            "category": "Causal",
            "mae": mae,
            "kendall_tau": tau,
        })
        logger.info(f"  {level_name}: MAE={mae:.4f}, τ={tau:.3f}")

    csv = RESULTS / "05_correlational_vs_causal.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(csv, index=False)
    return new_rows


# ============================================================
# Exp 08 — predictive validation, 80/20 split, BackElim
# ============================================================

def patch_exp_08(train_frac: float = 0.8, split_seed: int = 42) -> Dict[str, float]:
    logger.info("=== Exp 08 — predictive validation (80/20 split) ===")
    journeys = pd.read_parquet(DATA / "journeys.parquet")
    gt_a = json.load(open(DATA / "ground_truth.json"))["ground_truth_A"]["channel_credits"]

    rng = np.random.default_rng(split_seed)
    user_ids = journeys["user_id"].unique()
    rng.shuffle(user_ids)
    n_train = int(len(user_ids) * train_frac)
    train_uids = set(user_ids[:n_train])
    train = journeys[journeys["user_id"].isin(train_uids)]
    test = journeys[~journeys["user_id"].isin(train_uids)]

    r = compute_survival_attribution(train)
    test_labels = test.groupby("user_id")["converted"].first()

    weights = test["channel"].astype(str).map(r.channel_credits).fillna(0.0)
    scores = (test.assign(_w=weights).groupby("user_id")["_w"].sum())
    labels = test_labels.reindex(scores.index)

    s = scores.values
    y = labels.astype(int).values
    s_norm = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.full_like(s, 0.5, dtype=float)

    auc = float(roc_auc_score(y, s))
    pr_auc = float(average_precision_score(y, s))
    brier = float(brier_score_loss(y, s_norm))
    gt_mae = compute_mae(r.channel_credits, gt_a)
    gt_tau = compute_kendall_tau(r.channel_credits, gt_a)

    new_row = {
        "method": "Survival/Poisson (BackElim)",
        "category": "Causal",
        "auc": auc, "pr_auc": pr_auc, "brier": brier,
        "gt_mae": gt_mae, "gt_tau": gt_tau,
    }
    csv = RESULTS / "08_predictive_validation.csv"
    df = pd.read_csv(csv)
    df = df[~df["method"].astype(str).str.contains("Survival", na=False)]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).sort_values(
        "auc", ascending=False
    ).reset_index(drop=True)
    df.to_csv(csv, index=False)
    logger.info(f"  Patched: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}, GT-MAE={gt_mae:.4f}")
    return new_row


# ============================================================
# Exp 10 — bootstrap stability, 5 resamples × full 100K, AICPE label
# ============================================================

def patch_exp_10(n_bootstrap: int = 5, base_seed: int = 1234) -> List[Dict[str, float]]:
    logger.info("=== Exp 10 — bootstrap stability (B=5) ===")
    journeys = pd.read_parquet(DATA / "journeys.parquet")
    n_users = journeys["user_id"].nunique()
    all_uids = journeys["user_id"].unique()
    user_journeys = {uid: g for uid, g in journeys.groupby("user_id", sort=False)}

    raw_rows = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(base_seed + b)
        sampled = rng.choice(all_uids, size=n_users, replace=True)
        parts = []
        for new_uid, orig_uid in enumerate(sampled):
            g = user_journeys[orig_uid].copy()
            g["user_id"] = new_uid
            parts.append(g)
        sample = pd.concat(parts, ignore_index=True)
        r = compute_survival_attribution(sample)
        for ch, credit in r.channel_credits.items():
            raw_rows.append({
                "method": "Survival/Poisson (AICPE)",  # preserve original label
                "bootstrap_idx": b,
                "channel": ch,
                "credit": float(credit),
            })
        logger.info(f"  bootstrap {b+1}/{n_bootstrap} done")

    raw_df = pd.DataFrame(raw_rows)

    # Update raw CSV (preserve other methods)
    raw_csv = RESULTS / "10_bootstrap_raw.csv"
    raw_existing = pd.read_csv(raw_csv)
    raw_existing = raw_existing[~raw_existing["method"].astype(str).str.contains("Survival", na=False)]
    raw_combined = pd.concat([raw_existing, raw_df], ignore_index=True)
    raw_combined.to_csv(raw_csv, index=False)

    # Aggregate to 10_bootstrap_stability.csv (per-channel mean/std/CV/etc.)
    agg = raw_df.groupby(["method", "channel"]).agg(
        mean=("credit", "mean"),
        std=("credit", "std"),
        count=("credit", "count"),
        q025=("credit", lambda s: float(np.quantile(s, 0.025))),
        q975=("credit", lambda s: float(np.quantile(s, 0.975))),
    ).reset_index()
    agg["cv"] = agg["std"] / agg["mean"].abs().replace(0, np.nan)
    agg["ci_width"] = agg["q975"] - agg["q025"]
    agg["category"] = "Causal"

    stab_csv = RESULTS / "10_bootstrap_stability.csv"
    stab_existing = pd.read_csv(stab_csv)
    stab_existing = stab_existing[~stab_existing["method"].astype(str).str.contains("Survival", na=False)]
    stab_combined = pd.concat([stab_existing, agg], ignore_index=True)
    stab_combined.to_csv(stab_csv, index=False)

    # Mean CV across channels for this method
    mean_cv = float(agg["cv"].mean())
    logger.info(f"  Patched: mean CV across channels = {mean_cv:.3f}")
    return agg.to_dict(orient="records")


# ============================================================
# Driver
# ============================================================

def main():
    summary: Dict[str, object] = {}
    summary["exp_01"] = patch_exp_01()
    summary["exp_02"] = patch_exp_02()
    summary["exp_03"] = patch_exp_03()
    summary["exp_04"] = patch_exp_04()
    summary["exp_05"] = patch_exp_05()
    summary["exp_08"] = patch_exp_08()
    summary["exp_10"] = patch_exp_10()
    print("\n=== Patch summary ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            mae = v.get("mae") or v.get("gt_mae") or v.get("mae_with")
            print(f"  {k}: MAE={mae}")
        elif isinstance(v, list) and v and isinstance(v[0], dict) and "mae" in v[0]:
            for row in v:
                slice_key = (row.get("n_users") or row.get("variant") or
                             row.get("confounding_level"))
                print(f"  {k} [{slice_key}]: MAE={row['mae']:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: {len(v)} rows")


if __name__ == "__main__":
    main()
