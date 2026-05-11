"""Experiment 08 — Out-of-Sample Predictive Validation.

Real-world validation: in production, attribution is fit on past data and
must generalize to new users. If a method's channel weights do not predict
held-out conversions, the credits are not a usable signal regardless of
their proximity to ground truth.

Hypothesis: GT-MAE (vs simulation truth) is rank-correlated with OOS predictive
performance. Methods that approximate the DGP intensity should also score
held-out journeys well.

Note on "out-of-time": journey timestamps in this DGP restart at 0 per user,
so calendar time is not available. We use a random user split (seed-fixed),
which still validates "out-of-sample" generalization. Rename in framing.

Setup:
    - Random 80/20 user split (seed=42).
    - Re-fit each method on train journeys → channel_credits.
    - Score test journey j as: score(j) = Σ_TP w_k where k = channel(TP).
    - Compute AUC, PR-AUC, Brier (after min-max scaling) on test conversion.

Outputs:
    - results/part1/08_predictive_validation.csv
    - results/part1/08_oos_auc_bar.png
    - results/part1/08_gt_mae_vs_oos_auc.png
    - results/part1/08_calibration_top3.png
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.config_loader import load_dgp_config
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae
from part1_simulation.models.causal.camta import compute_camta_attribution
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.propensity import (
    compute_doubly_robust_attribution, compute_ipw_attribution,
)
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.lstm_attention import compute_lstm_attention_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import run_all_rule_based
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.transformer import compute_transformer_attribution

logger = logging.getLogger(__name__)

METHOD_CATEGORIES = {
    "Last Click": "Rule-based", "First Click": "Rule-based",
    "Linear": "Rule-based", "Time Decay (7.0d)": "Rule-based",
    "Position-Based (40%/40%)": "Rule-based",
    "Markov (order=1)": "Statistical", "Markov (order=2)": "Statistical",
    "Shapley (model-based)": "Game-theoretic",
    "LSTM+Attention (attn weights)": "Deep Learning",
    "LSTM+Attention (LOO)": "Deep Learning",
    "Transformer (2L/2H)": "Deep Learning",
    "Incremental Shapley": "Causal (incremental)",
    "Survival/Poisson (BackElim)": "Causal (incremental)",
    "Survival/Poisson (AICPE)": "Causal (incremental)",
    "Survival/Poisson (Shapley)": "Causal (incremental)",
    "IPW": "Causal (debiased)", "Doubly Robust": "Causal (debiased)", "DML": "Causal (debiased)",
    "CAMTA (Causal Attention)": "Causal (incremental)",
}

CATEGORY_COLORS = {
    "Rule-based": "#4ECDC4", "Statistical": "#45B7D1",
    "Game-theoretic": "#96CEB4", "Deep Learning": "#FFEAA7",
    "Causal (debiased)": "#DDA0DD", "Causal (incremental)": "#B5D8B5",
}


def split_users(journeys: pd.DataFrame, train_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random 80/20 user split."""
    rng = np.random.default_rng(seed)
    user_ids = journeys["user_id"].unique()
    rng.shuffle(user_ids)
    n_train = int(len(user_ids) * train_frac)
    train_uids = set(user_ids[:n_train])

    train_mask = journeys["user_id"].isin(train_uids)
    return journeys.loc[train_mask].copy(), journeys.loc[~train_mask].copy()


def fit_all_methods(train_journeys: pd.DataFrame) -> List[AttributionResult]:
    """Fit every method on the train slice. Mirrors experiment 01."""
    results: List[AttributionResult] = []

    logger.info(f"Train: {train_journeys['user_id'].nunique()} users, "
                f"{train_journeys['converted'].groupby(train_journeys['user_id']).first().sum()} conv")

    logger.info("[1/12] Rule-based (5)")
    results.extend(run_all_rule_based(train_journeys))

    logger.info("[2/12] Markov order=1, 2")
    results.append(compute_markov_attribution(train_journeys, order=1))
    results.append(compute_markov_attribution(train_journeys, order=2))

    logger.info("[3/12] Shapley (model-based)")
    results.append(compute_shapley_model_based(train_journeys))

    logger.info("[4/12] LSTM + Attention")
    lstm_attn, model, info = compute_lstm_attention_attribution(
        train_journeys, method="attention", epochs=30,
    )
    results.append(lstm_attn)
    lstm_loo, _, _ = compute_lstm_attention_attribution(
        train_journeys, method="loo", model=model, training_info=info,
    )
    results.append(lstm_loo)

    logger.info("[5/12] Transformer")
    tf_result, _, _ = compute_transformer_attribution(train_journeys, epochs=30)
    results.append(tf_result)

    logger.info("[6/12] Incremental Shapley")
    results.append(compute_incremental_shapley(train_journeys, sample_users=3000))

    logger.info("[7/12] Survival/Poisson")
    results.append(compute_survival_attribution(train_journeys))

    logger.info("[8/12] IPW")
    results.append(compute_ipw_attribution(train_journeys))

    logger.info("[9/12] Doubly Robust")
    results.append(compute_doubly_robust_attribution(train_journeys))

    logger.info("[10/12] DML")
    results.append(compute_dml_attribution(train_journeys))

    logger.info("[11/12] CAMTA")
    results.append(compute_camta_attribution(train_journeys, epochs=25))

    logger.info(f"[12/12] Done. {len(results)} methods fit on train.")
    return results


def score_test_journeys(
    test_journeys: pd.DataFrame,
    credits: Dict[str, float],
) -> pd.Series:
    """Score = Σ_TP credit[channel(TP)] per user. Returns Series indexed by user_id."""
    weights = test_journeys["channel"].astype(str).map(credits).fillna(0.0)
    scored = (
        test_journeys.assign(_w=weights)
        .groupby("user_id")["_w"]
        .sum()
    )
    return scored


def evaluate_predictions(scores: pd.Series, labels: pd.Series) -> Dict[str, float]:
    """AUC, PR-AUC, Brier (min-max scaled to [0,1]). NaN-safe."""
    s = scores.values
    y = labels.astype(int).values
    if s.max() > s.min():
        s_norm = (s - s.min()) / (s.max() - s.min())
    else:
        s_norm = np.full_like(s, 0.5, dtype=float)

    return {
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s_norm)),
    }


def plot_oos_auc(eval_df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart of OOS AUC, colored by category."""
    df = eval_df.sort_values("auc", ascending=False)
    colors = [CATEGORY_COLORS.get(METHOD_CATEGORIES.get(m, ""), "#999") for m in df["method"]]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(range(len(df)), df["auc"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"], fontsize=9)
    ax.set_xlabel("Out-of-Sample AUC (test users)", fontsize=12)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    ax.set_title("Experiment 08: OOS Predictive AUC by Method", fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0.45, max(0.85, float(df["auc"].max()) + 0.02))
    for bar, val in zip(bars, df["auc"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    legend.append(plt.Line2D([0], [0], color="gray", linestyle="--", label="random"))
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_oos_auc_bar.png", dpi=150)
    plt.close()


def plot_gt_mae_vs_oos_auc(eval_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: GT-MAE × OOS-AUC. Negative slope = sim benchmark validates real-world."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in eval_df.iterrows():
        cat = METHOD_CATEGORIES.get(row["method"], "")
        color = CATEGORY_COLORS.get(cat, "#999")
        ax.scatter(row["gt_mae"], row["auc"], c=color, s=140,
                   edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(row["method"], (row["gt_mae"], row["auc"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Correlation annotation
    if len(eval_df) >= 3:
        corr = eval_df[["gt_mae", "auc"]].corr().iloc[0, 1]
        ax.text(
            0.02, 0.98, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.set_xlabel("GT-MAE (lower = closer to truth)", fontsize=12)
    ax.set_ylabel("OOS AUC (higher = better generalization)", fontsize=12)
    ax.set_title(
        "Sim-Benchmark vs Real-World Signal\n"
        "(negative correlation → GT-MAE predicts deployable performance)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend, loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_gt_mae_vs_oos_auc.png", dpi=150)
    plt.close()


def plot_calibration_top3(
    test_journeys: pd.DataFrame,
    test_labels: pd.Series,
    method_credits: Dict[str, Dict[str, float]],
    top_methods: List[str],
    output_dir: str,
) -> None:
    """Calibration curve (decile binning) for top-3 methods by AUC."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for m in top_methods:
        scores = score_test_journeys(test_journeys, method_credits[m])
        df = pd.DataFrame({"score": scores, "y": test_labels.reindex(scores.index).values})
        df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
        cal = df.groupby("decile").agg(
            mean_score=("score", "mean"),
            cvr=("y", "mean"),
            n=("y", "size"),
        )
        # Min-max within method for x-axis comparability
        if cal["mean_score"].max() > cal["mean_score"].min():
            cal["x"] = (
                (cal["mean_score"] - cal["mean_score"].min())
                / (cal["mean_score"].max() - cal["mean_score"].min())
            )
        else:
            cal["x"] = 0.5
        ax.plot(cal["x"], cal["cvr"], marker="o", label=m, linewidth=2)

    ax.set_xlabel("Score decile (min-max normalized)", fontsize=11)
    ax.set_ylabel("Observed conversion rate", fontsize=11)
    ax.set_title("Calibration — Top-3 OOS AUC Methods", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_calibration_top3.png", dpi=150)
    plt.close()


def run_experiment_08(
    data_dir: str = "data/simulation",
    output_dir: str = "results/part1",
    train_frac: float = 0.8,
    split_seed: int = 42,
) -> pd.DataFrame:
    """Run Experiment 08: out-of-sample predictive validation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    journeys = pd.read_parquet(f"{data_dir}/journeys.parquet")
    with open(f"{data_dir}/ground_truth.json") as f:
        gt = json.load(f)
    gt_a = gt["ground_truth_A"]["channel_credits"]

    train, test = split_users(journeys, train_frac, split_seed)
    logger.info(f"Split: train={train['user_id'].nunique()} users, "
                f"test={test['user_id'].nunique()} users")

    test_labels = test.groupby("user_id")["converted"].first()
    test_conv_rate = float(test_labels.mean())
    logger.info(f"Test conversion rate: {test_conv_rate:.4f}")

    results = fit_all_methods(train)

    rows = []
    method_credits: Dict[str, Dict[str, float]] = {}
    for r in results:
        scores = score_test_journeys(test, r.channel_credits)
        labels = test_labels.reindex(scores.index)
        metrics = evaluate_predictions(scores, labels)
        rows.append({
            "method": r.method,
            "category": METHOD_CATEGORIES.get(r.method, "Unknown"),
            "auc": metrics["auc"],
            "pr_auc": metrics["pr_auc"],
            "brier": metrics["brier"],
            "gt_mae": compute_mae(r.channel_credits, gt_a),
            "gt_tau": compute_kendall_tau(r.channel_credits, gt_a),
        })
        method_credits[r.method] = r.channel_credits

    eval_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)

    # Sanity checks
    assert (eval_df["auc"] >= 0.4).all(), "AUC < 0.4 — sign error?"
    assert eval_df["auc"].max() >= 0.6, "Top method AUC too low — pipeline issue"

    eval_df.to_csv(output_path / "08_predictive_validation.csv", index=False)

    plot_oos_auc(eval_df, str(output_path))
    plot_gt_mae_vs_oos_auc(eval_df, str(output_path))
    plot_calibration_top3(
        test, test_labels, method_credits,
        top_methods=eval_df["method"].head(3).tolist(),
        output_dir=str(output_path),
    )

    print(f"\n{'='*80}")
    print("Experiment 08: Out-of-Sample Predictive Validation")
    print(f"{'='*80}")
    print(f"Test users: {test['user_id'].nunique():,}, conv rate: {test_conv_rate:.4f}")
    print(f"\n{'Method':<35s} {'AUC':>7s} {'PR-AUC':>8s} {'Brier':>8s} {'GT-MAE':>8s}")
    print("-" * 70)
    for _, row in eval_df.iterrows():
        print(f"{row['method']:<35s} {row['auc']:>7.4f} {row['pr_auc']:>8.4f} "
              f"{row['brier']:>8.4f} {row['gt_mae']:>8.4f}")

    corr = eval_df[["gt_mae", "auc"]].corr().iloc[0, 1]
    print(f"\nGT-MAE vs OOS-AUC Pearson correlation: {corr:.4f}")
    print(f"  (negative → sim benchmark validates real-world performance)")

    return eval_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    run_experiment_08()
