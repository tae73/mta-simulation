"""Experiment 11 — Convergent Validity (Cross-Method Agreement).

Real-world validation: when ground truth is unavailable, agreement across
heterogeneous methods is the primary signal of robust insight.

Hypothesis: A consensus channel ranking (mean rank across all 18 methods)
approximates ground truth as well as the single best individual method.

Setup:
    - Reconstruct each method's channel_credits from `01_method_accuracy.csv`
      (credit_k = gt_a_k + bias_k, then re-normalize).
    - Compute pairwise Kendall's Tau matrix (method × method).
    - Compute per-channel rank disagreement (std of rank across methods).
    - Compute consensus rank (mean rank), compare vs GT-A ranking.

Inputs: results/part1/01_method_accuracy.csv, data/simulation/ground_truth.json
Outputs:
    - results/part1/11_convergent_validity.csv (per-method consensus metrics)
    - results/part1/11_tau_matrix.csv (pairwise tau matrix)
    - results/part1/11_method_agreement_heatmap.png
    - results/part1/11_channel_disagreement.png
    - results/part1/11_consensus_vs_gt.png
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
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import linkage, leaves_list

from part1_simulation import CHANNEL_NAMES
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae

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


def reconstruct_credits(eval_df: pd.DataFrame, gt_a: Dict[str, float]) -> pd.DataFrame:
    """Reconstruct channel credit matrix (method × channel) from bias columns.

    credit_k = max(0, gt_a_k + bias_k), then normalize so each row sums to 1.0.
    """
    bias_cols = [c for c in eval_df.columns if c.startswith("bias_")]
    channels = [c.replace("bias_", "") for c in bias_cols]

    rows = []
    for _, row in eval_df.iterrows():
        credits = {ch: max(0.0, gt_a.get(ch, 0.0) + row[f"bias_{ch}"]) for ch in channels}
        total = sum(credits.values()) or 1.0
        credits = {ch: v / total for ch, v in credits.items()}
        rows.append({"method": row["method"], **credits})

    return pd.DataFrame(rows).set_index("method")[sorted(channels)]


def compute_pairwise_tau(credit_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Kendall's Tau between every method pair."""
    methods = credit_matrix.index.tolist()
    n = len(methods)
    tau_mat = np.eye(n)

    for i, m_i in enumerate(methods):
        for j in range(i + 1, n):
            m_j = methods[j]
            tau, _ = scipy_stats.kendalltau(
                credit_matrix.loc[m_i].values,
                credit_matrix.loc[m_j].values,
            )
            tau_val = 0.0 if np.isnan(tau) else float(tau)
            tau_mat[i, j] = tau_mat[j, i] = tau_val

    return pd.DataFrame(tau_mat, index=methods, columns=methods)


def compute_channel_disagreement(credit_matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-channel rank std across methods. Higher = more contested."""
    rank_matrix = credit_matrix.rank(axis=1, ascending=False)
    return pd.DataFrame({
        "channel": rank_matrix.columns,
        "rank_std": rank_matrix.std(axis=0).values,
        "rank_mean": rank_matrix.mean(axis=0).values,
        "credit_mean": credit_matrix.mean(axis=0).values,
        "credit_std": credit_matrix.std(axis=0).values,
    })


def compute_consensus(
    credit_matrix: pd.DataFrame,
    gt_a: Dict[str, float],
) -> Tuple[Dict[str, float], float, float]:
    """Consensus = mean rank → flip to credits via softmax-like normalization.

    Returns:
        (consensus_credits, consensus_tau_vs_gt, consensus_mae_vs_gt)
    """
    rank_matrix = credit_matrix.rank(axis=1, ascending=False)
    mean_rank = rank_matrix.mean(axis=0)  # lower rank = better channel

    # Convert mean rank → consensus "score" (inverted), then normalize
    n_ch = len(mean_rank)
    raw_score = (n_ch + 1 - mean_rank).clip(lower=0)
    consensus_credits = (raw_score / raw_score.sum()).to_dict()

    tau_vs_gt = compute_kendall_tau(consensus_credits, gt_a)
    mae_vs_gt = compute_mae(consensus_credits, gt_a)
    return consensus_credits, tau_vs_gt, mae_vs_gt


def plot_tau_heatmap(tau_df: pd.DataFrame, output_dir: str) -> None:
    """Hierarchically clustered Kendall's Tau heatmap."""
    distance = 1 - tau_df.values
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2

    condensed = distance[np.triu_indices_from(distance, k=1)]
    link = linkage(condensed, method="average")
    order = leaves_list(link)

    methods_sorted = [tau_df.index[i] for i in order]
    tau_sorted = tau_df.loc[methods_sorted, methods_sorted]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(tau_sorted.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(methods_sorted)))
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_xticklabels(methods_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(methods_sorted, fontsize=8)

    for i in range(len(methods_sorted)):
        for j in range(len(methods_sorted)):
            v = tau_sorted.values[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=6.5)

    plt.colorbar(im, ax=ax, label="Kendall's Tau")
    ax.set_title(
        "Experiment 11: Pairwise Cross-Method Agreement\n"
        "(hierarchically clustered; tau≈1 → similar channel rankings)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_method_agreement_heatmap.png", dpi=150)
    plt.close()


def plot_channel_disagreement(disagreement_df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart: per-channel rank std across methods."""
    df_sorted = disagreement_df.sort_values("rank_std", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        df_sorted["channel"], df_sorted["rank_std"],
        color="#FF6B6B", edgecolor="white",
    )
    for bar, val in zip(bars, df_sorted["rank_std"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.2f}", ha="center", fontsize=10)

    ax.set_ylabel("Rank Std across 18 methods", fontsize=12)
    ax.set_title(
        "Per-Channel Rank Disagreement\n"
        "(higher = methods disagree on this channel's rank)",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_channel_disagreement.png", dpi=150)
    plt.close()


def plot_consensus_vs_gt(
    credit_matrix: pd.DataFrame,
    gt_a: Dict[str, float],
    consensus_credits: Dict[str, float],
    consensus_tau: float,
    consensus_mae: float,
    output_dir: str,
) -> None:
    """Two-panel: (left) consensus rank vs GT rank, (right) tau histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    channels = sorted(gt_a.keys())
    gt_ranks = pd.Series(gt_a)[channels].rank(ascending=False)
    consensus_ranks = pd.Series(consensus_credits)[channels].rank(ascending=False)

    ax1.scatter(gt_ranks, consensus_ranks, s=200, c="#3498DB",
                edgecolor="black", zorder=3)
    for ch in channels:
        ax1.annotate(
            ch, (gt_ranks[ch], consensus_ranks[ch]),
            textcoords="offset points", xytext=(8, 8), fontsize=10,
        )
    n = len(channels)
    ax1.plot([1, n], [1, n], "k--", alpha=0.4, label="perfect agreement")
    ax1.set_xlabel("Ground Truth A rank", fontsize=11)
    ax1.set_ylabel("Consensus rank (mean across 18 methods)", fontsize=11)
    ax1.set_title(
        f"Consensus vs Ground Truth\nTau={consensus_tau:.3f}, MAE={consensus_mae:.4f}",
        fontsize=12,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.invert_xaxis()

    # Per-method tau vs GT (compare consensus to individual methods)
    method_taus = []
    for m in credit_matrix.index:
        method_credits = credit_matrix.loc[m].to_dict()
        method_taus.append((m, compute_kendall_tau(method_credits, gt_a)))
    method_taus.sort(key=lambda x: x[1], reverse=True)

    methods, taus = zip(*method_taus)
    colors = ["#FF6B6B" if m == "Consensus (mean rank)" else "#4ECDC4" for m in methods]
    y_pos = np.arange(len(methods))
    ax2.barh(y_pos, taus, color=colors, edgecolor="white")
    ax2.axvline(x=consensus_tau, color="red", linestyle="--", linewidth=2,
                label=f"Consensus tau={consensus_tau:.3f}")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=8)
    ax2.set_xlabel("Kendall's Tau vs GT-A", fontsize=11)
    ax2.set_title("Individual methods vs Consensus", fontsize=12)
    ax2.invert_yaxis()
    ax2.legend(loc="lower right")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/11_consensus_vs_gt.png", dpi=150)
    plt.close()


def run_experiment_11(
    eval_csv: str = "results/part1/01_method_accuracy.csv",
    gt_path: str = "data/simulation/ground_truth.json",
    output_dir: str = "results/part1",
) -> pd.DataFrame:
    """Run Experiment 11: convergent validity analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(eval_csv)
    with open(gt_path) as f:
        gt = json.load(f)
    gt_a = gt["ground_truth_A"]["channel_credits"]

    logger.info(f"Loaded {len(eval_df)} methods from {eval_csv}")

    credit_matrix = reconstruct_credits(eval_df, gt_a)
    tau_df = compute_pairwise_tau(credit_matrix)
    disagreement_df = compute_channel_disagreement(credit_matrix)
    consensus_credits, consensus_tau, consensus_mae = compute_consensus(credit_matrix, gt_a)

    # Per-method summary
    summary_rows = []
    for m in credit_matrix.index:
        method_credits = credit_matrix.loc[m].to_dict()
        # Mean tau against all other methods (community agreement)
        peer_taus = [tau_df.loc[m, n] for n in tau_df.columns if n != m]
        summary_rows.append({
            "method": m,
            "category": METHOD_CATEGORIES.get(m, "Unknown"),
            "tau_vs_gt": compute_kendall_tau(method_credits, gt_a),
            "mae_vs_gt": compute_mae(method_credits, gt_a),
            "mean_tau_vs_peers": float(np.mean(peer_taus)),
            "min_tau_vs_peers": float(np.min(peer_taus)),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("tau_vs_gt", ascending=False)

    # Add consensus row
    consensus_row = pd.DataFrame([{
        "method": "Consensus (mean rank)",
        "category": "Aggregate",
        "tau_vs_gt": consensus_tau,
        "mae_vs_gt": consensus_mae,
        "mean_tau_vs_peers": float("nan"),
        "min_tau_vs_peers": float("nan"),
    }])
    summary_df = pd.concat([consensus_row, summary_df], ignore_index=True)

    # Sanity checks
    assert (np.diag(tau_df.values) == 1.0).all(), "Tau diagonal must be 1.0"
    assert np.allclose(tau_df.values, tau_df.values.T), "Tau matrix must be symmetric"

    # Save
    summary_df.to_csv(output_path / "11_convergent_validity.csv", index=False)
    tau_df.to_csv(output_path / "11_tau_matrix.csv")
    disagreement_df.to_csv(output_path / "11_channel_disagreement.csv", index=False)

    # Plot
    plot_tau_heatmap(tau_df, str(output_path))
    plot_channel_disagreement(disagreement_df, str(output_path))
    plot_consensus_vs_gt(
        credit_matrix, gt_a, consensus_credits, consensus_tau, consensus_mae,
        str(output_path),
    )

    # Print summary
    print(f"\n{'='*80}")
    print("Experiment 11: Convergent Validity (GT-Free Cross-Method Agreement)")
    print(f"{'='*80}")
    print(f"\nConsensus rank vs GT-A:  tau={consensus_tau:.4f}, MAE={consensus_mae:.4f}")
    print(f"Best individual method:  tau={summary_df.iloc[1]['tau_vs_gt']:.4f} "
          f"({summary_df.iloc[1]['method']})")

    print(f"\nTop-5 disagreement channels:")
    for _, row in disagreement_df.sort_values("rank_std", ascending=False).head(5).iterrows():
        print(f"  {row['channel']:20s}: rank_std={row['rank_std']:.2f}, "
              f"credit={row['credit_mean']:.3f}±{row['credit_std']:.3f}")

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")
    run_experiment_11()
