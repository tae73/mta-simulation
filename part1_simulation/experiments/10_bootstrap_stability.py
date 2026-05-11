"""Experiment 10 — Bootstrap Stability.

Real-world validation: practitioners run a method ONCE on the available data
and act on its output. Two methods with similar mean accuracy but different
finite-sample variability impose very different decision risk.

Hypothesis: causal/structural methods (Survival/Poisson, Shapley model-based)
have lower bootstrap CV than DL methods (LSTM, Transformer) at comparable
sample size, because they exploit DGP structure rather than learning a flexible
function class from limited data.

Setup:
    - For each method, draw N bootstrap samples (with replacement) of users.
      Sample size = 5K users (default) for tractable runtime.
    - Tier-1 (light: rule-based, Markov, Shapley): N=100 bootstraps
    - Tier-2 (DL: LSTM, Transformer, Incremental Shapley, IPW, DR): N=20
    - Tier-3 (heavy: DML, CAMTA, Survival/Poisson): N=5
    (User-approved N values from plan are practical reductions; CV
    estimates remain meaningful at these sample counts, with wider
    uncertainty bands for heavy-tier methods documented in the report.)
    - For each (method, channel), compute:
        bootstrap mean, std, CV = std / mean
        95% CI width = q97.5 - q2.5

Outputs:
    - results/part1/10_bootstrap_stability.csv (per method × channel)
    - results/part1/10_method_cv_bar.png
    - results/part1/10_bootstrap_violin.png
    - results/part1/10_ci_width_heatmap.png
"""

import logging
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation import AttributionResult, CHANNEL_NAMES
from part1_simulation.experiments._common import (
    CATEGORY_COLORS,
    METHOD_CATEGORIES,
    prepare_output_dir,
    setup_experiment_logging,
)
from part1_simulation.models.causal.camta import compute_camta_attribution
from part1_simulation.models.causal.dml import compute_dml_attribution
from part1_simulation.models.causal.incremental_shapley import compute_incremental_shapley
from part1_simulation.models.causal.propensity import (
    compute_doubly_robust_attribution,
    compute_ipw_attribution,
)
from part1_simulation.models.causal.survival_attribution import compute_survival_attribution
from part1_simulation.models.lstm_attention import compute_lstm_attention_attribution
from part1_simulation.models.markov import compute_markov_attribution
from part1_simulation.models.rule_based import (
    compute_first_click,
    compute_last_click,
    compute_linear,
    compute_position_based,
    compute_time_decay,
)
from part1_simulation.models.shapley import compute_shapley_model_based
from part1_simulation.models.transformer import compute_transformer_attribution

logger = logging.getLogger(__name__)


# (method_name, callable, tier)  — tier determines bootstrap N
METHOD_REGISTRY: List[Tuple[str, Callable[[pd.DataFrame], AttributionResult], str]] = [
    ("Last Click",                compute_last_click,                          "light"),
    ("First Click",               compute_first_click,                         "light"),
    ("Linear",                    compute_linear,                              "light"),
    ("Time Decay (7.0d)",         compute_time_decay,                          "light"),
    ("Position-Based (40%/40%)",  compute_position_based,                      "light"),
    ("Markov (order=1)",          lambda j: compute_markov_attribution(j, 1), "light"),
    ("Markov (order=2)",          lambda j: compute_markov_attribution(j, 2), "light"),
    ("Shapley (model-based)",     compute_shapley_model_based,                 "light"),
    ("LSTM+Attention (attn weights)",
        lambda j: compute_lstm_attention_attribution(j, method="attention", epochs=15)[0],
        "medium"),
    ("Transformer (2L/2H)",
        lambda j: compute_transformer_attribution(j, epochs=15)[0],
        "medium"),
    ("Incremental Shapley",
        lambda j: compute_incremental_shapley(j, sample_users=1500),
        "medium"),
    ("IPW",                       compute_ipw_attribution,                     "medium"),
    ("Doubly Robust",             compute_doubly_robust_attribution,           "medium"),
    ("Survival/Poisson (AICPE)",  compute_survival_attribution,                "heavy"),
    ("DML",                       compute_dml_attribution,                     "heavy"),
    ("CAMTA (Causal Attention)",
        lambda j: compute_camta_attribution(j, epochs=15),
        "heavy"),
]

TIER_N = {"light": 100, "medium": 20, "heavy": 5}

def bootstrap_users(
    journeys: pd.DataFrame,
    n_users: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample n_users with replacement, return their full journeys."""
    all_uids = journeys["user_id"].unique()
    sampled = rng.choice(all_uids, size=n_users, replace=True)

    # Map sampled uids to journeys (with duplication via pandas merge)
    user_journeys: Dict[int, pd.DataFrame] = {
        uid: g for uid, g in journeys.groupby("user_id", sort=False)
    }

    # Re-id duplicates so models that group by user_id see distinct units
    parts = []
    for new_uid, orig_uid in enumerate(sampled):
        g = user_journeys[orig_uid].copy()
        g["user_id"] = new_uid
        parts.append(g)

    return pd.concat(parts, ignore_index=True)


def run_bootstrap_for_method(
    method_name: str,
    method_fn: Callable[[pd.DataFrame], AttributionResult],
    journeys: pd.DataFrame,
    n_bootstrap: int,
    sample_size: int,
    base_seed: int,
) -> pd.DataFrame:
    """Run one method on N bootstrap samples. Returns long-format credit DataFrame."""
    rows = []
    start = time.time()

    for b in range(n_bootstrap):
        rng = np.random.default_rng(base_seed + b)
        sample = bootstrap_users(journeys, sample_size, rng)
        try:
            result = method_fn(sample)
        except Exception as e:
            logger.warning(f"  {method_name} boot {b} failed: {e}")
            continue

        for ch, credit in result.channel_credits.items():
            rows.append({
                "method": method_name,
                "bootstrap_idx": b,
                "channel": ch,
                "credit": float(credit),
            })

        if (b + 1) % max(1, n_bootstrap // 5) == 0:
            elapsed = time.time() - start
            logger.info(f"    {method_name}: {b + 1}/{n_bootstrap} ({elapsed:.1f}s)")

    return pd.DataFrame(rows)


def aggregate_bootstrap(boot_df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, channel): mean, std, CV, 95% CI width."""
    summary = (
        boot_df.groupby(["method", "channel"])["credit"]
        .agg(["mean", "std", "count",
              ("q025", lambda s: s.quantile(0.025)),
              ("q975", lambda s: s.quantile(0.975))])
        .reset_index()
    )
    summary["cv"] = summary["std"] / summary["mean"].abs().replace(0, np.nan)
    summary["ci_width"] = summary["q975"] - summary["q025"]
    return summary


def plot_method_cv_bar(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Mean CV across channels per method (lower = more stable)."""
    method_cv = (
        summary_df.groupby("method")["cv"].mean()
        .reset_index()
        .sort_values("cv")
    )
    colors = [
        CATEGORY_COLORS.get(METHOD_CATEGORIES.get(m, ""), "#999")
        for m in method_cv["method"]
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(range(len(method_cv)), method_cv["cv"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(method_cv)))
    ax.set_yticklabels(method_cv["method"], fontsize=9)
    ax.set_xlabel("Mean Coefficient of Variation across channels (lower = more stable)",
                  fontsize=11)
    ax.set_title(
        "Experiment 10: Bootstrap Stability\n"
        "(per-method mean CV across 7 channels)",
        fontsize=13,
    )
    ax.invert_yaxis()
    for bar, val in zip(bars, method_cv["cv"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_method_cv_bar.png", dpi=150)
    plt.close()


def plot_bootstrap_violin(boot_df: pd.DataFrame, output_dir: str) -> None:
    """Violin plot of bootstrap distributions for key channels (paid only)."""
    paid_channels = ["Display", "Social", "Paid Search", "Email"]
    methods_show = (
        boot_df.groupby("method")["bootstrap_idx"].max()
        .sort_values().index.tolist()  # ordered, fewer-N first
    )

    fig, axes = plt.subplots(1, len(paid_channels), figsize=(5 * len(paid_channels), 7),
                             sharey=False)

    for ax, ch in zip(axes, paid_channels):
        ch_df = boot_df[boot_df["channel"] == ch]
        data_per_method = [
            ch_df[ch_df["method"] == m]["credit"].values
            for m in methods_show
        ]
        # Drop empty
        keep = [(m, d) for m, d in zip(methods_show, data_per_method) if len(d) > 0]
        if not keep:
            continue
        keep_methods, keep_data = zip(*keep)

        parts = ax.violinplot(keep_data, vert=False, showmeans=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#3498DB")
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")

        ax.set_yticks(range(1, len(keep_methods) + 1))
        ax.set_yticklabels(keep_methods, fontsize=8)
        ax.set_xlabel(f"Credit for {ch}", fontsize=10)
        ax.set_title(ch, fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Bootstrap Distribution per Method — Paid Channels",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_bootstrap_violin.png", dpi=150)
    plt.close()


def plot_ci_width_heatmap(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of 95% CI width (method × channel)."""
    pivot = summary_df.pivot(index="method", columns="channel", values="ci_width")
    pivot = pivot.reindex(columns=list(CHANNEL_NAMES))

    # Order methods by mean CI width
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean").drop(columns="_mean")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 0.05 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="95% CI width")
    ax.set_title("Bootstrap 95% CI Width — Method × Channel\n(narrower = more stable)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/10_ci_width_heatmap.png", dpi=150)
    plt.close()


def run_experiment_10(
    data_dir: str = "data/simulation",
    output_dir: str = "results/part1",
    sample_size: int = 5_000,
    base_seed: int = 42,
    methods_filter: List[str] = None,
) -> pd.DataFrame:
    """Run Experiment 10: bootstrap stability."""
    output_path = prepare_output_dir(output_dir)

    journeys = pd.read_parquet(f"{data_dir}/journeys.parquet")
    logger.info(f"Loaded {journeys['user_id'].nunique()} users")

    boot_dfs = []
    for method_name, method_fn, tier in METHOD_REGISTRY:
        if methods_filter and method_name not in methods_filter:
            continue
        n_boot = TIER_N[tier]
        logger.info(f"\n=== {method_name} (tier={tier}, N={n_boot}) ===")
        df = run_bootstrap_for_method(
            method_name, method_fn, journeys, n_boot, sample_size, base_seed,
        )
        if not df.empty:
            df["tier"] = tier
            boot_dfs.append(df)

    boot_df = pd.concat(boot_dfs, ignore_index=True)
    summary_df = aggregate_bootstrap(boot_df)
    summary_df["category"] = summary_df["method"].map(METHOD_CATEGORIES)

    # Sanity checks
    assert (summary_df["cv"].dropna() >= 0).all(), "CV must be non-negative"
    assert (summary_df["ci_width"] >= 0).all(), "CI width must be non-negative"

    boot_df.to_csv(output_path / "10_bootstrap_raw.csv", index=False)
    summary_df.to_csv(output_path / "10_bootstrap_stability.csv", index=False)

    plot_method_cv_bar(summary_df, str(output_path))
    plot_bootstrap_violin(boot_df, str(output_path))
    plot_ci_width_heatmap(summary_df, str(output_path))

    method_cv = (
        summary_df.groupby("method")["cv"].mean().sort_values()
    )
    print(f"\n{'='*80}")
    print("Experiment 10: Bootstrap Stability")
    print(f"{'='*80}")
    print(f"Sample size per bootstrap: {sample_size:,}")
    print(f"\nMean CV across channels (lower = more stable):")
    for m, cv in method_cv.items():
        print(f"  {m:<35s}: {cv:.4f}")

    return summary_df


if __name__ == "__main__":
    setup_experiment_logging(use_timestamp=True)
    run_experiment_10()
