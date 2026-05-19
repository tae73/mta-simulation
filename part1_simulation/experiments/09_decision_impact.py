"""Experiment 09 — Decision Impact: Expected Revenue Lift from Reallocation.

Real-world validation: a method's allocation MAE is hard to interpret;
"how much extra revenue do I get if I follow this attribution?" is the
question stakeholders actually ask.

Hypothesis: methods with low allocation MAE (Exp 07) also produce high
expected revenue lift, but the relationship is not perfectly monotonic
because lift weights allocations by channel efficiency. Methods that put
mass on a high-efficiency channel (e.g., Email) win disproportionately.

CAVEAT (interpretation, not a bug): ``revenue_lift_pct`` and
``allocation_mae`` are BOTH deterministic functions of the same
``derive_method_allocation`` output (credit/cost share). Their reported
correlation is therefore partly definitional/mechanical — it quantifies how
allocation error translates into the efficiency-weighted revenue objective,
NOT an independent empirical validation. The non-mechanical content is the
*shape* of that translation (efficiency weighting, the GT-optimal vs pure-LP
ceiling gap), not the existence of correlation itself. Treat the scatter as
"how much does allocation error cost in revenue units", not as evidence that
allocation MAE predicts an out-of-sample outcome.

Setup (deterministic — closed-form Approach A):
    Under Linear Response (Approach A), expected paid conversions equal
        Σ_k spend_k × efficiency_k
    where efficiency_k = β_k · E[f_k] / cost_per_TP_k. Forward simulating
    a fresh user pool under each allocation reduces to this closed form,
    so we evaluate analytically (more rigorous, no MC noise).

    Reference points (same units, expected paid revenue $):
        baseline_rev   = current observed spend share × efficiencies × budget
        gt_optimal_rev = GT-optimal allocation (proportional-to-efficiency rule;
                         optimal under a concave/saturation response, sub-optimal
                         under pure linear)
        ceiling_rev    = pure linear LP optimum = all-budget-on-best-channel

    Reported metric:
        revenue_lift_pct = method_rev / baseline_rev - 1.0   (vs current spending)
        revenue_vs_gt    = method_rev / gt_optimal_rev       (vs project GT)

    Note: a method can exceed gt_optimal_rev under pure linear response by
    over-concentrating on the most efficient channel (Email). This is reported
    transparently rather than capped, since it reveals which methods would
    behave aggressively in a saturation-free regime.

Outputs:
    - results/part1/09_decision_impact.csv
    - results/part1/09_lift_bar.png
    - results/part1/09_alloc_mae_vs_lift.png
    - results/part1/09_revenue_waterfall_top.png
"""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part1_simulation.config_loader import load_budget_config
from part1_simulation.experiments._common import (
    CATEGORY_COLORS_LEGACY as CATEGORY_COLORS,
    METHOD_CATEGORIES_LEGACY as METHOD_CATEGORIES,
    load_journeys_and_gt,
    prepare_output_dir,
    reconstruct_credits_from_eval,
    setup_experiment_logging,
)

logger = logging.getLogger(__name__)


def compute_baseline_spend_share(journeys: pd.DataFrame, paid_channels: List[str]) -> Dict[str, float]:
    """Baseline allocation = observed paid spend share."""
    paid_mask = journeys["channel"].isin(paid_channels)
    spend_by_ch = (
        journeys.loc[paid_mask]
        .groupby("channel", observed=True)["touchpoint_cost"]
        .sum()
    )
    total = spend_by_ch.sum()
    if total <= 0:
        n = len(paid_channels)
        return {ch: 1.0 / n for ch in paid_channels}
    return {ch: float(spend_by_ch.get(ch, 0.0) / total) for ch in paid_channels}


def expected_paid_conversions(
    allocation: Dict[str, float],
    efficiency: Dict[str, float],
    total_budget: float,
) -> float:
    """Approach A: Σ_k (alloc_k × budget) × efficiency_k."""
    return float(total_budget * sum(
        allocation.get(ch, 0.0) * eff for ch, eff in efficiency.items()
    ))


def derive_method_allocation(
    credits: Dict[str, float],
    cost_per_tp: Dict[str, float],
    paid_channels: List[str],
) -> Dict[str, float]:
    """Linear-Response derived allocation: prop to credit / cost (paid only)."""
    eff = {}
    for ch in paid_channels:
        c = cost_per_tp.get(ch, 0.0)
        if c > 0:
            eff[ch] = credits.get(ch, 0.0) / c
    total = sum(eff.values())
    if total <= 0:
        return {ch: 1.0 / len(paid_channels) for ch in paid_channels}
    return {ch: v / total for ch, v in eff.items()}


def plot_lift_bar(
    df: pd.DataFrame,
    baseline_rev: float,
    gt_optimal_rev: float,
    ceiling_rev: float,
    output_dir: str,
) -> None:
    """Horizontal bar of revenue_lift_pct per method."""
    df_sorted = df.sort_values("revenue_lift_pct", ascending=False)
    colors = [
        CATEGORY_COLORS.get(METHOD_CATEGORIES.get(m, ""), "#999")
        for m in df_sorted["method"]
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(range(len(df_sorted)), df_sorted["revenue_lift_pct"],
                   color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["method"], fontsize=9)
    ax.set_xlabel("Revenue Lift % vs current baseline spend (paid channels only)",
                  fontsize=11)

    gt_lift_pct = gt_optimal_rev / baseline_rev - 1.0
    ceiling_lift_pct = ceiling_rev / baseline_rev - 1.0
    ax.axvline(x=gt_lift_pct, color="green", linestyle="--", alpha=0.7,
               label=f"GT-optimal (proportional rule, +{gt_lift_pct:.0%})")
    ax.axvline(x=ceiling_lift_pct, color="purple", linestyle="--", alpha=0.5,
               label=f"Linear LP ceiling (all-on-Email, +{ceiling_lift_pct:.0%})")
    ax.axvline(x=0.0, color="red", linestyle="--", alpha=0.4, label="baseline (0%)")
    ax.invert_yaxis()
    for bar, val in zip(bars, df_sorted["revenue_lift_pct"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"+{val:.0%}", va="center", fontsize=8)
    ax.set_title(
        f"Experiment 09: Decision Impact — Expected Revenue Lift vs Baseline\n"
        f"(baseline=${baseline_rev:,.0f}, GT-optimal=${gt_optimal_rev:,.0f}, "
        f"linear ceiling=${ceiling_rev:,.0f})",
        fontsize=11,
    )

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    legend.extend([
        plt.Line2D([0], [0], color="green", linestyle="--", label="GT-optimal"),
        plt.Line2D([0], [0], color="purple", linestyle="--", label="LP ceiling"),
    ])
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_lift_bar.png", dpi=150)
    plt.close()


def plot_alloc_mae_vs_lift(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: allocation_mae × revenue_lift_pct."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        cat = METHOD_CATEGORIES.get(row["method"], "")
        color = CATEGORY_COLORS.get(cat, "#999")
        ax.scatter(row["allocation_mae"], row["revenue_lift_pct"],
                   c=color, s=140, edgecolor="black", linewidth=0.7, zorder=3)
        ax.annotate(row["method"], (row["allocation_mae"], row["revenue_lift_pct"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

    if len(df) >= 3:
        corr = df[["allocation_mae", "revenue_lift_pct"]].corr().iloc[0, 1]
        ax.text(
            0.02, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.4, label="baseline (no change)")
    ax.set_xlabel("Allocation MAE vs GT-optimal (lower = closer to project GT rule)",
                  fontsize=12)
    ax.set_ylabel("Revenue Lift % vs baseline", fontsize=12)
    ax.set_title(
        "Allocation MAE (benchmark metric) vs Revenue Lift (operational metric)\n"
        "(when negatively correlated, sim benchmark predicts deployment outcome)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_alloc_mae_vs_lift.png", dpi=150)
    plt.close()


def plot_revenue_waterfall(
    df: pd.DataFrame,
    baseline_rev: float,
    gt_optimal_rev: float,
    ceiling_rev: float,
    output_dir: str,
    top_k: int = 5,
    bottom_k: int = 3,
) -> None:
    """Bar chart of expected paid revenue: baseline → top/bottom methods → ceilings."""
    df_sorted = df.sort_values("revenue_lift_pct", ascending=False)
    top = df_sorted.head(top_k)
    bottom = df_sorted.tail(bottom_k)

    labels = (
        ["Baseline"]
        + top["method"].tolist()
        + bottom["method"].tolist()
        + ["GT-Optimal\n(proportional)", "Linear LP\nceiling"]
    )
    values = (
        [baseline_rev]
        + top["expected_paid_revenue"].tolist()
        + bottom["expected_paid_revenue"].tolist()
        + [gt_optimal_rev, ceiling_rev]
    )
    colors = (
        ["#888"] + ["#27AE60"] * top_k + ["#E74C3C"] * bottom_k
        + ["#2C3E50", "#9B59B6"]
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Expected Paid Revenue ($, log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(
        "Expected Paid Revenue: Baseline → Top/Bottom Methods → GT/LP Ceilings",
        fontsize=12,
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.10,
                f"${val:,.0f}", ha="center", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/09_revenue_waterfall_top.png", dpi=150)
    plt.close()


def run_experiment_09(
    data_dir: str = "data/simulation",
    output_dir: str = "results/part1",
    eval_csv: str = "results/part1/01_method_accuracy.csv",
) -> pd.DataFrame:
    """Run Experiment 09: deterministic decision-impact analysis."""
    output_path = prepare_output_dir(output_dir)

    journeys, gt, gt_a = load_journeys_and_gt(data_dir)
    gt_budget = gt["ground_truth_budget"]
    gt_optimal = gt_budget["optimal_allocation_fraction"]
    efficiency = gt_budget["channel_efficiency"]
    total_budget = float(gt_budget["total_budget"])
    revenue_per_conv = float(gt_budget["revenue_per_conversion"])

    paid_channels = sorted(efficiency.keys())
    logger.info(f"Paid channels: {paid_channels}")

    budget_config = load_budget_config()
    cost_per_tp = {
        cd.channel_name: cd.base_cost_per_touchpoint
        for cd in budget_config.cost_defs
    }

    # Reference points
    baseline_alloc = compute_baseline_spend_share(journeys, paid_channels)
    baseline_conv = expected_paid_conversions(baseline_alloc, efficiency, total_budget)
    gt_optimal_conv = expected_paid_conversions(gt_optimal, efficiency, total_budget)
    # Pure linear LP ceiling: all-on-best-channel
    best_ch = max(efficiency, key=efficiency.get)
    lp_ceiling_alloc = {ch: (1.0 if ch == best_ch else 0.0) for ch in paid_channels}
    ceiling_conv = expected_paid_conversions(lp_ceiling_alloc, efficiency, total_budget)

    baseline_rev = baseline_conv * revenue_per_conv
    gt_optimal_rev = gt_optimal_conv * revenue_per_conv
    ceiling_rev = ceiling_conv * revenue_per_conv

    logger.info(f"Baseline ($current spend): {baseline_conv:.1f} conv "
                f"(${baseline_rev:,.0f})")
    logger.info(f"GT-optimal (proportional rule): {gt_optimal_conv:.1f} conv "
                f"(${gt_optimal_rev:,.0f})")
    logger.info(f"Linear LP ceiling (all-on-{best_ch}): {ceiling_conv:.1f} conv "
                f"(${ceiling_rev:,.0f})")
    logger.info(f"Baseline allocation share: {baseline_alloc}")

    # Reconstruct each method's credits → allocation → expected revenue
    eval_df = pd.read_csv(eval_csv)
    attr_results = reconstruct_credits_from_eval(eval_df, gt_a)

    rows = []
    for r in attr_results:
        method_alloc = derive_method_allocation(
            r.channel_credits, cost_per_tp, paid_channels,
        )
        method_conv = expected_paid_conversions(method_alloc, efficiency, total_budget)
        method_rev = method_conv * revenue_per_conv

        revenue_lift_pct = (method_rev / baseline_rev - 1.0) if baseline_rev > 0 else 0.0
        revenue_vs_gt = method_rev / gt_optimal_rev if gt_optimal_rev > 0 else 0.0

        alloc_mae = float(np.mean([
            abs(method_alloc.get(ch, 0.0) - gt_optimal[ch])
            for ch in paid_channels
        ]))

        row = {
            "method": r.method,
            "category": METHOD_CATEGORIES.get(r.method, "Unknown"),
            "expected_paid_conversions": method_conv,
            "expected_paid_revenue": method_rev,
            "absolute_lift_vs_baseline": method_rev - baseline_rev,
            "revenue_lift_pct": revenue_lift_pct,
            "revenue_vs_gt": revenue_vs_gt,
            "allocation_mae": alloc_mae,
        }
        for ch in paid_channels:
            row[f"alloc_{ch}"] = method_alloc.get(ch, 0.0)
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values("revenue_lift_pct", ascending=False)
        .reset_index(drop=True)
    )

    # Sanity checks
    assert gt_optimal_rev >= baseline_rev, "GT-optimal must beat baseline"
    assert ceiling_rev >= gt_optimal_rev, "LP ceiling must beat GT-optimal"
    assert (df["expected_paid_revenue"] <= ceiling_rev + 1e-6).all(), \
        "No method should exceed pure LP ceiling"
    assert (df["allocation_mae"] >= 0).all()

    df.to_csv(output_path / "09_decision_impact.csv", index=False)

    plot_lift_bar(df, baseline_rev, gt_optimal_rev, ceiling_rev, str(output_path))
    plot_alloc_mae_vs_lift(df, str(output_path))
    plot_revenue_waterfall(df, baseline_rev, gt_optimal_rev, ceiling_rev, str(output_path))

    print(f"\n{'='*80}")
    print("Experiment 09: Decision Impact — Expected Revenue Lift")
    print(f"{'='*80}")
    print(f"Total budget:                    ${total_budget:,.0f}")
    print(f"Baseline (current spend):        ${baseline_rev:,.0f}")
    print(f"GT-optimal (proportional rule):  ${gt_optimal_rev:,.0f}  "
          f"(+{gt_optimal_rev / baseline_rev - 1:.0%})")
    print(f"Linear LP ceiling (all-on-{best_ch}): ${ceiling_rev:,.0f}  "
          f"(+{ceiling_rev / baseline_rev - 1:.0%})\n")
    print(f"{'Method':<35s} {'Lift %':>10s} {'vs GT':>9s} {'Alloc MAE':>10s}")
    print("-" * 68)
    for _, row in df.iterrows():
        print(f"{row['method']:<35s} {row['revenue_lift_pct']:>+9.1%} "
              f"{row['revenue_vs_gt']:>8.2f}x "
              f"{row['allocation_mae']:>10.4f}")

    corr = df[["allocation_mae", "revenue_lift_pct"]].corr().iloc[0, 1]
    print(f"\nAllocation MAE vs Revenue Lift % correlation: {corr:.4f}")

    return df


if __name__ == "__main__":
    setup_experiment_logging()
    run_experiment_09()
