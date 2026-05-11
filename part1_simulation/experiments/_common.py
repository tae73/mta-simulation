"""Shared utilities for part1 experiments.

Centralizes boilerplate that was previously duplicated across 11 experiment
scripts (01–11): method categorization, logging/output-dir setup, journey/GT
loading, bias-to-credits reconstruction, and the MAE/Tau scoring loop.

Experiment scripts should import the canonical constants from here rather than
redefining them; behavior is preserved exactly.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend for batch experiment runs
import pandas as pd

from part1_simulation import AttributionResult
from part1_simulation.evaluation.metrics import compute_kendall_tau, compute_mae

PathLike = Union[str, Path]


# === Canonical 5-tier method categorization ===
# Used by experiments 01, 07, 08, 10, 11. Superset of all entries each script
# references; extra keys are harmless (looked up via .get() with fallback).
METHOD_CATEGORIES: Dict[str, str] = {
    "Last Click": "Rule-based",
    "First Click": "Rule-based",
    "Linear": "Rule-based",
    "Time Decay (7.0d)": "Rule-based",
    "Position-Based (40%/40%)": "Rule-based",
    "Markov (order=1)": "Statistical",
    "Markov (order=2)": "Statistical",
    "Shapley (model-based)": "Game-theoretic",
    "Shapley (conv. rate)": "Game-theoretic",
    "LSTM+Attention (attn weights)": "Deep Learning",
    "LSTM+Attention (LOO)": "Deep Learning",
    "Transformer (2L/2H)": "Deep Learning",
    "Incremental Shapley": "Causal (incremental)",
    "Survival/Poisson (BackElim)": "Causal (incremental)",
    "Survival/Poisson (AICPE)": "Causal (incremental)",
    "Survival/Poisson (Shapley)": "Causal (incremental)",
    "IPW": "Causal (debiased)",
    "Doubly Robust": "Causal (debiased)",
    "DML": "Causal (debiased)",
    "CAMTA (Causal Attention)": "Causal (incremental)",
}

CATEGORY_COLORS: Dict[str, str] = {
    "Rule-based": "#4ECDC4",
    "Statistical": "#45B7D1",
    "Game-theoretic": "#96CEB4",
    "Deep Learning": "#FFEAA7",
    "Causal (debiased)": "#DDA0DD",
    "Causal (incremental)": "#B5D8B5",
}


# === Legacy 2-tier causal grouping (experiment 09 only) ===
# Exp 09 intentionally collapses Causal (incremental)/(debiased) into "Causal"
# and breaks out CAMTA as "Causal DL". Preserved here for output stability.
METHOD_CATEGORIES_LEGACY: Dict[str, str] = {
    "Last Click": "Rule-based", "First Click": "Rule-based",
    "Linear": "Rule-based", "Time Decay (7.0d)": "Rule-based",
    "Position-Based (40%/40%)": "Rule-based",
    "Markov (order=1)": "Statistical", "Markov (order=2)": "Statistical",
    "Shapley (model-based)": "Game-theoretic",
    "LSTM+Attention (attn weights)": "Deep Learning",
    "LSTM+Attention (LOO)": "Deep Learning",
    "Transformer (2L/2H)": "Deep Learning",
    "Incremental Shapley": "Causal",
    "Survival/Poisson (BackElim)": "Causal",
    "Survival/Poisson (AICPE)": "Causal",
    "IPW": "Causal", "Doubly Robust": "Causal", "DML": "Causal",
    "CAMTA (Causal Attention)": "Causal DL",
}

CATEGORY_COLORS_LEGACY: Dict[str, str] = {
    "Rule-based": "#4ECDC4",
    "Statistical": "#45B7D1",
    "Game-theoretic": "#96CEB4",
    "Deep Learning": "#FFEAA7",
    "Causal": "#DDA0DD",
    "Causal DL": "#FF6B6B",
}


def setup_experiment_logging(use_timestamp: bool = False) -> None:
    """Standard logging + warnings setup for experiment scripts."""
    if use_timestamp:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")


def prepare_output_dir(output_dir: PathLike) -> Path:
    """Create `output_dir` if missing and return as Path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_journeys_and_gt(
    data_dir: PathLike = "data/simulation",
) -> Tuple[pd.DataFrame, Dict, Dict[str, float]]:
    """Load (journeys parquet, full ground_truth json, ground_truth_A credits)."""
    journeys = pd.read_parquet(f"{data_dir}/journeys.parquet")
    with open(f"{data_dir}/ground_truth.json") as f:
        gt = json.load(f)
    gt_a = gt["ground_truth_A"]["channel_credits"]
    return journeys, gt, gt_a


def reconstruct_credits_from_eval(
    eval_df: pd.DataFrame,
    gt_a: Dict[str, float],
) -> List[AttributionResult]:
    """Reconstruct AttributionResults from `bias_<channel>` columns.

    credit_k = max(0, gt_a_k + bias_k), then normalized so each row sums to 1.0.
    Used by experiments 07, 09, 11. When the row total is 0, divides by 1.0
    (returns an all-zero credit dict).
    """
    bias_cols = [c for c in eval_df.columns if c.startswith("bias_")]
    channels = [c.replace("bias_", "") for c in bias_cols]

    results: List[AttributionResult] = []
    for _, row in eval_df.iterrows():
        credits = {
            ch: max(0.0, gt_a.get(ch, 0.0) + row[f"bias_{ch}"])
            for ch in channels
        }
        total = sum(credits.values()) or 1.0
        credits = {ch: v / total for ch, v in credits.items()}
        results.append(AttributionResult(
            method=row["method"],
            channel_credits=credits,
            channel_credits_raw=credits,
            metadata={},
        ))
    return results


def score_methods_against_gt(
    methods: Mapping[str, AttributionResult],
    gt: Dict[str, float],
    extra: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Compute MAE/Kendall's Tau for each method against `gt`.

    Returns a list of row dicts ready for `pd.DataFrame(...)`:
    each row contains `{**extra, "method", "mae", "kendall_tau"}`.
    """
    extra = extra or {}
    rows: List[Dict[str, object]] = []
    for method_name, result in methods.items():
        rows.append({
            **extra,
            "method": method_name,
            "mae": compute_mae(result.channel_credits, gt),
            "kendall_tau": compute_kendall_tau(result.channel_credits, gt),
        })
    return rows
