"""End-to-end Data Generation Process (DGP) pipeline.

Orchestrates: user segments → channel sequences → timestamps → conversion decisions.
Includes alpha-0 binary search calibration to hit target conversion rate (2-3%).

Output: Long-format journey DataFrame (one row per touchpoint per user)
conforming to JOURNEY_SCHEMA.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from part1_simulation import BudgetConfig, CHANNEL_NAMES, DGPConfig, JOURNEY_SCHEMA, SegmentDef
from part1_simulation.dgp.channel_config import (
    build_transition_matrices,
    sample_next_channel,
)
from part1_simulation.dgp.conversion_model import (
    compute_log_intensity,
    decide_conversion,
    intensity_to_conversion_prob,
)
from part1_simulation.dgp.cost_model import assign_touchpoint_costs, compute_cost_summary
from part1_simulation.dgp.user_segments import assign_segments

logger = logging.getLogger(__name__)


# ============================================================
# Step 2: Generate Channel Sequences
# ============================================================

def generate_channel_sequences(
    users_df: pd.DataFrame,
    config: DGPConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate position-dependent channel sequences for all users.

    For each user, starting from start_channel, applies the position-dependent
    transition matrix to sample subsequent channels.

    Returns:
        Long-format DataFrame: user_id, segment, touchpoint_idx, channel,
        journey_length, start_channel.
    """
    matrices = build_transition_matrices(config)

    rows: List[dict] = []
    for _, user in users_df.iterrows():
        user_id = user["user_id"]
        segment = user["segment"]
        journey_length = user["journey_length"]
        current_channel = user["start_channel"]

        for step in range(journey_length):
            rows.append({
                "user_id": user_id,
                "segment": segment,
                "touchpoint_idx": step,
                "channel": current_channel,
                "journey_length": journey_length,
            })

            if step < journey_length - 1:
                position_ratio = (step + 1) / journey_length
                current_channel = sample_next_channel(
                    current_channel, position_ratio, matrices, rng,
                )

    df = pd.DataFrame(rows)
    df["user_id"] = df["user_id"].astype(np.int64)
    df["touchpoint_idx"] = df["touchpoint_idx"].astype(np.int64)
    df["journey_length"] = df["journey_length"].astype(np.int64)
    df["segment"] = df["segment"].astype("category")
    df["channel"] = df["channel"].astype("category")
    return df


# ============================================================
# Step 3: Assign Timestamps
# ============================================================

def assign_timestamps(
    journeys: pd.DataFrame,
    inter_arrival_lambda_hours: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add timestamps to each touchpoint using Exponential inter-arrival times.

    First touchpoint of each user is at t=0. Subsequent intervals drawn from
    Exponential(scale=inter_arrival_lambda_hours), then cumulative-summed.
    """
    n_rows = len(journeys)
    # Draw inter-arrival times for all rows, then zero out first touchpoints
    inter_arrivals = rng.exponential(
        scale=inter_arrival_lambda_hours, size=n_rows,
    )

    timestamps = np.empty(n_rows, dtype=np.float64)
    cumsum = 0.0
    prev_user = -1

    for i in range(n_rows):
        uid = journeys.iloc[i]["user_id"]
        if uid != prev_user:
            cumsum = 0.0
            prev_user = uid
        else:
            cumsum += inter_arrivals[i]
        timestamps[i] = cumsum

    return journeys.assign(timestamp=timestamps)


# ============================================================
# Step 4: Compute Conversions
# ============================================================

def _get_segment_def(segment_name: str, config: DGPConfig) -> SegmentDef:
    """Lookup SegmentDef by name."""
    for seg in config.segments:
        if seg.name == segment_name:
            return seg
    raise ValueError(f"Unknown segment: {segment_name}")


def compute_conversions(
    journeys: pd.DataFrame,
    config: DGPConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute conversion intensity and make conversion decisions per user.

    For each user, evaluates the log-intensity at their last touchpoint time,
    then makes a Bernoulli draw.

    Returns:
        DataFrame with added columns: converted, is_last_touchpoint, conversion_intensity.
    """
    # Group journeys by user for per-user processing
    user_groups = journeys.groupby("user_id", sort=False)

    converted_map = {}
    intensity_map = {}

    for user_id, group in user_groups:
        channels = group["channel"].tolist()
        timestamps = group["timestamp"].tolist()
        observation_time = timestamps[-1]
        segment_name = group["segment"].iloc[0]
        segment_def = _get_segment_def(segment_name, config)

        log_intensity = compute_log_intensity(
            channels, timestamps, observation_time, config, segment_def,
        )
        converted = decide_conversion(log_intensity, rng)

        converted_map[user_id] = converted
        intensity_map[user_id] = log_intensity

    # Map back to the long-format DataFrame
    journeys = journeys.assign(
        converted=journeys["user_id"].map(converted_map).astype(bool),
        conversion_intensity=journeys["user_id"].map(intensity_map).astype(np.float64),
        is_last_touchpoint=(
            journeys["touchpoint_idx"] == journeys["journey_length"] - 1
        ),
    )

    return journeys


# ============================================================
# Step 5: Validation
# ============================================================

def validate_generated_data(journeys: pd.DataFrame, config: DGPConfig) -> dict:
    """Run validation checks on generated data. Returns summary statistics.

    Checks:
    1. Conversion rate in expected range
    2. Journey length distribution is right-skewed
    3. Channel frequency distribution
    4. Timestamp monotonicity within users
    5. Converted journeys have different channel composition
    """
    user_level = journeys.groupby("user_id").agg(
        converted=("converted", "first"),
        journey_length=("journey_length", "first"),
        segment=("segment", "first"),
    )

    n_users = len(user_level)
    n_converted = user_level["converted"].sum()
    conversion_rate = n_converted / n_users

    stats = {
        "n_users": int(n_users),
        "n_converted": int(n_converted),
        "conversion_rate": float(conversion_rate),
        "journey_length_mean": float(user_level["journey_length"].mean()),
        "journey_length_median": float(user_level["journey_length"].median()),
        "journey_length_skew": float(user_level["journey_length"].skew()),
        "segment_counts": user_level["segment"].value_counts().to_dict(),
        "channel_frequency": journeys["channel"].value_counts(normalize=True).to_dict(),
        "conversion_rate_by_segment": (
            user_level.groupby("segment", observed=True)["converted"]
            .mean()
            .to_dict()
        ),
    }

    # Validation checks (log warnings, don't fail)
    if not (0.015 <= conversion_rate <= 0.04):
        logger.warning(
            f"Conversion rate {conversion_rate:.4f} outside expected range [0.015, 0.04]"
        )

    if stats["journey_length_skew"] < 0.3:
        logger.warning(
            f"Journey length skewness {stats['journey_length_skew']:.3f} is low "
            "(expected > 0.5 for right-skewed)"
        )

    # Check timestamp monotonicity
    ts_diff = journeys.groupby("user_id")["timestamp"].diff()
    n_violations = (ts_diff.dropna() < 0).sum()
    if n_violations > 0:
        logger.error(f"Found {n_violations} timestamp monotonicity violations!")
    stats["timestamp_violations"] = int(n_violations)

    return stats


# ============================================================
# Alpha-0 Calibration
# ============================================================

def calibrate_alpha_0(
    config: DGPConfig,
    target_low: float = 0.02,
    target_high: float = 0.03,
    n_calibration: int = 5000,
    max_iterations: int = 20,
) -> float:
    """Binary search on alpha_0 to hit target conversion rate.

    Generates small samples with different alpha_0 values until the conversion
    rate falls within [target_low, target_high].

    Args:
        config: base DGP config (alpha_0 will be overridden).
        target_low: minimum acceptable conversion rate.
        target_high: maximum acceptable conversion rate.
        n_calibration: sample size for each calibration run.
        max_iterations: maximum binary search iterations.

    Returns:
        Calibrated alpha_0 value.
    """
    target_mid = (target_low + target_high) / 2
    lo, hi = -10.0, 0.0  # search bounds for alpha_0

    logger.info(f"Calibrating alpha_0 for conversion rate [{target_low}, {target_high}]...")

    for iteration in range(max_iterations):
        alpha_0 = (lo + hi) / 2
        cal_config = config._replace(
            n_users=n_calibration,
            alpha_0=alpha_0,
            random_seed=config.random_seed + iteration,
        )

        rng = np.random.default_rng(cal_config.random_seed)
        users = assign_segments(
            cal_config.n_users, cal_config.segments, cal_config.max_touchpoints, rng,
        )
        journeys = generate_channel_sequences(users, cal_config, rng)
        journeys = assign_timestamps(journeys, cal_config.inter_arrival_lambda_hours, rng)
        journeys = compute_conversions(journeys, cal_config, rng)

        rate = journeys.groupby("user_id")["converted"].first().mean()
        logger.info(
            f"  iter {iteration}: alpha_0={alpha_0:.4f}, conversion_rate={rate:.4f}"
        )

        if target_low <= rate <= target_high:
            logger.info(f"  Calibration converged: alpha_0={alpha_0:.4f}")
            return alpha_0

        if rate < target_mid:
            lo = alpha_0  # need higher alpha_0 to increase conversion
        else:
            hi = alpha_0  # need lower alpha_0 to decrease conversion

    logger.warning(
        f"Calibration did not converge after {max_iterations} iterations. "
        f"Last alpha_0={alpha_0:.4f}, rate={rate:.4f}"
    )
    return alpha_0


# ============================================================
# Main Pipeline
# ============================================================

def generate_all_journeys(
    config: DGPConfig,
    calibrate: bool = True,
    budget_config: Optional[BudgetConfig] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Full DGP pipeline: segments → sequences → timestamps → conversions [→ costs].

    Args:
        config: DGP configuration.
        calibrate: if True, run alpha_0 calibration before full generation.
        budget_config: optional cost layer configuration. If provided, adds
            touchpoint_cost column. Does NOT affect conversion decisions.

    Returns:
        Tuple of (journey DataFrame, summary stats dict).
    """
    n_steps = 6 if budget_config is not None else 5

    if calibrate:
        calibrated_alpha_0 = calibrate_alpha_0(config)
        config = config._replace(alpha_0=calibrated_alpha_0)

    rng = np.random.default_rng(config.random_seed)

    # Step 1: Assign users to segments
    logger.info(f"Step 1/{n_steps}: Assigning user segments...")
    users = assign_segments(config.n_users, config.segments, config.max_touchpoints, rng)

    # Step 2: Generate channel sequences
    logger.info(f"Step 2/{n_steps}: Generating channel sequences...")
    journeys = generate_channel_sequences(users, config, rng)

    # Step 3: Assign timestamps
    logger.info(f"Step 3/{n_steps}: Assigning timestamps...")
    journeys = assign_timestamps(journeys, config.inter_arrival_lambda_hours, rng)

    # Step 4: Compute conversions
    logger.info(f"Step 4/{n_steps}: Computing conversions...")
    journeys = compute_conversions(journeys, config, rng)

    # Step 5 (optional): Assign costs — observation layer, after conversion decisions
    if budget_config is not None:
        logger.info(f"Step 5/{n_steps}: Assigning touchpoint costs...")
        journeys = assign_touchpoint_costs(journeys, budget_config, rng)

    # Final step: Validate
    logger.info(f"Step {n_steps}/{n_steps}: Validating generated data...")
    stats = validate_generated_data(journeys, config)
    stats["calibrated_alpha_0"] = config.alpha_0

    # Add cost summary if cost layer was applied
    if budget_config is not None:
        cost_summary = compute_cost_summary(journeys)
        stats["cost_summary"] = cost_summary
        logger.info(
            f"Cost layer applied: total spend ${cost_summary['total_spend']:,.2f}, "
            f"CPA ${cost_summary['cost_per_conversion']:.2f}"
        )

    logger.info(
        f"Generation complete: {stats['n_users']} users, "
        f"{stats['n_converted']} converted ({stats['conversion_rate']:.4f})"
    )

    return journeys, stats


def _make_json_serializable(obj: object) -> object:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


def save_generated_data(
    journeys: pd.DataFrame,
    stats: dict,
    output_dir: str,
) -> None:
    """Save journey data and summary statistics to disk.

    Args:
        journeys: the generated journey DataFrame.
        stats: summary statistics dict from validate_generated_data.
        output_dir: directory to write files to.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    journeys.to_parquet(output_path / "journeys.parquet", index=False)

    serializable_stats = _make_json_serializable(stats)

    with open(output_path / "summary_stats.json", "w") as f:
        json.dump(serializable_stats, f, indent=2)

    # Save cost summary separately if present
    if "cost_summary" in stats:
        cost_data = _make_json_serializable(stats["cost_summary"])
        with open(output_path / "cost_summary.json", "w") as f:
            json.dump(cost_data, f, indent=2)
        logger.info(f"Cost summary saved to {output_path / 'cost_summary.json'}")

    logger.info(f"Saved to {output_path}/")


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate MTA simulation data")
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Config directory path")
    parser.add_argument("--output-dir", type=str, default="data/simulation")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-cost", action="store_true", help="Skip cost layer")
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    from part1_simulation.config_loader import load_budget_config, load_dgp_config

    overrides = list(args.overrides)
    if args.n_users is not None:
        overrides.append(f"n_users={args.n_users}")

    config = load_dgp_config(
        config_dir=args.config,
        overrides=overrides,
    )

    budget_config = None if args.no_cost else load_budget_config(config_dir=args.config)

    journeys, stats = generate_all_journeys(
        config, calibrate=not args.no_calibrate, budget_config=budget_config,
    )
    save_generated_data(journeys, stats, args.output_dir)

    print(f"\nSummary:")
    print(f"  Users: {stats['n_users']:,}")
    print(f"  Converted: {stats['n_converted']:,} ({stats['conversion_rate']:.4f})")
    print(f"  Mean journey length: {stats['journey_length_mean']:.2f}")
    print(f"  Journey length skewness: {stats['journey_length_skew']:.3f}")
    print(f"  Conversion rate by segment:")
    for seg, rate in stats["conversion_rate_by_segment"].items():
        print(f"    {seg}: {rate:.4f}")
