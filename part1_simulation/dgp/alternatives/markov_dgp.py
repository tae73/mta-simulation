"""Discrete-time Markov chain DGP — alternative probability process.

State space: 7 channels + Conversion + Drop-off (9 states).
At each step from a channel state, user transitions to:
    - Another channel (with channel-specific probability)
    - Conversion (absorbing, channel-specific conversion probability)
    - Drop-off (absorbing, channel-specific drop probability)

This DGP naturally favors Markov-based methods (1st/2nd-order Markov, removal
effect attribution) — included as a "DGP-method matching" baseline. If a
method does NOT exploit Markov structure, it should still extract reasonable
channel rankings.

Ground truth (channel credit):
    Removal effect — for each channel k, compute P(reach Conv | k removed
    from state space) vs P(reach Conv | full chain). The drop in conversion
    probability per channel = removal effect, normalized.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from part1_simulation import CHANNEL_NAMES, JOURNEY_SCHEMA

# Per-channel (conv_prob, drop_prob): rest = transitions to other channels
# Tuned so total conversion rate ≈ 2-3% and ranking favors lower-funnel channels.
_CHANNEL_CONV_DROP = {
    "Display":         (0.0015, 0.30),  # awareness — low conv, moderate drop
    "Social":          (0.0012, 0.35),
    "Organic Search":  (0.0040, 0.22),
    "Paid Search":     (0.0110, 0.18),  # bottom-funnel — highest conv
    "Email":           (0.0070, 0.20),
    "Referral":        (0.0035, 0.25),
    "Direct":          (0.0055, 0.20),
}

# Initial state distribution (channel start probabilities)
_INITIAL_DIST = {
    "Display": 0.18, "Social": 0.13, "Organic Search": 0.20,
    "Paid Search": 0.15, "Email": 0.10, "Referral": 0.10, "Direct": 0.14,
}


def _build_transition_matrix() -> Tuple[np.ndarray, Dict[str, int]]:
    """Build 9×9 stochastic transition matrix.

    State order: 7 channels (CHANNEL_NAMES order) + Conv (idx 7) + Drop (idx 8).
    Conv and Drop are absorbing.
    """
    n_ch = len(CHANNEL_NAMES)
    n_states = n_ch + 2  # +Conv, +Drop
    P = np.zeros((n_states, n_states), dtype=np.float64)

    # Channel transitions
    for i, ch in enumerate(CHANNEL_NAMES):
        conv_p, drop_p = _CHANNEL_CONV_DROP[ch]
        cont_p = 1.0 - conv_p - drop_p

        # Distribute cont_p across other channels with simple bias
        # Bottom-funnel channels (Paid Search, Email, Direct) more attractive.
        attractiveness = np.array([
            0.10, 0.08, 0.18, 0.20, 0.18, 0.10, 0.16,
        ])
        attractiveness /= attractiveness.sum()
        # Don't transition to self with high prob (self-loop suppressed)
        attractiveness[i] *= 0.3
        attractiveness /= attractiveness.sum()

        for j in range(n_ch):
            P[i, j] = cont_p * attractiveness[j]

        P[i, n_ch] = conv_p     # to Conv
        P[i, n_ch + 1] = drop_p  # to Drop

    # Absorbing: P[Conv, Conv] = P[Drop, Drop] = 1
    P[n_ch, n_ch] = 1.0
    P[n_ch + 1, n_ch + 1] = 1.0

    state_idx = {ch: i for i, ch in enumerate(CHANNEL_NAMES)}
    state_idx["Conv"] = n_ch
    state_idx["Drop"] = n_ch + 1
    return P, state_idx


def _simulate_user(
    P: np.ndarray,
    state_idx: Dict[str, int],
    initial_p: np.ndarray,
    max_steps: int,
    rng: np.random.Generator,
) -> Tuple[list, bool, int]:
    """Simulate a single user's path.

    Returns:
        (channel_path, converted, n_steps)
    """
    n_ch = len(CHANNEL_NAMES)
    state = rng.choice(n_ch, p=initial_p)
    path = [CHANNEL_NAMES[state]]

    for step in range(1, max_steps):
        state = rng.choice(P.shape[0], p=P[state])
        if state == n_ch:  # Conv
            return path, True, step
        if state == n_ch + 1:  # Drop
            return path, False, step
        path.append(CHANNEL_NAMES[state])

    return path, False, max_steps


def _absorption_probabilities(P: np.ndarray, n_ch: int) -> np.ndarray:
    """For each transient state, compute P(absorbed in Conv).

    Standard Markov chain formula: B = (I - Q)^{-1} R
    where Q = transient×transient block, R = transient×absorbing block.
    """
    Q = P[:n_ch, :n_ch]
    R = P[:n_ch, n_ch:n_ch + 1]  # only Conv column
    N = np.linalg.inv(np.eye(n_ch) - Q)
    B = N @ R  # P(eventual conv | start state i)
    return B.flatten()


def _compute_ground_truth_removal(
    P: np.ndarray,
    initial_p: np.ndarray,
    n_ch: int,
) -> Dict[str, float]:
    """GT credit = removal effect per channel.

    For each channel k:
        baseline_conv = P(eventual Conv | start ~ initial_p)
        without_k = baseline conv when channel k is removed (transitions
            re-routed to Drop)
        credit_k = (baseline - without_k) / baseline
    Then normalize to sum=1.
    """
    baseline_conv_per_state = _absorption_probabilities(P, n_ch)
    baseline_total = float(initial_p @ baseline_conv_per_state)

    raw: Dict[str, float] = {}
    for i, ch in enumerate(CHANNEL_NAMES):
        # Build P_without_k: re-route ALL transitions to channel k → Drop.
        # This is the standard Markov removal-effect formulation.
        P_mod = P.copy()
        # Add k's incoming probability to Drop
        P_mod[:, n_ch + 1] += P_mod[:, i]
        P_mod[:, i] = 0.0
        # Channel k is now an isolated state — we'll just zero its row too
        P_mod[i, :] = 0.0
        P_mod[i, n_ch + 1] = 1.0  # if started there, drop immediately

        # Initial: redistribute k's mass to Drop (treat as if started "no path")
        init_mod = initial_p.copy()
        # mass at i is dropped — total prob conserved by treating those users
        # as starting in a "removed channel" → Drop immediately.
        # For absorption calc, set init_mod[i]=0 (those users contribute 0 conv).
        init_mod[i] = 0.0

        without_per_state = _absorption_probabilities(P_mod, n_ch)
        without_total = float(init_mod @ without_per_state)
        raw[ch] = max(0.0, baseline_total - without_total)

    total = sum(raw.values())
    if total > 0:
        return {k: v / total for k, v in raw.items()}
    return {k: 1.0 / n_ch for k in CHANNEL_NAMES}


def generate_dgp_markov(
    n_users: int = 20_000,
    seed: int = 42,
    max_steps: int = 20,
    inter_arrival_lambda_hours: float = 48.0,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Generate journeys via discrete Markov chain.

    Returns:
        journeys, gt_credits, metadata
    """
    rng = np.random.default_rng(seed)

    P, state_idx = _build_transition_matrix()
    initial_p = np.array([_INITIAL_DIST[ch] for ch in CHANNEL_NAMES])
    initial_p /= initial_p.sum()

    # Simulate each user
    rows = []
    n_converted = 0
    for uid in range(n_users):
        path, converted, n_steps = _simulate_user(
            P, state_idx, initial_p, max_steps, rng,
        )
        if converted:
            n_converted += 1

        # Random segment for compatibility (Markov DGP doesn't use segments)
        segment = rng.choice(["New", "Exploratory", "Loyal"], p=[0.5, 0.3, 0.2])
        # Cumulative timestamps
        ts = np.cumsum(rng.exponential(inter_arrival_lambda_hours, len(path)))
        ts -= ts[0]  # start at 0

        for k, ch in enumerate(path):
            rows.append({
                "user_id": uid,
                "segment": segment,
                "touchpoint_idx": k,
                "channel": ch,
                "timestamp": float(ts[k]),
                "is_last_touchpoint": (k == len(path) - 1),
                "converted": converted,
                "journey_length": len(path),
                "conversion_intensity": 0.0,  # not modeled
                "touchpoint_cost": 0.0,
            })

    journeys = pd.DataFrame(rows)
    for col, dtype in JOURNEY_SCHEMA.items():
        if col in journeys.columns:
            try:
                journeys[col] = journeys[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    # Ground truth via removal effect
    gt_credits = _compute_ground_truth_removal(P, initial_p, len(CHANNEL_NAMES))

    metadata = {
        "dgp_name": "markov",
        "n_users": n_users,
        "seed": seed,
        "max_steps": max_steps,
        "channel_conv_drop": dict(_CHANNEL_CONV_DROP),
        "n_converted": n_converted,
        "conversion_rate": n_converted / n_users,
    }
    return journeys, gt_credits, metadata


if __name__ == "__main__":
    j, gt, meta = generate_dgp_markov(n_users=5000, seed=42)
    print(f"shape: {j.shape}, users: {j.user_id.nunique()}")
    print(f"conv rate: {meta['conversion_rate']:.4f}")
    print(f"GT sum: {sum(gt.values()):.4f}")
    print("GT credits (sorted):")
    for k, v in sorted(gt.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v:.4f}")
