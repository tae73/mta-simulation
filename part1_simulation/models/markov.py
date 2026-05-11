"""Markov Chain attribution with Removal Effect.

Builds a transition probability matrix from observed journeys, including
absorbing states (Conversion, Null). Computes Removal Effect:
for each channel, remove it and measure the drop in total conversion probability.

Supports:
    - 1st-order Markov (state = single channel)
    - 2nd-order Markov (state = channel pair)
    - Higher-order with Laplace smoothing for sparse states
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from part1_simulation import AttributionResult, CHANNEL_NAMES


# Special state names
_START = "(Start)"
_CONVERSION = "(Conversion)"
_NULL = "(Null)"


def _build_sequences(journeys: pd.DataFrame) -> List[Tuple[List[str], bool]]:
    """Extract channel sequences and conversion labels per user.

    Returns list of (channel_list, converted_bool) tuples.
    """
    sequences = []
    for user_id, group in journeys.groupby("user_id", sort=False):
        channels = group.sort_values("touchpoint_idx")["channel"].tolist()
        converted = bool(group["converted"].iloc[0])
        sequences.append((channels, converted))
    return sequences


def build_transition_matrix_order1(
    sequences: List[Tuple[List[str], bool]],
    laplace_alpha: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """Build 1st-order Markov transition matrix.

    States: Start, 7 channels, Conversion, Null (10 total).
    Rows sum to 1.0 (after normalization).

    Args:
        sequences: list of (channel_sequence, converted) tuples.
        laplace_alpha: Laplace smoothing parameter.

    Returns:
        (transition_matrix, state_names) — matrix is (n_states x n_states).
    """
    states = [_START] + list(CHANNEL_NAMES) + [_CONVERSION, _NULL]
    state_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    counts = np.full((n_states, n_states), laplace_alpha, dtype=np.float64)

    for channels, converted in sequences:
        # Start → first channel
        counts[state_idx[_START], state_idx[channels[0]]] += 1

        # Channel → channel transitions
        for i in range(len(channels) - 1):
            counts[state_idx[channels[i]], state_idx[channels[i + 1]]] += 1

        # Last channel → absorbing state
        if converted:
            counts[state_idx[channels[-1]], state_idx[_CONVERSION]] += 1
        else:
            counts[state_idx[channels[-1]], state_idx[_NULL]] += 1

    # Absorbing states: self-loops
    counts[state_idx[_CONVERSION], state_idx[_CONVERSION]] = 1.0
    counts[state_idx[_NULL], state_idx[_NULL]] = 1.0

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    matrix = counts / row_sums

    return matrix, states


def build_transition_matrix_order2(
    sequences: List[Tuple[List[str], bool]],
    laplace_alpha: float = 1e-3,
) -> Tuple[np.ndarray, List[str]]:
    """Build 2nd-order Markov transition matrix.

    States: Start, (ch_i, ch_j) pairs for all channel combinations,
    Conversion, Null.

    Args:
        sequences: list of (channel_sequence, converted) tuples.
        laplace_alpha: Laplace smoothing (important for sparse 2nd-order).

    Returns:
        (transition_matrix, state_names).
    """
    # Build pair states
    pair_states = [
        f"{ch1}|{ch2}" for ch1 in CHANNEL_NAMES for ch2 in CHANNEL_NAMES
    ]
    states = [_START] + list(CHANNEL_NAMES) + pair_states + [_CONVERSION, _NULL]
    state_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)

    counts = np.full((n_states, n_states), laplace_alpha, dtype=np.float64)

    for channels, converted in sequences:
        if len(channels) == 1:
            # Start → single channel → absorbing
            counts[state_idx[_START], state_idx[channels[0]]] += 1
            dest = _CONVERSION if converted else _NULL
            counts[state_idx[channels[0]], state_idx[dest]] += 1
        else:
            # Start → first channel
            counts[state_idx[_START], state_idx[channels[0]]] += 1

            # First channel → first pair
            pair_key = f"{channels[0]}|{channels[1]}"
            counts[state_idx[channels[0]], state_idx[pair_key]] += 1

            # Pair → pair transitions
            for i in range(1, len(channels) - 1):
                from_pair = f"{channels[i-1]}|{channels[i]}"
                to_pair = f"{channels[i]}|{channels[i+1]}"
                counts[state_idx[from_pair], state_idx[to_pair]] += 1

            # Last pair → absorbing
            last_pair = f"{channels[-2]}|{channels[-1]}"
            dest = _CONVERSION if converted else _NULL
            counts[state_idx[last_pair], state_idx[dest]] += 1

    # Absorbing states
    counts[state_idx[_CONVERSION], state_idx[_CONVERSION]] = 1.0
    counts[state_idx[_NULL], state_idx[_NULL]] = 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix = counts / row_sums

    return matrix, states


def compute_absorption_probability(
    matrix: np.ndarray,
    states: List[str],
) -> float:
    """Compute probability of reaching Conversion from Start.

    Uses the fundamental matrix of absorbing Markov chains:
    N = (I - Q)^{-1}, where Q is the transient-to-transient submatrix.
    """
    state_idx = {s: i for i, s in enumerate(states)}
    conv_idx = state_idx[_CONVERSION]
    null_idx = state_idx[_NULL]
    start_idx = state_idx[_START]

    # Transient states = all except Conversion and Null
    absorbing = {conv_idx, null_idx}
    transient = [i for i in range(len(states)) if i not in absorbing]
    transient_idx_map = {old: new for new, old in enumerate(transient)}

    n_transient = len(transient)
    Q = matrix[np.ix_(transient, transient)]
    R = matrix[np.ix_(transient, list(absorbing))]

    # Fundamental matrix N = (I - Q)^{-1}
    try:
        N = np.linalg.inv(np.eye(n_transient) - Q)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        N = np.linalg.pinv(np.eye(n_transient) - Q)

    # Absorption probabilities B = N * R
    B = N @ R

    # Find which column in R corresponds to Conversion
    absorbing_list = sorted(absorbing)
    conv_col = absorbing_list.index(conv_idx)

    # P(Conversion | Start)
    start_transient_idx = transient_idx_map.get(start_idx)
    if start_transient_idx is not None:
        return float(B[start_transient_idx, conv_col])
    return 0.0


def compute_removal_effect(
    matrix: np.ndarray,
    states: List[str],
    channel: str,
) -> float:
    """Compute Removal Effect for a channel.

    Remove the channel by redirecting all its outgoing transitions to Null.
    Measure the drop in conversion probability from Start.

    Args:
        matrix: transition matrix.
        states: state name list.
        channel: channel name to remove.

    Returns:
        Removal effect = P_base - P_removed (non-negative).
    """
    state_idx = {s: i for i, s in enumerate(states)}
    base_prob = compute_absorption_probability(matrix, states)

    # Create modified matrix: channel's row → all probability to Null
    modified = matrix.copy()
    null_idx = state_idx[_NULL]

    # For 1st-order: remove the channel state directly
    if channel in state_idx:
        ch_idx = state_idx[channel]
        modified[ch_idx, :] = 0.0
        modified[ch_idx, null_idx] = 1.0

    # For 2nd-order: also remove all pair states containing this channel
    for s, idx in state_idx.items():
        if "|" in s and channel in s.split("|"):
            modified[idx, :] = 0.0
            modified[idx, null_idx] = 1.0

    removed_prob = compute_absorption_probability(modified, states)
    return max(0.0, base_prob - removed_prob)


def compute_markov_attribution(
    journeys: pd.DataFrame,
    order: int = 1,
    laplace_alpha: float = 0.0,
) -> AttributionResult:
    """Compute Markov Chain attribution using Removal Effect.

    Args:
        journeys: long-format journey DataFrame.
        order: Markov order (1 or 2).
        laplace_alpha: Laplace smoothing parameter.

    Returns:
        AttributionResult with normalized channel credits.
    """
    sequences = _build_sequences(journeys)

    if order == 1:
        matrix, states = build_transition_matrix_order1(
            sequences, laplace_alpha=laplace_alpha,
        )
    elif order == 2:
        matrix, states = build_transition_matrix_order2(
            sequences, laplace_alpha=max(laplace_alpha, 1e-3),
        )
    else:
        raise ValueError(f"Order {order} not supported (use 1 or 2)")

    # Compute removal effect for each channel
    removal_effects: Dict[str, float] = {}
    for channel in CHANNEL_NAMES:
        removal_effects[channel] = compute_removal_effect(matrix, states, channel)

    # Normalize
    total = sum(removal_effects.values())
    if total > 0:
        normalized = {k: v / total for k, v in removal_effects.items()}
    else:
        normalized = {k: 1.0 / len(CHANNEL_NAMES) for k in CHANNEL_NAMES}

    return AttributionResult(
        method=f"Markov (order={order})",
        channel_credits=normalized,
        channel_credits_raw=removal_effects,
        metadata={
            "order": order,
            "base_conversion_prob": compute_absorption_probability(matrix, states),
            "n_states": len(states),
        },
    )
