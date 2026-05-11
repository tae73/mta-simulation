"""Channel definitions and position-dependent transition probability matrices.

Three 7x7 transition matrices govern channel sequencing:
- early  (position ≤ 0.3): upper-funnel dominance, awareness → exploration
- mid    (0.3 < position ≤ 0.7): balanced exploration across channels
- late   (position > 0.7): convergence to lower-funnel conversion channels

Design rationale: Real marketing journeys show funnel progression from
awareness (Display, Social) → consideration (Organic, Referral) → conversion
(Paid Search, Email, Direct). Position-dependent matrices capture this.
"""

from typing import Dict, Tuple

import numpy as np

from part1_simulation import CHANNEL_NAMES, DGPConfig

# Channel index mapping (consistent across all matrices)
CHANNEL_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}
IDX_TO_CHANNEL: Dict[int, str] = {i: name for i, name in enumerate(CHANNEL_NAMES)}

# Indices by funnel position for readability
_DISPLAY = 0
_SOCIAL = 1
_ORGANIC = 2
_PAID = 3
_EMAIL = 4
_REFERRAL = 5
_DIRECT = 6


def _build_early_matrix() -> np.ndarray:
    """Early journey (position ≤ 0.3): upper-funnel channels dominate.

    Display/Social have high retention and cross-transition.
    Organic/Referral emerge as exploration targets.
    Paid Search/Direct are rare this early.
    """
    #                     Disp   Soc    Org    Paid   Email  Ref    Dir
    return np.array([
        # From Display
        [0.10,  0.25,  0.30,  0.08,  0.07,  0.15,  0.05],
        # From Social
        [0.20,  0.10,  0.25,  0.08,  0.12,  0.18,  0.07],
        # From Organic Search
        [0.15,  0.20,  0.15,  0.12,  0.10,  0.18,  0.10],
        # From Paid Search
        [0.10,  0.12,  0.20,  0.15,  0.13,  0.15,  0.15],
        # From Email
        [0.12,  0.18,  0.20,  0.10,  0.15,  0.15,  0.10],
        # From Referral
        [0.15,  0.22,  0.25,  0.08,  0.10,  0.10,  0.10],
        # From Direct
        [0.12,  0.15,  0.25,  0.12,  0.13,  0.13,  0.10],
    ], dtype=np.float64)


def _build_mid_matrix() -> np.ndarray:
    """Mid journey (0.3 < position ≤ 0.7): balanced exploration.

    Organic Search becomes central hub.
    Email/Referral gain as mid-funnel channels.
    Paid Search starts appearing more frequently.
    """
    #                     Disp   Soc    Org    Paid   Email  Ref    Dir
    return np.array([
        # From Display
        [0.08,  0.15,  0.25,  0.18,  0.12,  0.12,  0.10],
        # From Social
        [0.12,  0.08,  0.22,  0.15,  0.18,  0.15,  0.10],
        # From Organic Search
        [0.10,  0.12,  0.10,  0.22,  0.15,  0.16,  0.15],
        # From Paid Search
        [0.05,  0.08,  0.15,  0.18,  0.18,  0.12,  0.24],
        # From Email
        [0.08,  0.10,  0.18,  0.20,  0.12,  0.12,  0.20],
        # From Referral
        [0.10,  0.15,  0.20,  0.18,  0.15,  0.07,  0.15],
        # From Direct
        [0.08,  0.10,  0.18,  0.22,  0.18,  0.12,  0.12],
    ], dtype=np.float64)


def _build_late_matrix() -> np.ndarray:
    """Late journey (position > 0.7): convergence to conversion channels.

    Paid Search, Email, and Direct dominate.
    Display/Social drop significantly.
    Strong funneling toward purchase-intent channels.
    """
    #                     Disp   Soc    Org    Paid   Email  Ref    Dir
    return np.array([
        # From Display
        [0.05,  0.08,  0.12,  0.30,  0.20,  0.08,  0.17],
        # From Social
        [0.05,  0.05,  0.10,  0.28,  0.25,  0.10,  0.17],
        # From Organic Search
        [0.03,  0.05,  0.07,  0.30,  0.20,  0.10,  0.25],
        # From Paid Search
        [0.02,  0.03,  0.05,  0.20,  0.25,  0.08,  0.37],
        # From Email
        [0.03,  0.05,  0.08,  0.28,  0.12,  0.07,  0.37],
        # From Referral
        [0.03,  0.07,  0.10,  0.28,  0.22,  0.05,  0.25],
        # From Direct
        [0.02,  0.03,  0.05,  0.30,  0.25,  0.05,  0.30],
    ], dtype=np.float64)


def _validate_matrix(matrix: np.ndarray, name: str) -> None:
    """Verify transition matrix is valid (rows sum to 1.0, non-negative)."""
    assert matrix.shape == (7, 7), f"{name}: expected (7,7), got {matrix.shape}"
    assert np.all(matrix >= 0), f"{name}: contains negative values"
    row_sums = matrix.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, atol=1e-10,
        err_msg=f"{name}: rows don't sum to 1.0 — {row_sums}",
    )


def build_transition_matrices(config: DGPConfig) -> Dict[str, np.ndarray]:
    """Build the three position-dependent transition matrices.

    Returns:
        Dict with keys "early", "mid", "late", each mapping to a 7x7 ndarray.
    """
    matrices = {
        "early": _build_early_matrix(),
        "mid": _build_mid_matrix(),
        "late": _build_late_matrix(),
    }
    for name, matrix in matrices.items():
        _validate_matrix(matrix, name)
    return matrices


def select_regime(position_ratio: float) -> str:
    """Select transition matrix regime based on position in journey.

    Args:
        position_ratio: current_step / total_journey_length, in [0, 1].
    """
    if position_ratio <= 0.3:
        return "early"
    elif position_ratio <= 0.7:
        return "mid"
    else:
        return "late"


def sample_next_channel(
    current_channel: str,
    position_ratio: float,
    matrices: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> str:
    """Sample the next channel from the position-dependent transition matrix.

    Args:
        current_channel: name of the current channel.
        position_ratio: current_step / total_journey_length.
        matrices: dict of regime → 7x7 transition matrix.
        rng: numpy random generator.

    Returns:
        Name of the next channel.
    """
    regime = select_regime(position_ratio)
    row_idx = CHANNEL_TO_IDX[current_channel]
    probs = matrices[regime][row_idx]
    next_idx = rng.choice(7, p=probs)
    return IDX_TO_CHANNEL[next_idx]
