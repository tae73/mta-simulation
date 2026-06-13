"""Unit tests for dgp/channel_config.py — position-dependent transition matrices.

Guards the 7-channel invariant (CLAUDE.md): three 7x7 stochastic matrices
keyed by funnel regime, the regime boundary logic (≤0.3 / ≤0.7), and the
deterministic next-channel sampler.

Sections:
    1.  build_transition_matrices  — keys, shape, non-negativity, row-stochastic
    2.  select_regime              — boundary thresholds (0.3, 0.7) incl. exact edges
    3.  sample_next_channel        — reproducibility + valid channel output
    4.  _validate_matrix           — direct validation of the row-sum/shape guard
"""
from __future__ import annotations

import numpy as np
import pytest

from part1_simulation import CHANNEL_NAMES
from part1_simulation.dgp.channel_config import (
    CHANNEL_TO_IDX,
    IDX_TO_CHANNEL,
    _validate_matrix,
    build_transition_matrices,
    sample_next_channel,
    select_regime,
)
from part1_simulation.tests._journey_factory import default_dgp_config


# ============================================================
# 1. build_transition_matrices — keys / shape / stochasticity
# ============================================================

def test_build_transition_matrices_keys_exact():
    """Returns exactly the three funnel-regime keys."""
    matrices = build_transition_matrices(default_dgp_config())
    assert set(matrices.keys()) == {"early", "mid", "late"}


def test_build_transition_matrices_shape_7x7():
    """Each regime matrix is a 7x7 ndarray (7-channel invariant)."""
    matrices = build_transition_matrices(default_dgp_config())
    for name, matrix in matrices.items():
        assert isinstance(matrix, np.ndarray), f"{name}: not an ndarray"
        assert matrix.shape == (7, 7), f"{name}: shape {matrix.shape}"
        assert matrix.shape[0] == len(CHANNEL_NAMES)


def test_build_transition_matrices_non_negative():
    """No negative transition probabilities."""
    matrices = build_transition_matrices(default_dgp_config())
    for name, matrix in matrices.items():
        assert np.all(matrix >= 0), f"{name}: contains negative entries"


def test_build_transition_matrices_rows_sum_to_one():
    """Every row of every regime matrix is a probability distribution."""
    matrices = build_transition_matrices(default_dgp_config())
    for name, matrix in matrices.items():
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(7), atol=1e-10,
            err_msg=f"{name}: rows don't sum to 1.0",
        )


def test_build_transition_matrices_config_independent():
    """Matrices are hand-specified constants, independent of the config object."""
    m_default = build_transition_matrices(default_dgp_config())
    m_other = build_transition_matrices(default_dgp_config(n_users=500, seed=7))
    for key in ("early", "mid", "late"):
        np.testing.assert_allclose(m_default[key], m_other[key], atol=1e-12)


# ============================================================
# 2. select_regime — boundary thresholds (≤0.3 → early, ≤0.7 → mid, else late)
# ============================================================

def test_select_regime_interior_values():
    """Documented interior cases: 0.2→early, 0.5→mid, 0.9→late."""
    assert select_regime(0.2) == "early"
    assert select_regime(0.5) == "mid"
    assert select_regime(0.9) == "late"


def test_select_regime_lower_boundary_0_3_inclusive():
    """position_ratio == 0.3 is `early` (code uses `<= 0.3`)."""
    assert select_regime(0.3) == "early"


def test_select_regime_upper_boundary_0_7_inclusive():
    """position_ratio == 0.7 is `mid` (code uses `<= 0.7`)."""
    assert select_regime(0.7) == "mid"


def test_select_regime_just_past_boundaries():
    """Just above each threshold flips to the next regime."""
    assert select_regime(0.3 + 1e-9) == "mid"
    assert select_regime(0.7 + 1e-9) == "late"


def test_select_regime_extremes():
    """Endpoints of the [0, 1] domain."""
    assert select_regime(0.0) == "early"
    assert select_regime(1.0) == "late"


# ============================================================
# 3. sample_next_channel — determinism + valid output
# ============================================================

def test_sample_next_channel_returns_valid_channel():
    """Sampled channel is always one of the 7 canonical channel names."""
    matrices = build_transition_matrices(default_dgp_config())
    rng = np.random.default_rng(123)
    for _ in range(50):
        nxt = sample_next_channel("Display", 0.1, matrices, rng)
        assert nxt in CHANNEL_NAMES


def test_sample_next_channel_reproducible_fresh_seeds():
    """Two fresh same-seed generators yield identical sampled channels."""
    matrices = build_transition_matrices(default_dgp_config())
    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    seq_a = [sample_next_channel("Social", 0.5, matrices, rng_a) for _ in range(30)]
    seq_b = [sample_next_channel("Social", 0.5, matrices, rng_b) for _ in range(30)]
    assert seq_a == seq_b


def test_sample_next_channel_single_call_deterministic():
    """A single call with fresh same-seed rngs gives the same result."""
    matrices = build_transition_matrices(default_dgp_config())
    out1 = sample_next_channel("Paid Search", 0.9, matrices, np.random.default_rng(99))
    out2 = sample_next_channel("Paid Search", 0.9, matrices, np.random.default_rng(99))
    assert out1 == out2
    assert out1 in CHANNEL_NAMES


def test_sample_next_channel_respects_regime():
    """The sampler draws from the row of the regime selected by position_ratio."""
    matrices = build_transition_matrices(default_dgp_config())
    # Late regime (position 0.9), Direct source row: Paid Search and Direct share
    # the top mass (0.30 each). Sampled channels must all have positive
    # probability in that row, and the most-sampled one must be a top-mass channel.
    regime_row = matrices["late"][CHANNEL_TO_IDX["Direct"]]
    rng = np.random.default_rng(7)
    counts = {ch: 0 for ch in CHANNEL_NAMES}
    for _ in range(2000):
        nxt = sample_next_channel("Direct", 0.9, matrices, rng)
        counts[nxt] += 1
    # Every sampled channel must have positive probability in the regime row.
    for ch, c in counts.items():
        if c > 0:
            assert regime_row[CHANNEL_TO_IDX[ch]] > 0
    # The two dominant channels (Paid Search, Direct @ 0.30) should be sampled most.
    top = max(counts, key=counts.get)
    assert top in ("Paid Search", "Direct")


def test_index_mappings_are_inverse():
    """CHANNEL_TO_IDX and IDX_TO_CHANNEL are consistent inverses over the 7 channels."""
    assert len(CHANNEL_TO_IDX) == 7
    assert len(IDX_TO_CHANNEL) == 7
    for ch in CHANNEL_NAMES:
        assert IDX_TO_CHANNEL[CHANNEL_TO_IDX[ch]] == ch


# ============================================================
# 4. _validate_matrix — direct guard checks
# ============================================================

def test_validate_matrix_accepts_valid():
    """A proper row-stochastic 7x7 matrix passes validation silently."""
    matrices = build_transition_matrices(default_dgp_config())
    # Should not raise.
    _validate_matrix(matrices["mid"], "mid")


def test_validate_matrix_rejects_wrong_shape():
    """Non-(7,7) shape triggers an AssertionError."""
    bad = np.full((3, 3), 1.0 / 3.0)
    with pytest.raises(AssertionError):
        _validate_matrix(bad, "bad_shape")


def test_validate_matrix_rejects_negative():
    """A negative entry triggers an AssertionError."""
    bad = build_transition_matrices(default_dgp_config())["early"].copy()
    bad[0, 0] = -0.1
    bad[0, 1] += 0.1  # keep row sum at 1.0 so the negative check is what fires
    with pytest.raises(AssertionError):
        _validate_matrix(bad, "bad_negative")


def test_validate_matrix_rejects_bad_row_sum():
    """A row that does not sum to 1.0 triggers the assert_allclose check."""
    bad = build_transition_matrices(default_dgp_config())["late"].copy()
    bad[0, 0] += 0.5  # row 0 now sums to 1.5
    with pytest.raises(AssertionError):
        _validate_matrix(bad, "bad_rowsum")
