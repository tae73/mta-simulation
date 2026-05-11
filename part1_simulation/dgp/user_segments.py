"""User segment assignment and journey length generation.

Generates a DataFrame of users with:
- Segment assignment (New 50%, Exploratory 30%, Loyal 20%)
- Journey length from Geometric(p) + offset distribution
- Starting channel drawn from segment's preferred channels

Du et al. (2019) framework: user heterogeneity (d_i) drives segment membership,
which in turn creates selection bias (confounding) for causal experiments.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from part1_simulation import DGPConfig, SegmentDef


def generate_journey_length(
    segment: SegmentDef,
    n: int,
    max_touchpoints: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw journey lengths from Geometric(p) + offset, capped at max_touchpoints.

    Args:
        segment: segment definition with geometric_p and geometric_offset.
        n: number of lengths to generate.
        max_touchpoints: upper cap on journey length.
        rng: numpy random generator.

    Returns:
        Array of journey lengths (int64).
    """
    raw_lengths = rng.geometric(p=segment.geometric_p, size=n) + segment.geometric_offset
    return np.clip(raw_lengths, a_min=1, a_max=max_touchpoints)


def sample_start_channels(
    segment: SegmentDef,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample starting channels from segment's preferred list.

    Args:
        segment: segment definition with start_channels tuple.
        n: number of channels to sample.
        rng: numpy random generator.

    Returns:
        Array of channel name strings.
    """
    indices = rng.integers(0, len(segment.start_channels), size=n)
    return np.array([segment.start_channels[i] for i in indices])


def assign_segments(
    n_users: int,
    segments: Tuple[SegmentDef, ...],
    max_touchpoints: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Assign each user a segment, journey length, and starting channel.

    Args:
        n_users: total number of users to generate.
        segments: tuple of SegmentDef (proportions must sum to 1.0).
        max_touchpoints: maximum allowed journey length.
        rng: numpy random generator.

    Returns:
        DataFrame with columns: user_id, segment, journey_length, start_channel.
    """
    proportions = np.array([s.proportion for s in segments])
    assert abs(proportions.sum() - 1.0) < 1e-6, (
        f"Segment proportions must sum to 1.0, got {proportions.sum()}"
    )

    # Assign segments via multinomial split
    segment_counts = rng.multinomial(n_users, proportions)

    frames = []
    user_id_offset = 0

    for seg, count in zip(segments, segment_counts):
        journey_lengths = generate_journey_length(seg, count, max_touchpoints, rng)
        start_channels = sample_start_channels(seg, count, rng)

        df = pd.DataFrame({
            "user_id": np.arange(user_id_offset, user_id_offset + count, dtype=np.int64),
            "segment": seg.name,
            "journey_length": journey_lengths.astype(np.int64),
            "start_channel": start_channels,
        })
        frames.append(df)
        user_id_offset += count

    result = pd.concat(frames, ignore_index=True)
    result["segment"] = result["segment"].astype("category")
    return result
