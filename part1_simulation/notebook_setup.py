"""Notebook setup helper — consolidates boilerplate that lived in cell 1
of every part1 notebook (warnings/rcParams + method-category constants).

Usage from a notebook:

    %matplotlib inline
    from part1_simulation.notebook_setup import (
        setup_notebook, METHOD_CATEGORIES, CATEGORY_COLORS,
    )
    setup_notebook()

The `%matplotlib inline` magic stays in the notebook (magics cannot be
called from a function).
"""

import warnings
from typing import Tuple

import matplotlib.pyplot as plt

# Re-export canonical method categorizations defined in experiments._common
# so notebooks and experiment scripts share the same source of truth.
from part1_simulation.experiments._common import (  # noqa: F401
    CATEGORY_COLORS,
    CATEGORY_COLORS_LEGACY,
    METHOD_CATEGORIES,
    METHOD_CATEGORIES_LEGACY,
)


def setup_notebook(
    figsize: Tuple[float, float] = (12, 6),
    font_size: int = 11,
    silence_warnings: bool = True,
) -> None:
    """Apply consistent matplotlib rcParams + filter warnings for notebooks."""
    if silence_warnings:
        warnings.filterwarnings("ignore")
    plt.rcParams.update({
        "figure.figsize": figsize,
        "font.size": font_size,
        "axes.titlesize": font_size + 3,
        "axes.labelsize": font_size + 1,
    })
