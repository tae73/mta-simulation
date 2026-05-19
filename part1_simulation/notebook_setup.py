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
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

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


# Canonical path constants. Resolve identically to the legacy hardcoded
# relative paths (`../../data/simulation`, `../../results/part1`) used by the
# part1 notebooks, so substituting these is behavior-neutral.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data" / "simulation"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "part1"


def load_exp_csv(name: str, results_dir: Path = RESULTS_DIR) -> "pd.DataFrame":
    """Load an experiment results CSV with a clear error if it is missing.

    ``name`` may be a bare experiment id ("01") or a full file name
    ("01_method_accuracy.csv"). On the normal path this is exactly
    ``pd.read_csv(f"{results_dir}/{file}")``; the only behavioral change is a
    explicit, actionable ``FileNotFoundError`` instead of a downstream
    ``NameError`` when a results CSV has not been generated yet.
    """
    results_dir = Path(results_dir)
    if name.endswith(".csv"):
        path = results_dir / name
    else:
        matches = sorted(results_dir.glob(f"{name}*.csv"))
        path = matches[0] if matches else results_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment results not found: {path}\n"
            f"Generate it first, e.g.: "
            f"python -m part1_simulation.experiments.{name.split('_')[0]}_* "
            f"(see CLAUDE.md Pipeline Usage)."
        )
    return pd.read_csv(path)
