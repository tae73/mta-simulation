"""Patch METHOD_CATEGORIES + CATEGORY_COLORS in notebook 05/06 markdown source cells.

Updates the setup cell that defines categorization, used by visualization.
Output cells will be auto-refreshed when notebooks are re-executed.
"""

import json
import re
from pathlib import Path

import nbformat

NB_FILES = [
    "notebooks/part1/05_experiments_and_insights.ipynb",
    "notebooks/part1/06_cost_and_budget_optimization.ipynb",
]


def patch_notebook(path: str) -> int:
    p = Path(path)
    nb = nbformat.read(p, as_version=4)
    n_changes = 0

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        original = src

        # Replace category labels in METHOD_CATEGORIES dict literal
        src = src.replace(
            '"Survival/Poisson (BackElim)": "Causal"',
            '"Survival/Poisson (BackElim)": "Causal (incremental)"',
        )
        src = src.replace(
            '"Survival/Poisson (AICPE)": "Causal"',
            '"Survival/Poisson (AICPE)": "Causal (incremental)"',
        )
        src = src.replace(
            '"Incremental Shapley": "Causal"',
            '"Incremental Shapley": "Causal (incremental)"',
        )
        src = src.replace(
            '"IPW": "Causal"', '"IPW": "Causal (debiased)"'
        )
        src = src.replace(
            '"Doubly Robust": "Causal"', '"Doubly Robust": "Causal (debiased)"'
        )
        src = src.replace('"DML": "Causal"', '"DML": "Causal (debiased)"')
        src = src.replace(
            '"CAMTA (Causal Attention)": "Causal DL"',
            '"CAMTA (Causal Attention)": "Causal (incremental)"',
        )

        # CATEGORY_COLORS replacement
        src = src.replace(
            '"Causal": "#DDA0DD", "Causal DL": "#FF6B6B",',
            '"Causal (debiased)": "#DDA0DD", "Causal (incremental)": "#B5D8B5",',
        )

        # Insert Shapley row idempotently
        if (
            '"Survival/Poisson (AICPE)": "Causal (incremental)"' in src
            and '"Survival/Poisson (Shapley)"' not in src
        ):
            src = src.replace(
                '"Survival/Poisson (AICPE)": "Causal (incremental)",\n',
                '"Survival/Poisson (AICPE)": "Causal (incremental)",\n'
                '    "Survival/Poisson (Shapley)": "Causal (incremental)",\n',
                1,
            )

        if src != original:
            cell.source = src
            n_changes += 1

    if n_changes:
        nbformat.write(nb, p)
    return n_changes


def main():
    for f in NB_FILES:
        n = patch_notebook(f)
        print(f"  {f}: {n} cells changed")


if __name__ == "__main__":
    main()
