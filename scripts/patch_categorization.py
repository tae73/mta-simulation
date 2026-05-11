"""Patch METHOD_CATEGORIES (Option 1) across all experiment scripts.

Splits "Causal" into "Causal (debiased)" and "Causal (incremental)".
Adds "Survival/Poisson (Shapley)" to all dicts.
"""

import re
from pathlib import Path

FILES = [
    "part1_simulation/experiments/01_method_accuracy.py",
    "part1_simulation/experiments/07_budget_optimization.py",
    "part1_simulation/experiments/08_predictive_validation.py",
    "part1_simulation/experiments/10_bootstrap_stability.py",
    "part1_simulation/experiments/11_convergent_validity.py",
]

# Replacements (literal string replace; idempotent if patterns are exact)
REPLACEMENTS = [
    # Survival/Poisson family → Causal (incremental)
    ('"Survival/Poisson (BackElim)": "Causal"',
     '"Survival/Poisson (BackElim)": "Causal (incremental)"'),
    ('"Survival/Poisson (AICPE)": "Causal"',
     '"Survival/Poisson (AICPE)": "Causal (incremental)"'),
    ('"Survival/Poisson (AICPE)":    "Causal"',
     '"Survival/Poisson (AICPE)":    "Causal (incremental)"'),

    # Incremental Shapley → Causal (incremental)
    ('"Incremental Shapley": "Causal"',
     '"Incremental Shapley": "Causal (incremental)"'),

    # IPW / DR / DML → Causal (debiased)
    ('"IPW": "Causal"',     '"IPW": "Causal (debiased)"'),
    ('"Doubly Robust": "Causal"', '"Doubly Robust": "Causal (debiased)"'),
    ('"DML": "Causal"',     '"DML": "Causal (debiased)"'),

    # CAMTA → Causal (incremental) (was "Causal DL")
    ('"CAMTA (Causal Attention)": "Causal DL"',
     '"CAMTA (Causal Attention)": "Causal (incremental)"'),

    # CATEGORY_COLORS update
    ('"Causal": "#DDA0DD",\n    "Causal DL": "#FF6B6B",',
     '"Causal (debiased)": "#DDA0DD",\n    "Causal (incremental)": "#B5D8B5",'),
    ('"Causal": "#DDA0DD", "Causal DL": "#FF6B6B",',
     '"Causal (debiased)": "#DDA0DD", "Causal (incremental)": "#B5D8B5",'),
]

# Add Shapley row right after AICPE in METHOD_CATEGORIES dicts (idempotent)
SHAPLEY_INSERT_PATTERNS = [
    # 4-space indent
    ('    "Survival/Poisson (AICPE)": "Causal (incremental)",\n',
     '    "Survival/Poisson (AICPE)": "Causal (incremental)",\n'
     '    "Survival/Poisson (Shapley)": "Causal (incremental)",\n'),
    # multi-space variant
    ('    "Survival/Poisson (AICPE)":    "Causal (incremental)",\n',
     '    "Survival/Poisson (AICPE)":    "Causal (incremental)",\n'
     '    "Survival/Poisson (Shapley)":  "Causal (incremental)",\n'),
]


def patch_file(path: str) -> int:
    p = Path(path)
    text = p.read_text()
    n_changes = 0
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            n_changes += 1

    # Idempotent Shapley insert
    for old, new in SHAPLEY_INSERT_PATTERNS:
        if old in text and "Survival/Poisson (Shapley)" not in text:
            text = text.replace(old, new, 1)
            n_changes += 1

    if n_changes:
        p.write_text(text)
    return n_changes


def main():
    total = 0
    for f in FILES:
        n = patch_file(f)
        print(f"  {f}: {n} changes")
        total += n
    print(f"Total: {total} changes across {len(FILES)} files")


if __name__ == "__main__":
    main()
