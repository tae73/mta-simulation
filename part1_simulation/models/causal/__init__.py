"""Causal MTA methods (incremental Shapley, survival/Poisson, IPW/DR/DML, CAMTA).

Public re-exports for convenient import paths.
"""

from part1_simulation.models.causal.survival_attribution import (
    TIME_BIN_EDGES_HOURS,
    compute_aicpe_attribution,  # deprecated
    compute_backwards_elimination_attribution,
    compute_survival_attribution,
    compute_synergy_report,
)

__all__ = [
    "TIME_BIN_EDGES_HOURS",
    "compute_survival_attribution",
    "compute_backwards_elimination_attribution",
    "compute_synergy_report",
    "compute_aicpe_attribution",
]
