"""Alternative DGP probability processes for robustness validation.

Each module exports `generate_dgp_<name>(n_users, seed, **params)` that returns:
    (journeys: pd.DataFrame, ground_truth: Dict[str, float], metadata: Dict[str, Any])

DGPs:
- logistic_dgp: P(conv) = sigmoid(α + Σ w_k · count_k) — favors LR-based methods
- markov_dgp:   discrete Markov chain with Conv/Drop states — favors Markov methods
- cox_dgp:      Cox PH with Weibull baseline — non-log-linear baseline test
- hawkes_dgp:   Multivariate self-exciting Hawkes — favors Survival self-excitation
"""
