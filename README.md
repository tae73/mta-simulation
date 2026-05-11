# Marketing Attribution: From Simulation to Scale

## Overview

This project systematically compares **Marketing Attribution** methodologies against simulation ground truth, validates at scale on real data, and triangulates with aggregate-level Media Mix Modeling:

1. **Part 1 — Simulation-Based MTA Comparison**: Custom DGP integrating Du et al. (2019), Shender et al. (2023), and CDA (2025). 7 interpretable channels, 100K users, 10+ attribution methods evaluated against known ground truth
2. **Part 2 — Criteo Attribution at Scale**: 16.5M events, 2.6M journeys. LSTM/Transformer scalability validation with Attention vs SHAP vs Leave-One-Out attribution extraction
3. **Part 3 — Bayesian Media Mix Modeling**: PyMC-Marketing with Adstock/Saturation transformations, Budget Optimization, Prior Sensitivity Analysis

**Triangulation**: Cross-validate user-level MTA and aggregate-level MMM results, present methodology selection decision framework.

## Project Structure

```
mta-simulation/
├── part1_simulation/
│   ├── dgp/                     # Data Generation Process (Du et al. + Shender + CDA)
│   │   ├── channel_config.py    # 7-channel definitions, transition matrices
│   │   ├── user_segments.py     # New/Exploratory/Loyal user generation
│   │   ├── conversion_model.py  # Log-linear intensity + Poisson process
│   │   └── generate_data.py     # End-to-end journey generation
│   ├── models/                  # MTA methodology implementations
│   │   ├── rule_based.py        # Last/First/Linear/Time Decay/Position-Based
│   │   ├── markov.py            # 1st, 2nd, higher-order Markov Chain
│   │   ├── shapley.py           # Exact Shapley (7ch = 128 coalitions)
│   │   ├── lstm_attention.py    # LSTM + Attention sequence model
│   │   ├── transformer.py       # Encoder-only Transformer
│   │   └── causal/              # Incremental Shapley, Survival, IPW/DR, DML, CAMTA
│   ├── experiments/             # Experiment notebooks (01-06)
│   └── results/                 # Result visualizations
├── part2_criteo/
│   ├── preprocessing/           # Journey reconstruction (uid grouping, attribution window)
│   ├── models/                  # LSTM + Attention, Transformer
│   ├── experiments/             # Experiment notebooks (07-10)
│   └── results/
├── part3_mmm/
│   ├── eda/                     # Time series analysis, cross-correlation
│   ├── models/                  # PyMC-Marketing Bayesian MMM
│   ├── optimization/            # Budget optimization (Lagrange/numerical)
│   └── results/
├── integration/                 # Triangulation analysis
│   ├── triangulation.ipynb
│   └── decision_framework.md
├── configs/                     # Hydra YAML config groups
├── docs/                        # Project plan, methodology guide
└── PLAN.md                      # Progress tracking
```

## Key Components

### Simulation DGP (`part1_simulation/dgp/`)
- `generate_data.py`: End-to-end pipeline for 100K user journeys with ground truth parameters
- Conversion model: log-linear intensity combining channel effects ($\beta$), temporal decay ($f_{channel}$), cross-channel synergy ($\delta$), user heterogeneity ($d_i$)

### Attribution Models (`part1_simulation/models/`)
- Rule-based (5), Markov Chain, Shapley Value, LSTM + Attention, Transformer
- Causal: Incremental Shapley, Survival/Poisson, IPW/Doubly Robust, DML, CAMTA variant

### Criteo Deep Learning (`part2_criteo/models/`)
- LSTM (hidden=128) + Multi-head Attention, Transformer (2-layer, 4 heads)
- Attribution extraction: Attention Weight, SHAP (DeepExplainer), Leave-One-Out

### Bayesian MMM (`part3_mmm/models/`)
- PyMC-Marketing: Geometric/Weibull Adstock, Hill Saturation, NUTS sampling
- Budget optimization with estimated response curves

## Quick Start

```bash
# 1. Generate simulation data (Part 1)
python part1_simulation/dgp/generate_data.py \
    --n-users 100000 --config configs/dgp/default.yaml \
    --output-dir data/simulation

# 2. Run all attribution models
python part1_simulation/models/run_all.py \
    --data-dir data/simulation --results-dir results/part1

# 3. Evaluate vs ground truth
python part1_simulation/experiments/evaluate.py \
    --results-dir results/part1 --ground-truth data/simulation/ground_truth.json
```

## Installation

```bash
pip install -e ".[dev]"
```

## Dataset

| Dataset | Part | Scale | Ground Truth |
|---------|------|-------|--------------|
| Custom Simulation | Part 1 | 100K users, 7 channels, 2-3% conversion | Yes |
| Criteo Attribution (2018) | Part 2 | 16.5M events, 2.6M journeys | No |
| Robyn dt_simulated_weekly | Part 3 | 208 weeks, 6 media channels | Partially |
| PyMC-Marketing simulated | Part 3 | Configurable | Yes |

## References

- Du et al., "Causally Driven Incremental Multi Touch Attribution Using a Recurrent Neural Network" (ArXiv 1902.00215, AdKDD 2019)
- Shender et al., "A Time To Event Framework For Multi-touch Attribution" (Journal of Data Science, Vol.22, 2023)
- CDA, "Causal-driven Attribution: Estimating Channel Influence Without User-level Data" (ArXiv 2512.21211, 2025)
- Chernozhukov et al., "Double/Debiased Machine Learning for Treatment and Structural Parameters" (Econometrics Journal, 2018)
- PyMC-Marketing Documentation: pymc-marketing.readthedocs.io
