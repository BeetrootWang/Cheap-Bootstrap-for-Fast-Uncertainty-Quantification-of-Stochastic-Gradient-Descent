# Cheap Bootstrap for Fast Uncertainty Quantification of Stochastic Gradient Descent

Code accompanying the JMLR paper:

> Henry Lam and Zitong Wang. *Cheap Bootstrap for Fast Uncertainty Quantification of Stochastic Gradient Descent.* Journal of Machine Learning Research, 27(1-42), 2026.

## Repository Structure

```
.
├── cheap_bootstrap/              # Proposed methods (COfB and COnB)
│   ├── methods.py                # COfB + COnB for linear regression
│   ├── sparse_regression.py      # COfB + COnB for sparse high-dimensional regression
│   └── weak_convex/
│       ├── main_ridge.py         # COfB + COnB for ridge regression (weakly convex)
│       ├── run_experiment.py     # Ridge regression experiment runner
│       └── plot_heatmap_gap.py   # Heatmap and LaTeX table generation
│
├── experiments/                  # Scripts to reproduce all paper results
│   ├── linear_regression/        # Low-dimensional linear & logistic regression
│   │   ├── main.py               # All methods (COfB, COnB, and baselines)
│   │   ├── main_logR.py          # Logistic regression variants of all methods
│   │   ├── run_COfB.py           # Run COfB experiments
│   │   ├── run_COnB.py           # Run COnB experiments
│   │   ├── run_OB.py             # Run Online Bootstrap baseline
│   │   ├── run_plug_in.py        # Run plug-in variance estimator baseline
│   │   ├── run_RS.py             # Run Random Scaling baseline
│   │   ├── run_BM.py             # Run Batch Mean baseline
│   │   ├── run_HiGrad.py         # Run HiGrad baseline
│   │   └── result_analysis*.py   # Result parsing and LaTeX table generation
│   │
│   ├── sparse_high_dim/          # Sparse high-dimensional regression (Section 5.2)
│   │   ├── sparse_linear_regression.py  # COfB + COnB with Lasso model selection
│   │   └── result_analysis.py    # Parse results and generate LaTeX tables
│   │
│   ├── ridge/                    # Ridge regression / weakly convex
│   │   ├── main_ridge.py         # COfB + COnB for ridge regression
│   │   ├── run_experiment.py     # Experiment runner with hyperparameter search
│   │   └── plot_heatmap_gap.py   # Coverage gap heatmap visualization
│   │
│   └── timing/                   # Computational cost experiments
│       └── run_timing.py         # Wall-clock timing for COfB and COnB
│
├── requirements.txt
├── AI_GUIDE.md               # Context file for AI assistants
└── README.md
```

## Methods

### Proposed Methods

- **COfB (Cheap Offline Bootstrap)**: Runs SGD once on original data, then runs B bootstrap SGD replications on resampled data. Uses t-quantiles for confidence interval construction.
- **COnB (Cheap Online Bootstrap)**: Maintains B parallel perturbed SGD trajectories with Exp(1) random weights on gradients. Single-pass, online/streaming method.

Both methods use Polyak-Ruppert averaging (ASGD) with step size `eta_t = eta * (1 + t)^{-alpha}`.

### Baselines (in `experiments/`)

| Method | Description | Reference |
|--------|-------------|-----------|
| Delta / Plug-in | Asymptotic variance estimator | Chen et al. (2020) |
| BM | Batch Means | Chen et al. (2020) |
| RS | Random Scaling | Lee et al. (2022) |
| OB | Online Bootstrap | Fang et al. (2018) |
| HiGrad | Hierarchical incremental gradient descent | Su and Zhu (2018) |

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, joblib, scikit-learn, Pillow

## Quick Start: Using COfB and COnB

```python
import numpy as np
from cheap_bootstrap.methods import run_experiment_COfB, run_experiment_COnB

d = 20        # dimension
n = 10000     # number of samples / SGD steps
eta = 0.05    # initial step size
alpha = 0.501 # step size decay exponent
R = 10        # number of bootstrap replications

x_star = np.ones(d)       # true parameter
x_0 = np.zeros(d)         # initial point
var_epsilon = 1.0          # noise variance

# Run COfB experiment (10 independent trials)
run_experiment_COfB(d, n, eta, alpha, x_star, x_0, R, var_epsilon,
                    cov_a_str='identity', num_trials=10)

# Run COnB experiment
run_experiment_COnB(d, n, eta, alpha, x_star, x_0, B=10, var_epsilon=var_epsilon,
                    cov_a_str='identity', num_trials=10)
```

## Reproducing Paper Results

### Linear Regression Experiments

```bash
cd experiments/linear_regression

# Run each method (results written to Result_*.txt)
python run_COfB.py
python run_COnB.py
python run_plug_in.py
python run_RS.py
python run_BM.py
python run_OB.py
python run_HiGrad.py
```

### Sparse High-Dimensional Regression

```bash
cd experiments/sparse_high_dim
python sparse_linear_regression.py
python result_analysis.py    # generates LaTeX tables
```

### Ridge Regression (Weakly Convex)

```bash
cd experiments/ridge
python run_experiment.py
python plot_heatmap_gap.py   # generates heatmap figures
```

### Timing Experiments

```bash
cd experiments/timing
python run_timing.py
```

## Citation

```bibtex
@article{lam2026cheap,
  title={Cheap Bootstrap for Fast Uncertainty Quantification of Stochastic Gradient Descent},
  author={Lam, Henry and Wang, Zitong},
  journal={Journal of Machine Learning Research},
  volume={27},
  pages={1--42},
  year={2026}
}
```
