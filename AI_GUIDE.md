# AI Assistant Guide

This file provides context for AI assistants (Claude, ChatGPT, Copilot, etc.) helping users understand or modify this codebase.

## What This Repo Is

This is the code release for a **JMLR paper** on uncertainty quantification for stochastic gradient descent (SGD). The paper proposes two methods — **COfB** and **COnB** — for constructing confidence intervals around SGD estimates, and compares them against several baselines.

**Paper:** *Cheap Bootstrap for Fast Uncertainty Quantification of Stochastic Gradient Descent* by Henry Lam and Zitong Wang. JMLR, Volume 27, 2026.

## Key Concepts

### The Problem
Given a streaming/online optimization problem solved via SGD with Polyak-Ruppert averaging (ASGD), we want to construct confidence intervals for the true parameter `x*` using the averaged SGD output `x_bar`. The challenge is doing this **cheaply** — without expensive covariance estimation or many independent SGD runs.

### Proposed Methods

**COfB (Cheap Offline Bootstrap):**
1. Run SGD once on original data → get `x_bar` and save the data `(a_i, b_i)`.
2. Resample B bootstrap datasets (with replacement) from the saved data.
3. Run SGD on each bootstrap dataset → get B bootstrap estimates.
4. Use t-quantile of the bootstrap distribution to form confidence intervals.
- Key parameter: `R` (number of bootstrap replications, typically 3-10).

**COnB (Cheap Online Bootstrap):**
1. Run SGD once, but maintain B **parallel perturbed copies** alongside.
2. Each perturbed copy multiplies the gradient by an independent `Exp(1)` random weight.
3. Use the spread of perturbed estimates to form confidence intervals via t-quantile.
- Key parameter: `B` (number of perturbed runs, typically 3-10).
- Advantage: single-pass, no data storage needed.

### SGD Setup
- Step size: `eta_t = eta * (1 + t)^{-alpha}` with `alpha = 0.501` (Polyak-Ruppert regime).
- Output: Polyak-Ruppert average `x_bar = (1/n) * sum(x_t)`.
- CI: `x_bar +/- t_{0.975, B-1} * sigma_hat` where `sigma_hat` is estimated from bootstrap/perturbed runs.

### Baselines (in `experiments/`)
- **Delta / Plug-in**: Chen et al. (2020) — estimate the asymptotic covariance using eigenvalue regularization.
- **BM (Batch Means)**: Chen et al. (2020) — split the SGD trajectory into batches, estimate variance from batch means.
- **RS (Random Scaling)**: Lee et al. (2022) — weighted averaging with random signs.
- **OB (Online Bootstrap)**: Fang et al. (2018) — similar to COnB but uses **normal** quantile `z_{0.975}` instead of **t-quantile** `t_{0.975, B}`.
- **HiGrad**: Su and Zhu (2018) — hierarchical incremental gradient descent with K=2 stages.

### Important Distinction: COnB vs OB
These are very similar in implementation! Both maintain B perturbed SGD copies with `Exp(1)` weights. The key difference:
- **COnB** (proposed): uses `t.ppf(0.975, B)` — t-distribution quantile.
- **OB** (baseline): uses `norm.ppf(0.975)` — normal quantile.
In code, look for `t.ppf` vs `norm.ppf` to tell them apart.

## Code Architecture

### `cheap_bootstrap/` — Clean, reusable implementations
- `methods.py`: Core COfB + COnB for linear regression. Key functions:
  - `run_SGD_LR()` — SGD with Polyak-Ruppert averaging
  - `bootstrap_CI_COfB()` — COfB confidence interval computation
  - `run_SGD_LR_COnB()` — COnB (SGD + B perturbed copies)
  - `run_experiment_COfB()` / `run_experiment_COnB()` — parallel experiment runners
- `sparse_regression.py`: COfB + COnB for high-dimensional sparse regression (Lasso model selection + SGD inference).
- `weak_convex/main_ridge.py`: COfB + COnB for ridge regression (weakly convex).

### `experiments/` — Full paper reproduction
- `linear_regression/main.py`: **All** methods (proposed + baselines) in one file (~1500 lines). Each method has:
  - `run_SGD_LR_*()` — the SGD variant
  - `main_loop_*()` — single trial wrapper
  - `main_experiments_parallel_*()` — parallel runner that writes results to file
- `linear_regression/main_logR.py`: Same methods but for **logistic regression**.
- `linear_regression/run_*.py`: Experiment scripts that call into `main.py` / `main_logR.py`.
- `linear_regression/result_analysis*.py`: Parse result text files and generate LaTeX tables.
- `sparse_high_dim/`: Sparse regression experiment with Lasso screening.
- `ridge/`: Ridge regression / weakly convex experiment.
- `timing/`: Wall-clock timing comparison.

## Covariance Structures
Experiments are run under three covariance structures for the data `a_i`:
- **identity**: `Cov(a) = I`
- **toeplitz**: `Cov(a)[i,j] = 0.5^|i-j|`
- **equicorrelated**: `Cov(a) = 0.2 * 11^T + 0.8 * I`

## Result File Format
Experiment runners write results to `.txt` files with this format:
```
----->
     All ------ Cov Rate: 0.95  (0.02)  Avg Len: 1.23  (0.05)
     In T ----- Cov Rate: 0.93  (0.03)  Avg Len: 0.89  (0.04)   # (sparse only)
     Out of T - Cov Rate: 0.99  (0.01)  Avg Len: 0.01  (0.00)   # (sparse only)
     d: 500   n: 100   s: 3   B: 10   # Trials: 500
     eta: 0.05  alpha: 0.501  ...
```
- `Cov Rate`: empirical coverage probability (target: 0.95).
- `Avg Len`: average confidence interval length (smaller is better, given good coverage).
- `In T` / `Out of T`: coverage/length for coordinates inside/outside the true support (sparse regression only).

## Common Tasks

**"Run COfB on my data"** → See `cheap_bootstrap/methods.py`, function `run_experiment_COfB()`.

**"Reproduce Table X"** → Check `experiments/linear_regression/run_*.py` for the relevant method, then `result_analysis*.py` for LaTeX output.

**"Compare COfB vs a new baseline"** → Add your method to `experiments/linear_regression/main.py` following the pattern of existing `run_SGD_LR_*()` / `main_loop_*()` / `main_experiments_parallel_*()` functions.

**"Adapt to a new loss function"** → The SGD update rule is in the inner loop of `run_SGD_LR_O()`. Replace `(a_n @ x_prev - b_n) * a_n` with the gradient of your loss. The bootstrap/perturbation logic is independent of the loss.
