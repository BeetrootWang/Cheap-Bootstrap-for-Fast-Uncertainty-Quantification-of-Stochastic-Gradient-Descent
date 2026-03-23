"""
Cheap Offline Bootstrap (COfB) and Cheap Online Bootstrap (COnB)
for linear regression with SGD / Averaged SGD.

Reference: [paper title / arxiv link]
"""

import numpy as np
from scipy.stats import t
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def _make_cov(d, cov_a_str):
    """Build mean vector and covariance matrix for the design distribution."""
    mean_a = np.zeros(d)
    if cov_a_str == 'identity':
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)
    else:
        raise ValueError(f"Unknown covariance type: {cov_a_str}")
    return mean_a, cov_a


# ---------------------------------------------------------------------------
# SGD runner (Polyak-Ruppert averaging)
# ---------------------------------------------------------------------------

def run_SGD_LR(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha):
    """
    Run SGD for linear regression and return the averaged iterate.

    Parameters
    ----------
    seed        : int   – random seed for data generation
    x_star      : array – true regression parameter
    x_0         : array – initial point
    n           : int   – number of SGD steps
    eta         : float – initial step size
    var_epsilon : float – noise variance
    mean_a      : array – mean of design distribution
    cov_a       : array – covariance of design distribution
    alpha       : float – step-size decay exponent (eta_t = eta * (1+t)^{-alpha})

    Returns
    -------
    x_out         : averaged iterate (ASGD estimate)
    a_n_history   : (n, d) design matrix
    b_n_history   : list of observed responses
    """
    rng = np.random.default_rng(seed)
    d = len(x_0)
    x_prev = x_0.copy()
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=n)
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = []
    for iter_num in range(n):
        a_n = a_n_history[iter_num, :]
        epsilon_n = epsilon_n_history[iter_num]
        b_n = a_n @ x_star + epsilon_n
        eta_n = eta * (1 + iter_num) ** (-alpha)
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        x_history.append(x_n)
        b_n_history.append(b_n)
    x_out = np.mean(x_history, axis=0)
    return x_out, a_n_history, b_n_history


# ---------------------------------------------------------------------------
# COfB: Cheap Offline Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_CI_COfB(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    """
    Construct confidence interval radii via Cheap Offline Bootstrap (COfB).

    Resamples data R times with replacement, runs SGD on each bootstrap dataset,
    and uses t-quantiles to form the CI.

    Parameters
    ----------
    x_0          : array  – initial point (used for all bootstrap runs)
    n            : int    – number of SGD steps / data points
    R            : int    – number of bootstrap replications (e.g. 3, 5, 10)
    a_n_history  : (n, d) – observed design matrix
    b_n_history  : list   – observed responses
    eta          : float  – initial step size
    alpha        : float  – step-size decay exponent

    Returns
    -------
    CI_radius : array of shape (d,) – half-width of the 95% CI per coordinate
    """
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)
    bootstrap_samples_all = rng_b.integers(0, n, (R, n))
    for r in range(1, R + 1):
        x_prev = x_0.copy()
        x_history = []
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
        for iter_num in range(n):
            a_n = a_n_history[bootstrap_samples[iter_num]]
            b_n = b_n_history[bootstrap_samples[iter_num]]
            eta_n = eta * (1 + iter_num) ** (-alpha)
            x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
            x_prev = x_n
            x_history.append(x_n)
        bootstrap_output_history.append(np.mean(x_history, axis=0))
        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] done')
    t_val = t.ppf(0.975, R - 1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(
            np.sum((np.array(bootstrap_output_history)[:, ii] - bar_X_ii) ** 2 / (R - 1))
        )
        CI_radius.append(t_val * sigma_hat)
    return np.array(CI_radius)


def bootstrap_CI_COfB_last_iterate(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    """
    COfB variant that uses the last SGD iterate (instead of averaged iterate)
    for each bootstrap run. Used internally for the sparse regression experiment.
    """
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)
    bootstrap_samples_all = rng_b.integers(0, n, (R, n))
    for r in range(1, R + 1):
        x_prev = x_0.copy()
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
        for iter_num in range(n):
            a_n = a_n_history[bootstrap_samples[iter_num]]
            b_n = b_n_history[bootstrap_samples[iter_num]]
            eta_n = eta * (1 + iter_num) ** (-alpha)
            x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
            x_prev = x_n
        bootstrap_output_history.append(x_n)
        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] done')
    t_val = t.ppf(0.975, R - 1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(
            np.sum((np.array(bootstrap_output_history)[:, ii] - bar_X_ii) ** 2 / (R - 1))
        )
        CI_radius.append(t_val * sigma_hat)
    return np.array(CI_radius)


# ---------------------------------------------------------------------------
# COnB: Cheap Online Bootstrap
# ---------------------------------------------------------------------------

def run_SGD_LR_COnB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha):
    """
    Run SGD with Cheap Online Bootstrap (COnB) for linear regression.

    Maintains B parallel perturbed SGD trajectories alongside the main run.
    Each perturbed run uses an exponential(1) random weight on the gradient.

    Parameters
    ----------
    seed        : int   – random seed
    x_star      : array – true regression parameter
    x_0         : array – initial point
    n           : int   – number of SGD steps
    B           : int   – number of perturbed runs (e.g. 3, 5, 10)
    eta         : float – initial step size
    var_epsilon : float – noise variance
    mean_a      : array – mean of design distribution
    cov_a       : array – covariance of design distribution
    alpha       : float – step-size decay exponent

    Returns
    -------
    x_out     : averaged iterate (ASGD estimate)
    CI_radius : array of shape (d,) – half-width of the 95% CI per coordinate
    """
    rng = np.random.default_rng(seed)
    d = len(x_0)
    x_prev = x_0.copy()
    x_history = []
    x_B_prev = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_B = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=n)
    W = rng.exponential(1, [n, B])
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    for iter_num in range(n):
        a_n = a_n_history[iter_num, :]
        epsilon_n = epsilon_n_history[iter_num]
        b_n = a_n @ x_star + epsilon_n
        eta_n = eta * (1 + iter_num) ** (-alpha)
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        for ii in range(B):
            x_B[ii, :] = x_B_prev[ii, :] - W[iter_num, ii] * eta_n * (a_n @ x_B_prev[ii, :] - b_n) * a_n
            x_bar_B[ii, :] = (iter_num * x_bar_B_prev[ii, :] + x_B[ii, :]) / (iter_num + 1.0)
            x_B_prev[ii, :] = x_B[ii, :]
            x_bar_B_prev[ii, :] = x_bar_B[ii, :]
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    OB_Estimator = np.mean((x_bar_B - np.reshape(x_out, (1, d))) ** 2, axis=0) * n
    z = t.ppf(0.975, B)
    CI_radius = z * np.sqrt(OB_Estimator) / np.sqrt(n)
    return x_out, CI_radius


# ---------------------------------------------------------------------------
# Per-trial wrappers
# ---------------------------------------------------------------------------

def _trial_COfB(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LR(
        seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha
    )
    CI_radius = bootstrap_CI_COfB(x_0, n, R, a_n_history, b_n_history, eta, alpha)
    mean_len = np.mean(CI_radius * 2)
    std_len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]
    return mean_len, std_len, cover, CI_radius * 2, x_out


def _trial_COnB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LR_COnB(
        seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha
    )
    mean_len = np.mean(CI_radius * 2)
    std_len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]
    return mean_len, std_len, cover, CI_radius * 2, x_out


# ---------------------------------------------------------------------------
# Parallel experiment runners
# ---------------------------------------------------------------------------

def _collect_results(results, num_trials):
    mean_len_history, std_len_history, len_history, cov_history, x_out_history = [], [], [], [], []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])
    return mean_len_history, std_len_history, len_history, cov_history, x_out_history


def run_experiment_COfB(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials,
                        n_jobs=1, out_dir='.'):
    """
    Run COfB experiments in parallel across num_trials random seeds.

    Results are written to <out_dir>/Result_COfB{R}_{d}_{cov_a_str}.txt.
    """
    mean_a, cov_a = _make_cov(d, cov_a_str)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_trial_COfB)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials)
        for seed in range(1, 1 + num_trials)
    )
    mean_len_history, std_len_history, len_history, cov_history, _ = _collect_results(results, num_trials)
    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(f'Coverage: {np.mean(cov_history):.4f}')
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'Result_COfB{R}_{d}_{cov_a_str}.txt')
    with open(out_path, 'a') as f:
        f.write('----->\n')
        f.write(
            f'\t Cov Rate: {np.mean(cov_history):.4f} \t ({np.std(cov_history):.4f})'
            f'\tAvg Len: {np.mean(len_history):.6f} \t ({np.std(len_history) / num_trials:.6f})\n'
        )
        f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha}'
                f' \t # Trials: {num_trials}\n')


def run_experiment_COnB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials,
                        n_jobs=1, out_dir='.'):
    """
    Run COnB experiments in parallel across num_trials random seeds.

    Results are written to <out_dir>/Result_COnB{B}_{d}_{cov_a_str}.txt.
    """
    mean_a, cov_a = _make_cov(d, cov_a_str)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_trial_COnB)(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials)
        for seed in range(1, 1 + num_trials)
    )
    mean_len_history, std_len_history, len_history, cov_history, _ = _collect_results(results, num_trials)
    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(f'Coverage: {np.mean(cov_history):.4f}')
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'Result_COnB{B}_{d}_{cov_a_str}.txt')
    with open(out_path, 'a') as f:
        f.write('----->\n')
        f.write(
            f'\t Cov Rate: {np.mean(cov_history):.4f} \t ({np.std(cov_history):.4f})'
            f'\tAvg Len: {np.mean(len_history):.6f} \t ({np.std(len_history) / num_trials:.6f})\n'
        )
        f.write(f'\t d: {d} \t n: {n} \t B: {B} \t eta_0: {eta} \t alpha: {alpha}'
                f' \t # Trials: {num_trials}\n')
