"""
Sparse linear regression experiment: COfB and COnB with Lasso-based model selection.

Two-stage procedure:
  1. Model selection via Lasso to identify the active support T_hat.
  2. Uncertainty quantification (COfB or COnB) on the reduced model restricted to T_hat.

Reference: [paper title / arxiv link]
"""

import os
import numpy as np
from scipy.stats import t
from joblib import Parallel, delayed
from sklearn import linear_model


# ---------------------------------------------------------------------------
# Data generation helper
# ---------------------------------------------------------------------------

def _make_cov_matrix(d, cov_a_str):
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
# Bootstrap CI helper (used internally by COfB)
# ---------------------------------------------------------------------------

def _bootstrap_CI(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    """Run R bootstrap SGD runs and return t-based CI radii."""
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
    d_hat = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d_hat):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(
            np.sum((np.array(bootstrap_output_history)[:, ii] - bar_X_ii) ** 2 / (R - 1))
        )
        CI_radius.append(t_val * sigma_hat)
    return np.array(CI_radius)


# ---------------------------------------------------------------------------
# Per-trial functions
# ---------------------------------------------------------------------------

def _trial_COfB(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed, s):
    """
    Single trial of COfB for sparse linear regression.

    Stage 1: Lasso model selection.
    Stage 2: SGD + cheap offline bootstrap on the selected submodel.
    """
    rng = np.random.default_rng(seed)
    eta = params["eta"]
    alpha = params["alpha"]
    t_threshold = params["t_threshold"]
    lambda_coef = params["lambda_coef"]

    x_star_extend = np.pad(x_star, (0, d - len(x_star)), mode='constant', constant_values=0)
    mean_a, cov_a = _make_cov_matrix(d, cov_a_str)

    # Stage 1: Lasso model selection
    clf = linear_model.Lasso(alpha=lambda_coef * np.log(d) / n, fit_intercept=False, random_state=seed)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=n)
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = a_n_history @ x_star_extend + epsilon_n_history
    clf.fit(a_n_history, b_n_history)
    T_hat = np.where(clf.coef_ > t_threshold)[0]

    # Stage 2: SGD on selected submodel
    x_prev = x_0[T_hat].copy()
    x_history = []
    for iter_num in range(n):
        a_n = a_n_history[iter_num, T_hat]
        b_n = b_n_history[iter_num]
        eta_n = eta * (1 + iter_num) ** (-alpha)
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)

    # Stage 2: COfB CI on selected submodel
    CI_radius_hat = _bootstrap_CI(x_0[T_hat], n, R, a_n_history[:, T_hat], b_n_history, eta, alpha)

    # Map back to full d-dimensional space
    CI_radius = np.zeros(d)
    cover = [1] * d
    output_x = [0.0] * d
    for i, idx in enumerate(T_hat):
        CI_radius[idx] = CI_radius_hat[i]
        if abs(x_out[i] - x_star_extend[idx]) > CI_radius_hat[i]:
            cover[idx] = 0
        output_x[idx] = x_out[i]

    CI_radius_in = CI_radius[:s]
    CI_radius_out = CI_radius[s:]
    return (
        np.mean(CI_radius * 2), np.mean(CI_radius_in * 2), np.mean(CI_radius_out * 2),
        np.std(CI_radius * 2), np.std(CI_radius_in * 2), np.std(CI_radius_out * 2),
        cover, cover[:s], cover[s:], CI_radius * 2, output_x
    )


def _trial_COnB(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed, s):
    """
    Single trial of COnB for sparse linear regression.

    Stage 1: Lasso model selection.
    Stage 2: SGD with B parallel perturbed runs (exponential weights) on the selected submodel.
    """
    rng = np.random.default_rng(seed)
    eta = params["eta"]
    alpha = params["alpha"]
    t_threshold = params["t_threshold"]
    lambda_coef = params["lambda_coef"]

    x_star_extend = np.pad(x_star, (0, d - len(x_star)), mode='constant', constant_values=0)
    mean_a, cov_a = _make_cov_matrix(d, cov_a_str)

    # Stage 1: Lasso model selection
    clf = linear_model.Lasso(alpha=lambda_coef * np.log(d) / n, fit_intercept=False, random_state=seed)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=n)
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = a_n_history @ x_star_extend + epsilon_n_history
    clf.fit(a_n_history, b_n_history)
    T_hat = np.where(clf.coef_ > t_threshold)[0]

    # Stage 2: COnB on selected submodel
    x_prev = x_0[T_hat].copy()
    d_hat = len(T_hat)
    x_history = []
    x_B_prev = np.repeat(np.reshape(x_prev, (1, d_hat)), R, axis=0)
    x_B = np.repeat(np.reshape(x_prev, (1, d_hat)), R, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev, (1, d_hat)), R, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev, (1, d_hat)), R, axis=0)
    W = rng.exponential(1, [n, R])
    for iter_num in range(n):
        a_n = a_n_history[iter_num, T_hat]
        b_n = b_n_history[iter_num]
        eta_n = eta * (1 + iter_num) ** (-alpha)
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        for ii in range(R):
            x_B[ii, :] = x_B_prev[ii, :] - W[iter_num, ii] * eta_n * (a_n @ x_B_prev[ii, :] - b_n) * a_n
            x_bar_B[ii, :] = (iter_num * x_bar_B_prev[ii, :] + x_B[ii, :]) / (iter_num + 1.0)
            x_B_prev[ii, :] = x_B[ii, :]
            x_bar_B_prev[ii, :] = x_bar_B[ii, :]
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    OB_Estimator = np.mean((x_bar_B - np.reshape(x_out, (1, d_hat))) ** 2, axis=0) * n
    z = t.ppf(0.975, R)
    CI_radius_hat = z * np.sqrt(OB_Estimator) / np.sqrt(n)

    # Map back to full d-dimensional space
    CI_radius = np.zeros(d)
    cover = [1] * d
    output_x = [0.0] * d
    for i, idx in enumerate(T_hat):
        CI_radius[idx] = CI_radius_hat[i]
        if abs(x_out[i] - x_star_extend[idx]) > CI_radius_hat[i]:
            cover[idx] = 0
        output_x[idx] = x_out[i]

    CI_radius_in = CI_radius[:s]
    CI_radius_out = CI_radius[s:]
    return (
        np.mean(CI_radius * 2), np.mean(CI_radius_in * 2), np.mean(CI_radius_out * 2),
        np.std(CI_radius * 2), np.std(CI_radius_in * 2), np.std(CI_radius_out * 2),
        cover, cover[:s], cover[s:], CI_radius * 2, output_x
    )


# ---------------------------------------------------------------------------
# Parallel experiment runner (COfB + COnB)
# ---------------------------------------------------------------------------

def _collect_and_write(results, num_trials, s, d, n, R, params, x_star, out_path):
    """Aggregate parallel trial results and write to file."""
    cov_hist = np.array([r[6] for r in results])
    cov_in_hist = np.array([r[7] for r in results])
    cov_out_hist = np.array([r[8] for r in results])
    len_hist = np.array([r[9] for r in results])

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'a') as f:
        f.write('----->\n')
        f.write(
            f'\t All ------ Cov Rate: {np.mean(cov_hist):.4f} \t ({np.std(cov_hist):.4f})'
            f'\tAvg Len: {np.mean(len_hist):.6f} \t ({np.std(len_hist) / num_trials:.6f})\n'
        )
        f.write(
            f'\t In T ----- Cov Rate: {np.mean(cov_in_hist):.4f} \t ({np.std(cov_in_hist):.4f})'
            f'\tAvg Len: {np.mean(len_hist[:, :s]):.6f} \t ({np.std(len_hist[:, :s]) / num_trials:.6f})\n'
        )
        f.write(
            f'\t Out of T - Cov Rate: {np.mean(cov_out_hist):.4f} \t ({np.std(cov_out_hist):.4f})'
            f'\tAvg Len: {np.mean(len_hist[:, s:]):.6f} \t ({np.std(len_hist[:, s:]) / num_trials:.6f})\n'
        )
        f.write(f'\t d: {d} \t n: {n} \t s: {s} \t B: {R} \t # Trials: {num_trials}\n')
        for key, value in params.items():
            f.write(f'\t {key}: {value}')
        f.write('\n')
        f.write('\t True solution: [' + ', '.join(f'{v:.6f}' for v in x_star) + ']\n')


def run_sparse_experiment(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, num_trials, s,
                          n_jobs=12, out_dir='results'):
    """
    Run both COfB and COnB for sparse linear regression in parallel.

    Parameters
    ----------
    d           : int   – ambient dimension
    n           : int   – number of samples / SGD steps
    params      : dict  – {'eta', 'alpha', 't_threshold', 'lambda_coef'}
    x_star      : array – non-zero part of the true parameter (length s)
    x_0         : array – initial point (length d)
    R           : int   – number of bootstrap replications / perturbed runs
    var_epsilon : float – noise variance
    cov_a_str   : str   – covariance type: 'identity', 'toeplitz', or 'equi'
    num_trials  : int   – number of independent repetitions
    s           : int   – sparsity (number of non-zero coefficients)
    n_jobs      : int   – number of parallel workers
    out_dir     : str   – directory for output files
    """
    s_val = len(x_star)

    # COfB
    print(f'\n=== COfB | d={d}, s={s_val}, R={R}, cov={cov_a_str} ===')
    results_cofb = Parallel(n_jobs=n_jobs)(
        delayed(_trial_COfB)(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed, s_val)
        for seed in range(1, 1 + num_trials)
    )
    cofb_path = os.path.join(out_dir, f's{s_val}_High_Result_COfB{R}_{cov_a_str}.txt')
    print(f'---> Writing {cofb_path}')
    _collect_and_write(results_cofb, num_trials, s_val, d, n, R, params, x_star, cofb_path)

    # COnB
    print(f'\n=== COnB | d={d}, s={s_val}, R={R}, cov={cov_a_str} ===')
    results_conb = Parallel(n_jobs=n_jobs)(
        delayed(_trial_COnB)(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed, s_val)
        for seed in range(1, 1 + num_trials)
    )
    conb_path = os.path.join(out_dir, f's{s_val}_High_Result_COnB{R}_{cov_a_str}.txt')
    print(f'---> Writing {conb_path}')
    _collect_and_write(results_conb, num_trials, s_val, d, n, R, params, x_star, conb_path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = 100          # number of samples
    d = 500          # ambient dimension
    var_epsilon = 1
    alpha = 0.501
    num_trials = 500

    x_0 = np.zeros(d)
    os.makedirs('results', exist_ok=True)

    for s, u in [(3, 2), (15, 2)]:
        rng = np.random.default_rng(0)
        x_star = rng.uniform(0, u, s)
        for cov_a_str in ['identity', 'toeplitz', 'equi']:
            for B in [3, 10]:
                for eta in [0.02, 0.03, 0.04, 0.05]:
                    for t_threshold in [0.05, 0.1]:
                        for lambda_coef in [0.01, 0.001]:
                            params = {
                                "eta": eta,
                                "alpha": alpha,
                                "t_threshold": t_threshold,
                                "lambda_coef": lambda_coef,
                            }
                            run_sparse_experiment(
                                d, n, params, x_star, x_0, B, var_epsilon,
                                cov_a_str, num_trials, s,
                                out_dir='results',
                            )
