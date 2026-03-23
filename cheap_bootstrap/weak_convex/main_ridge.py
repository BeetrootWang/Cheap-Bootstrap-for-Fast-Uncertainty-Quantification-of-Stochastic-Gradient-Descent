import os
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import t


def rho_from_kappa(kappa, d):
    if kappa < 1:
        raise ValueError("kappa must be >= 1.")
    return (kappa - 1) / (kappa - 1 + d)


def make_equicorr_sigma(d, kappa):
    rho = rho_from_kappa(kappa, d)
    sigma = np.full((d, d), rho)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def simulate_data(seed, x_star, n, var_epsilon, sigma):
    rng = np.random.default_rng(seed)
    d = len(x_star)
    a_n = rng.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n)
    epsilon_n = rng.normal(0, var_epsilon, n)
    b_n = a_n @ x_star + epsilon_n
    return a_n, b_n


def ridge_population_target(sigma, x_star, lambda_reg):
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non negative.")
    d = len(x_star)
    return np.linalg.solve(sigma + lambda_reg * np.eye(d), sigma @ x_star)


def lr_schedule(eta, alpha, iter_num, schedule="power", schedule_params=None):
    if schedule_params is None:
        schedule_params = {}
    if schedule == "power":
        return eta * (1 + iter_num) ** (-alpha)
    if schedule == "constant":
        return eta
    if schedule == "piecewise":
        milestones = schedule_params.get("milestones", [])
        gamma = schedule_params.get("gamma", 1.0)
        factors = schedule_params.get("factors")
        factor = 1.0
        if factors is not None:
            for idx, m in enumerate(milestones):
                if iter_num >= m:
                    factor *= factors[min(idx, len(factors) - 1)]
        else:
            for m in milestones:
                if iter_num >= m:
                    factor *= gamma
        return eta * factor
    if schedule == "cosine":
        T = schedule_params.get("T")
        if T is None or T <= 0:
            raise ValueError("cosine schedule requires positive T in schedule_params.")
        eta_min = schedule_params.get("eta_min", 0.0)
        return eta_min + 0.5 * (eta - eta_min) * (1 + np.cos(np.pi * iter_num / T))
    raise ValueError("schedule must be one of: power, constant, piecewise, cosine.")


def init_sgd(x_0, d, init_mode, rng):
    if init_mode == "provided":
        if x_0 is None:
            raise ValueError("x_0 must be provided when init_mode='provided'.")
        if len(x_0) != d:
            raise ValueError("x_0 dimension must match a_n.")
        return x_0.copy()
    if init_mode == "zeros":
        return np.zeros(d)
    if init_mode == "normal":
        return rng.normal(0, 1, d)
    raise ValueError("init_mode must be one of: provided, zeros, normal.")


def ridge_solve(
    a_n,
    b_n,
    lambda_reg,
    x_0,
    eta,
    alpha,
    sgd_seed=None,
    init_mode="provided",
    schedule="power",
    schedule_params=None,
):
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non negative.")
    n = a_n.shape[0]
    d = a_n.shape[1]
    rng = np.random.default_rng(sgd_seed)
    x_prev = init_sgd(x_0, d, init_mode, rng)
    x_history = []
    if schedule_params is None:
        schedule_params = {}
    if schedule == "cosine" and "T" not in schedule_params:
        schedule_params = {**schedule_params, "T": n}
    for iter_num in range(n):
        a_t = a_n[iter_num, :]
        b_t = b_n[iter_num]
        eta_t = lr_schedule(eta, alpha, iter_num, schedule=schedule, schedule_params=schedule_params)
        grad = (a_t @ x_prev - b_t) * a_t + lambda_reg * x_prev
        x_prev = x_prev - eta_t * grad
        x_history.append(x_prev)
    return np.mean(x_history, axis=0)


def bootstrap_ci_ridge_COfB(
    a_n,
    b_n,
    lambda_reg,
    x_0,
    eta,
    alpha,
    r_boot,
    sgd_seed=None,
    init_mode="provided",
    schedule="power",
    schedule_params=None,
    seed=1,
):
    rng_b = np.random.default_rng(seed)
    n = a_n.shape[0]
    bootstrap_output_history = []
    bootstrap_samples_all = rng_b.integers(0, n, (r_boot, n))
    for r in range(1, r_boot + 1):
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
        init_seed = None if sgd_seed is None else sgd_seed + r
        x_hat = ridge_solve(
            a_n[bootstrap_samples],
            b_n[bootstrap_samples],
            lambda_reg,
            x_0,
            eta,
            alpha,
            sgd_seed=init_seed,
            init_mode=init_mode,
            schedule=schedule,
            schedule_params=schedule_params,
        )
        bootstrap_output_history.append(x_hat)
        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{r_boot}] Done')

    bootstrap_output_history = np.array(bootstrap_output_history)
    t_val = t.ppf(0.975, r_boot - 1)
    bar_x = np.mean(bootstrap_output_history, axis=0)
    sigma_hat = np.sqrt(
        np.sum((bootstrap_output_history - bar_x) ** 2, axis=0) / (r_boot - 1)
    )
    return t_val * sigma_hat


def run_ridge_SGD_COnB(
    a_n,
    b_n,
    x_prev,
    B,
    eta,
    alpha,
    lambda_reg,
    rng,
    schedule="power",
    schedule_params=None,
):
    n = a_n.shape[0]
    d = a_n.shape[1]
    x_history = []
    x_B_prev = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_B = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev, (1, d)), B, axis=0)
    if schedule_params is None:
        schedule_params = {}
    if schedule == "cosine" and "T" not in schedule_params:
        schedule_params = {**schedule_params, "T": n}
    W = rng.exponential(1, (n, B))
    for iter_num in range(n):
        a_t = a_n[iter_num, :]
        b_t = b_n[iter_num]
        eta_t = lr_schedule(eta, alpha, iter_num, schedule=schedule, schedule_params=schedule_params)
        grad = (a_t @ x_prev - b_t) * a_t + lambda_reg * x_prev
        x_prev = x_prev - eta_t * grad
        for ii in range(B):
            grad_b = (a_t @ x_B_prev[ii, :] - b_t) * a_t + lambda_reg * x_B_prev[ii, :]
            x_B[ii, :] = x_B_prev[ii, :] - W[iter_num, ii] * eta_t * grad_b
            x_bar_B[ii, :] = iter_num * x_bar_B_prev[ii, :] / (iter_num + 1.0) + x_B[ii, :] / (iter_num + 1.0)
            x_B_prev[ii, :] = x_B[ii, :]
            x_bar_B_prev[ii, :] = x_bar_B[ii, :]
        x_history.append(x_prev)
    x_out = np.mean(x_history, axis=0)
    ob_estimator = np.mean((x_bar_B - np.reshape(x_out, (1, d))) ** 2, axis=0) * n
    z = t.ppf(0.975, B)
    ci_radius = z * np.sqrt(ob_estimator) / np.sqrt(n)
    return x_out, ci_radius


def main_loop(
    seed,
    x_star,
    x_0,
    n,
    r_boot,
    lambda_reg,
    eta,
    alpha,
    var_epsilon,
    sigma,
    x_lambda,
    num_trials,
    sgd_seed,
    init_mode,
    schedule,
    schedule_params,
):
    print(f'Seed: [{seed}/{num_trials}] ...')
    a_n, b_n = simulate_data(seed, x_star, n, var_epsilon, sigma)
    x_hat = ridge_solve(
        a_n,
        b_n,
        lambda_reg,
        x_0,
        eta,
        alpha,
        sgd_seed=sgd_seed,
        init_mode=init_mode,
        schedule=schedule,
        schedule_params=schedule_params,
    )
    ci_radius = bootstrap_ci_ridge_COfB(
        a_n,
        b_n,
        lambda_reg,
        x_0,
        eta,
        alpha,
        r_boot,
        sgd_seed=sgd_seed,
        init_mode=init_mode,
        schedule=schedule,
        schedule_params=schedule_params,
    )

    mean_len = np.mean(ci_radius * 2)
    std_len = np.std(ci_radius * 2)
    cover = [1 if abs(x_hat[ii] - x_lambda[ii]) <= ci_radius[ii] else 0 for ii in range(len(x_hat))]
    return mean_len, std_len, cover, ci_radius * 2, x_hat


def main_loop_ridge_COnB(
    seed,
    x_star,
    x_0,
    n,
    B,
    lambda_reg,
    eta,
    alpha,
    var_epsilon,
    sigma,
    x_lambda,
    num_trials,
    sgd_seed,
    init_mode,
    schedule,
    schedule_params,
):
    print(f'Seed: [{seed}/{num_trials}] ...')
    a_n, b_n = simulate_data(seed, x_star, n, var_epsilon, sigma)
    rng = np.random.default_rng(seed if sgd_seed is None else sgd_seed)
    x_prev = init_sgd(x_0, len(x_star), init_mode, rng)
    x_hat, ci_radius = run_ridge_SGD_COnB(
        a_n,
        b_n,
        x_prev,
        B,
        eta,
        alpha,
        lambda_reg,
        rng,
        schedule=schedule,
        schedule_params=schedule_params,
    )
    mean_len = np.mean(ci_radius * 2)
    std_len = np.std(ci_radius * 2)
    cover = [1 if abs(x_hat[ii] - x_lambda[ii]) <= ci_radius[ii] else 0 for ii in range(len(x_hat))]
    return mean_len, std_len, cover, ci_radius * 2, x_hat


def main_experiments_ridge_COfB(
    d,
    n,
    lambda_reg,
    x_star,
    x_0,
    eta,
    alpha,
    r_boot,
    var_epsilon,
    kappa,
    num_trials,
    sgd_seed=None,
    init_mode="provided",
    schedule="power",
    schedule_params=None,
    n_jobs=10,
    out_dir="results",
):
    if len(x_star) != d:
        raise ValueError("x_star dimension must match d.")
    sigma = make_equicorr_sigma(d, kappa)
    x_lambda = ridge_population_target(sigma, x_star, lambda_reg)

    results = Parallel(n_jobs=n_jobs)(
        delayed(main_loop)(
            seed,
            x_star,
            x_0,
            n,
            r_boot,
            lambda_reg,
            eta,
            alpha,
            var_epsilon,
            sigma,
            x_lambda,
            num_trials,
            sgd_seed,
            init_mode,
            schedule,
            schedule_params,
        )
        for seed in range(1, 1 + num_trials)
    )

    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_hat_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_hat_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    mean_cov = float(np.mean(cov_history))
    avg_len = float(np.mean(len_history))
    std_len = float(np.std(len_history) / num_trials)
    print(mean_cov)

    os.makedirs(out_dir, exist_ok=True)
    lambda_tag = f"{lambda_reg}".replace(".", "p")
    kappa_tag = f"{kappa}".replace(".", "p")
    out_path = os.path.join(out_dir, f"Result_ridge_d{d}_kappa{kappa_tag}_lambda{lambda_tag}.txt")
    with open(out_path, "a") as f:
        f.write("----->\n")
        f.write(
            f"\t Cov Rate: {mean_cov} \t ({np.std(cov_history)}) "
            f"\tAvg Len: {avg_len} \t ({std_len}) \n"
        )
        f.write(
            f"\t d: {d} \t n: {n} \t kappa: {kappa} \t lambda: {lambda_reg} \t eta: {eta} "
            f"\t # Trials: {num_trials} \t R: {r_boot}\n"
        )

    return {"mean_coverage": mean_cov, "avg_len": avg_len, "std_len": std_len}


def main_experiments_ridge_COnB(
    d,
    n,
    lambda_reg,
    x_star,
    x_0,
    eta,
    alpha,
    B,
    var_epsilon,
    kappa,
    num_trials,
    sgd_seed=None,
    init_mode="provided",
    schedule="power",
    schedule_params=None,
    n_jobs=10,
    out_dir="results",
):
    if len(x_star) != d:
        raise ValueError("x_star dimension must match d.")
    sigma = make_equicorr_sigma(d, kappa)
    x_lambda = ridge_population_target(sigma, x_star, lambda_reg)

    results = Parallel(n_jobs=n_jobs)(
        delayed(main_loop_ridge_COnB)(
            seed,
            x_star,
            x_0,
            n,
            B,
            lambda_reg,
            eta,
            alpha,
            var_epsilon,
            sigma,
            x_lambda,
            num_trials,
            sgd_seed,
            init_mode,
            schedule,
            schedule_params,
        )
        for seed in range(1, 1 + num_trials)
    )

    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_hat_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_hat_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    mean_cov = float(np.mean(cov_history))
    avg_len = float(np.mean(len_history))
    std_len = float(np.std(len_history) / num_trials)
    print(mean_cov)

    os.makedirs(out_dir, exist_ok=True)
    lambda_tag = f"{lambda_reg}".replace(".", "p")
    kappa_tag = f"{kappa}".replace(".", "p")
    out_path = os.path.join(out_dir, f"Result_ridge_COnB_B{B}_d{d}_kappa{kappa_tag}_lambda{lambda_tag}.txt")
    with open(out_path, "a") as f:
        f.write("----->\n")
        f.write(
            f"\t Cov Rate: {mean_cov} \t ({np.std(cov_history)}) "
            f"\tAvg Len: {avg_len} \t ({std_len}) \n"
        )
        f.write(
            f"\t d: {d} \t n: {n} \t kappa: {kappa} \t lambda: {lambda_reg} \t eta: {eta} "
            f"\t # Trials: {num_trials} \t B: {B}\n"
        )

    return {"mean_coverage": mean_cov, "avg_len": avg_len, "std_len": std_len}
