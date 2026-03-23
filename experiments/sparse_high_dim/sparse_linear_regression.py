# High dimensional linear regression experiment
# s-sparse assumption
# Lasso screening approach
import numpy as np
from scipy.stats import t, norm
from joblib import Parallel, delayed
from sklearn import linear_model
from numpy import linalg as LA

def run_main_high_dim_experiment_COfB(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed):
    rng = np.random.default_rng(seed)
    x_star_extend = np.pad(x_star, (0, d-len(x_star)), mode='constant', constant_values=0)
    eta = params["eta"]
    alpha = params["alpha"]
    t_threshold = params["t_threshold"]
    lambda_coef = params["lambda_coef"]
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
               cov_a[ii,jj] = r**np.abs(ii-jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d,d)) + (1-r) * np.eye(d)
    ## step 1: Lasso model selection
    #   step 1 gives a candidate support for the true regression parameter
    clf = linear_model.Lasso(alpha = lambda_coef*np.log(d)/n, fit_intercept=False, random_state=seed)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = a_n_history @ x_star_extend + epsilon_n_history
    clf.fit(a_n_history, b_n_history)
    T_hat = np.where(clf.coef_ > t_threshold)[0]
    x_star_hat = x_star_extend[T_hat]
    ## step 2: UQ after model selection
    x_prev = x_0[T_hat]
    x_history = []
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, T_hat]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1+iter_num) ** (-alpha)
        # update rule
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    CI_radius = bootstrap_CI(x_0[T_hat], n, R, a_n_history[:, T_hat], b_n_history, params)

    CI_radius_hat_t = np.array(CI_radius)
    CI_radius = np.zeros(d)
    cover = [1 for _ in range(d)]
    output_x = [0 for _ in range(d)]
    for i,idx in enumerate(T_hat):
        CI_radius[idx] = CI_radius_hat_t[i]
        if abs(x_out[i] - x_star_extend[idx]) > CI_radius_hat_t[i]:
            cover[idx] = 0
        output_x[idx] = x_out[i]

    CI_radius_in = np.array(CI_radius[:s])
    CI_radius_out = np.array(CI_radius[s:])
    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    mean_Len_in = np.mean(CI_radius_in * 2)
    std_Len_in = np.std(CI_radius_in * 2)
    mean_Len_out = np.mean(CI_radius_out * 2)
    std_Len_out = np.std(CI_radius_out * 2)
    cover_in = cover[:s]
    cover_out = cover[s:]


    return mean_Len, mean_Len_in, mean_Len_out, std_Len, std_Len_in, std_Len_out, cover, cover_in, cover_out, CI_radius * 2, output_x

def bootstrap_CI(x_0, n, R, a_n_history, b_n_history, params):
    eta = params["eta"]
    alpha = params["alpha"]
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)  # random generator for bootstrap experiment
    bootstrap_samples_all = rng_b.integers(0, n, (R, n))  # bootstrap_samples[i] is the index of data for i-th iteration
    for r in range(1, R + 1):
        # which is selected uniformly from given data
        # SGD on bootstrap samples
        x_prev = x_0
        x_history = []
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
        for iter_num in range(n):
            # sample bootstrap data
            a_n = a_n_history[bootstrap_samples[iter_num]]
            b_n = b_n_history[bootstrap_samples[iter_num]]
            # update learning rate
            eta_n = eta * (1 + iter_num) ** (-alpha)
            # update rule
            x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
            x_prev = x_n
            # recording
            x_history.append(x_n)
        bootstrap_output_history.append(np.mean(x_history, axis=0))

        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] Done')
    # Compute Radius of CI
    t_val = t.ppf(0.975, R-1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(np.sum( (np.array(bootstrap_output_history)[:, ii] - bar_X_ii)**2 / (R - 1) ) )
        radius_d = t_val * sigma_hat
        CI_radius.append(radius_d)
    CI_radius = np.array(CI_radius)
    return CI_radius


def run_main_high_dim_experiment_COnB(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed):
    rng = np.random.default_rng(seed)
    x_star_extend = np.pad(x_star, (0, d-len(x_star)), mode='constant', constant_values=0)
    eta = params["eta"]
    alpha = params["alpha"]
    t_threshold = params["t_threshold"]
    lambda_coef = params["lambda_coef"]
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
               cov_a[ii,jj] = r**np.abs(ii-jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d,d)) + (1-r) * np.eye(d)
    ## step 1: Lasso model selection
    #   step 1 gives a candidate support for the true regression parameter
    clf = linear_model.Lasso(alpha = lambda_coef*np.log(d)/n, fit_intercept=False, random_state=seed)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = a_n_history @ x_star_extend + epsilon_n_history
    clf.fit(a_n_history, b_n_history)
    T_hat = np.where(clf.coef_ > t_threshold)[0]
    x_star_hat = x_star_extend[T_hat]
    ## step 2: UQ after model selection
    x_prev = x_0[T_hat]
    x_history = []
    # initialization for the perturbed runs
    x_B_prev = np.repeat(np.reshape(x_prev, (1,len(x_prev))), R, axis=0)
    x_B = np.repeat(np.reshape(x_prev, (1, len(x_prev))), R, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev, (1, len(x_prev))), R, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev, (1, len(x_prev))), R, axis=0)
    # pertubation scalars
    W = rng.exponential(1, [n,R])
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, T_hat]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1+iter_num) ** (-alpha)
        # update rule
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        # perturbed runs
        for ii in range(R):
            x_B[ii,:] = x_B_prev[ii,:] - W[iter_num, ii] * eta_n * (a_n @ x_B_prev[ii,:] - b_n) * a_n
            x_bar_B[ii,:] = iter_num * x_bar_B_prev[ii,:] / (iter_num+1.) + x_B[ii,:] / (iter_num+1.)
            x_B_prev[ii,:] = x_B[ii,:]
            x_bar_B_prev[ii,:] = x_bar_B[ii,:]
        # recording [for x_out]
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)

    OB_Estimator = np.mean((x_bar_B - np.repeat(np.reshape(x_out,(1,len(x_out))), R, axis=0))**2, axis=0) * n
    z = t.ppf(0.975,R)
    CI_radius = z * np.sqrt(OB_Estimator)/np.sqrt(n)
    CI_radius_hat_t = np.array(CI_radius)
    CI_radius = np.zeros(d)
    cover = [1 for _ in range(d)]
    output_x = [0 for _ in range(d)]
    for i,idx in enumerate(T_hat):
        CI_radius[idx] = CI_radius_hat_t[i]
        if abs(x_out[i] - x_star_extend[idx]) > CI_radius_hat_t[i]:
            cover[idx] = 0
        output_x[idx] = x_out[i]

    CI_radius_in = np.array(CI_radius[:s])
    CI_radius_out = np.array(CI_radius[s:])
    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    mean_Len_in = np.mean(CI_radius_in * 2)
    std_Len_in = np.std(CI_radius_in * 2)
    mean_Len_out = np.mean(CI_radius_out * 2)
    std_Len_out = np.std(CI_radius_out * 2)
    cover_in = cover[:s]
    cover_out = cover[s:]


    return mean_Len, mean_Len_in, mean_Len_out, std_Len, std_Len_in, std_Len_out, cover, cover_in, cover_out, CI_radius * 2, output_x



def main_high_dim_experiment(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed):
    results = Parallel(n_jobs = 12)(delayed(run_main_high_dim_experiment_COfB)(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed) for seed in range(1, 1+num_trials))

    cov_history = []
    cov_in_history = []
    cov_out_history = []
    mean_len_history = []
    mean_len_in_history = []
    mean_len_out_history = []
    std_len_history = []
    std_len_in_history = []
    std_len_out_history = []
    len_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        mean_len_in_history.append(results[ii][1])
        mean_len_out_history.append(results[ii][2])
        std_len_history.append(results[ii][3])
        std_len_in_history.append(results[ii][4])
        std_len_out_history.append(results[ii][5])
        cov_history.append(results[ii][6])
        cov_in_history.append(results[ii][7])
        cov_out_history.append(results[ii][8])
        len_history.append(results[ii][9])
        x_out_history.append(results[ii][10])
    cov_history = np.array(cov_history)
    cov_in_history = np.array(cov_in_history)
    cov_out_history = np.array(cov_out_history)
    mean_len_history = np.array(mean_len_history)
    mean_len_in_history = np.array(mean_len_in_history)
    mean_len_out_history = np.array(mean_len_out_history)
    std_len_history = np.array(std_len_history)
    std_len_in_history = np.array(std_len_in_history)
    std_len_out_history = np.array(std_len_out_history)
    len_history = np.array(len_history)
    x_out_history = np.array(x_out_history)

    print(f"---> Writing: COfB_HighDimResult/s{s}_u{u}_High_Result_COnB{R}_{cov_a_str}.txt")
    f = open(f"COfB_HighDimResult/s{s}_u{u}_High_Result_COfB{R}_{cov_a_str}.txt", "a")
    f.write('----->\n')
    f.write(
        f'\t All ------ Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history) / num_trials}) \n')
    f.write(
        f'\t In T ----- Cov Rate: {np.mean(cov_in_history)} \t ({np.std(cov_in_history)}) \tAvg Len: {np.mean(len_history[:,:s])} \t ({np.std(len_history[:,:s]) / num_trials}) \n'
    )
    f.write(
        f'\t Out of T - Cov Rate: {np.mean(cov_out_history)} \t ({np.std(cov_out_history)}) \tAvg Len: {np.mean(len_history[:,s:])} \t ({np.std(len_history[:,s:]) / num_trials}) \n'
    )
    f.write(f'\t d: {d} \t n: {n} \t s: {s} \t B: {R} \t # Trials: {num_trials}\n')
    for key, value in params.items():
        f.write(f'\t {key}: {value}')
    f.write('\n')
    f.write(f'\t True solution:           [')
    for ii in range(s):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write('...]\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(10):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write('...]\n')
    f.close()

    # COnB experiment
    results = Parallel(n_jobs=12)(
        delayed(run_main_high_dim_experiment_COnB)(d, n, params, x_star, x_0, R, var_epsilon, cov_a_str, seed) for seed
        in range(1, 1 + num_trials))

    cov_history = []
    cov_in_history = []
    cov_out_history = []
    mean_len_history = []
    mean_len_in_history = []
    mean_len_out_history = []
    std_len_history = []
    std_len_in_history = []
    std_len_out_history = []
    len_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        mean_len_in_history.append(results[ii][1])
        mean_len_out_history.append(results[ii][2])
        std_len_history.append(results[ii][3])
        std_len_in_history.append(results[ii][4])
        std_len_out_history.append(results[ii][5])
        cov_history.append(results[ii][6])
        cov_in_history.append(results[ii][7])
        cov_out_history.append(results[ii][8])
        len_history.append(results[ii][9])
        x_out_history.append(results[ii][10])
    cov_history = np.array(cov_history)
    cov_in_history = np.array(cov_in_history)
    cov_out_history = np.array(cov_out_history)
    mean_len_history = np.array(mean_len_history)
    mean_len_in_history = np.array(mean_len_in_history)
    mean_len_out_history = np.array(mean_len_out_history)
    std_len_history = np.array(std_len_history)
    std_len_in_history = np.array(std_len_in_history)
    std_len_out_history = np.array(std_len_out_history)
    len_history = np.array(len_history)
    x_out_history = np.array(x_out_history)
    print(f"---> Writing: COnB_HighDimResult/s{s}_u{u}_High_Result_COnB{R}_{cov_a_str}.txt")
    f = open(f"COnB_HighDimResult/s{s}_u{u}_High_Result_COnB{R}_{cov_a_str}.txt", "a")
    f.write('----->\n')
    f.write(
        f'\t All ------ Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history) / num_trials}) \n')
    f.write(
        f'\t In T ----- Cov Rate: {np.mean(cov_in_history)} \t ({np.std(cov_in_history)}) \tAvg Len: {np.mean(len_history[:, :s])} \t ({np.std(len_history[:, :s]) / num_trials}) \n'
    )
    f.write(
        f'\t Out of T - Cov Rate: {np.mean(cov_out_history)} \t ({np.std(cov_out_history)}) \tAvg Len: {np.mean(len_history[:, s:])} \t ({np.std(len_history[:, s:]) / num_trials}) \n'
    )
    f.write(f'\t d: {d} \t n: {n} \t s: {s} \t B: {R} \t # Trials: {num_trials}\n')
    for key, value in params.items():
        f.write(f'\t {key}: {value}')
    f.write('\n')
    f.write(f'\t True solution:           [')
    for ii in range(s):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write('...]\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(10):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write('...]\n')
    f.close()

    return

if __name__ == "__main__":
    n = 100 # sample size
    d = 500 # dimension
    # generate a random non-zero regression coefficient
    x_0 = np.zeros(d)
    var_epsilon = 1
    alpha = 0.501
    num_trials = 500
    for s, u in [(15, 2)]:
        rng = np.random.default_rng(0)
        x_star = rng.uniform(0, u, s)  # true regression parameter for first s dimensions comes from a uniform distribution on [0,2]
        for cov_a_str in ['equi']:
            for B in [3,10]:
                for eta in np.linspace(0.02, 0.08, 10):
                    for t_threshold in [0.1]:
                        for lambda_coef in [0.001]:
                            params = {
                                "eta": eta,
                                "alpha": alpha,
                                "t_threshold": t_threshold,
                                "lambda_coef": lambda_coef
                            }
                            main_high_dim_experiment(d, n, params, x_star, x_0, B, var_epsilon, cov_a_str,
                                     num_trials)
