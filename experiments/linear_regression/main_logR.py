# import packages
import numpy as np
from scipy.stats import t, norm
from joblib import Parallel, delayed

# F for linear regression
#   F(x) = \mathbb{E} [1/2 (a^T x - b)^2]
#        = 1/2 (x-x_star)@cov_a@(x-x_star) + var_epsilon
def F_LR(x, cov_a, x_star, var_epsilon):
    return .5 * (x - x_star) @ cov_a @ (x - x_star) + var_epsilon

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# SGD original loop (a_n, b_n) iid sample from normal
# rng: random generator
# x_prev: initial guess
# n: number of iterations
def run_SGD_LogR_O(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    return x_out, a_n_history, b_n_history

# SGD bootstrap loop
# compute bootstrap confidence interval
def bootstrap_LogR_CI(x_0, n, R, a_n_history, b_n_history, eta, alpha):
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
            a_n = a_n_history[bootstrap_samples[iter_num],:]
            b_n = b_n_history[bootstrap_samples[iter_num]]
            # update learning rate
            eta_n = eta * (1 + iter_num) ** (-alpha)
            # update rule
            x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
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

def run_SGD_LogR_std_O(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
    x_out = x_n
    return x_out, a_n_history, b_n_history

def run_SGD_LogR_plug_in(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha, delta=1e-6):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    hat_A = np.zeros((d,d))
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
        # update hat_A
        hat_A = hat_A + a_n.reshape((d,1)) @ a_n.reshape((1,d)) * sigmoid(a_n @ x_n) * sigmoid(-a_n @ x_n) / n
    x_out = np.mean(x_history, axis=0)

    # Compute \tilde A_n and S_n to get sigma hat
    # Use sigma hat to get CI_radius
    w,V = np.linalg.eig(hat_A)
    W = np.diag(w * (w>delta) + delta * (w<=delta))
    tilde_A_inv = np.linalg.inv(V @ W @ V.T)
    z = norm.ppf(0.975)
    CI_radius = z * np.sqrt(np.diag(tilde_A_inv)) / np.sqrt(n)
    return x_out, CI_radius

def run_SGD_LogR_RS(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    A_t = np.zeros((d,d))
    b_t = np.zeros((d,))
    x_bar = x_prev
    sos = 0
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        x_bar = x_bar * iter_num / (iter_num + 1.) + x_n / (iter_num+1.)
        # update A_t and B_t and sos
        A_t +=  (iter_num+1.)**2 * np.outer(x_bar, x_bar)
        b_t +=  (iter_num+1.)**2 * x_bar
        sos += (iter_num+1.)**2

    # CI for random scaling Estimator
    V_t = (A_t - np.outer(x_bar, b_t) - np.outer(b_t, x_bar) + sos * np.outer(x_bar, x_bar)) / (n*n)
    cv = 6.747 # P(T < cv) = 97.5%
    CI_radius = cv * np.sqrt(np.diag(V_t)) / np.sqrt(n)
    return x_bar, CI_radius

def run_SGD_LogR_BM(seed, x_star, x_prev, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)

    # CI for Batch Mean Estimator
    xk = 0
    x_bar_M = np.mean(x_history[int(np.floor(N ** (1/(1-alpha))))+1:n] , axis=0)
    BM_Estimator = np.zeros((d,d))
    for k in range(M+1):
        ek = int(np.floor(((k+1)* N) ** (1/(1-alpha))))
        nk = ek - xk
        x_bar_nk = np.mean(x_history[xk:ek+1] , axis=0)
        BM_Estimator += nk * (x_bar_nk - x_bar_M).reshape([d,1]) @ (x_bar_nk - x_bar_M).reshape([1,d]) /M
        xk = ek+1
    z = norm.ppf(0.975)
    CI_radius = z * np.sqrt(np.diag(BM_Estimator))/np.sqrt(n)
    return x_out, CI_radius

def run_SGD_logR_OB(seed, x_star, x_prev, n, B, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    x_B_prev = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_B = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    W = rng.exponential(1, [n,B])
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        for ii in range(B):
            x_B[ii,:] = x_B_prev[ii,:] + W[iter_num, ii] * eta_n * sigmoid(-b_n * (a_n @ x_B_prev[ii,:])) * b_n * a_n
            x_bar_B[ii,:] = iter_num * x_bar_B_prev[ii,:] / (iter_num+1.) + x_B[ii,:] / (iter_num+1.)
            x_B_prev[ii,:] = x_B[ii,:]
            x_bar_B_prev[ii,:] = x_bar_B[ii,:]
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    # CI for Online Bootstrap Estimator
    OB_Estimator = np.mean((x_bar_B - np.repeat(np.reshape(x_out,(1,d)), B, axis=0))**2, axis=0) * n
    z = norm.ppf(0.975)
    CI_radius = z * np.sqrt(OB_Estimator)/np.sqrt(n)
    return x_out, CI_radius


def run_SGD_logR_COB(seed, x_star, x_prev, n, B, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    x_B_prev = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_B = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_bar_B_prev = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    x_bar_B = np.repeat(np.reshape(x_prev,(1,d)), B, axis=0)
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)
    W = rng.exponential(1, [n,B])
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # Generate Perturbation
        for ii in range(B):
            x_B[ii,:] = x_B_prev[ii,:] + W[iter_num, ii] * eta_n * sigmoid(-b_n * (a_n @ x_B_prev[ii,:])) * b_n * a_n
            x_bar_B[ii,:] = iter_num * x_bar_B_prev[ii,:] / (iter_num+1.) + x_B[ii,:] / (iter_num+1.)
            x_B_prev[ii,:] = x_B[ii,:]
            x_bar_B_prev[ii,:] = x_bar_B[ii,:]
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)
    # CI for Online Bootstrap Estimator
    OB_Estimator = np.mean((x_bar_B - np.repeat(np.reshape(x_out,(1,d)), B, axis=0))**2, axis=0) * n
    z = t.ppf(0.975,B)
    CI_radius = z * np.sqrt(OB_Estimator)/np.sqrt(n)
    return x_out, CI_radius

def run_SGD_logR_HiGrad22(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    # K=2; B1 = B2 = 2; n0 = n1 = n2 = n/7
    # Naive implementation
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    p_n_history =sigmoid(a_n_history @ x_star)
    b_n_history = np.where(rng.binomial(size=n, n=1, p=p_n_history)>0,1,-1)

    n0 = int(n/7) # n0 = n1
    x_out = []
    # first stage
    for iter_num in range(n0):
        # sample data
        a_n = a_n_history[iter_num, :]
        b_n = b_n_history[iter_num]
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)

    # Second stage; B1=2, n1=n0=n/7
    x_prev_stage = x_prev
    x_3sdg_init = []
    for ii in [0,1]:
        x_history = []
        for iter_num in range(n0*(ii+1), n0*(ii+2)):
            # sample data
            a_n = a_n_history[iter_num, :]
            b_n = b_n_history[iter_num]
            # update learning rate
            eta_n = eta * (1 + iter_num-ii*n0) ** (-alpha)
            # update rule
            x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
            x_prev = x_n
            # recording
            x_history.append(x_n)
        x_prev = x_prev_stage # reset the initialization for the next branch in this stage
        x_3sdg_init.append(x_n)
    # third stage; B2=2, n2=n0=n/7
    x_3sdg_init = np.array(x_3sdg_init)
    for ii in [0,1]:
        for jj in [0,1]:
            x_prev = x_3sdg_init[ii,:]
            x_history = []
            for iter_num in range(n0*(3+2*ii+jj), n0*(4+2*ii+jj)):
                # sample data
                a_n = a_n_history[iter_num, :]
                b_n = b_n_history[iter_num]
                # update learning rate
                eta_n = eta * (1 + iter_num-2*ii*n0-jj*n0-n0) ** (-alpha)
                # update rule
                x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
                x_prev = x_n
                # recording
                x_history.append(x_n)
            x_out.append(np.mean(x_history, axis=0))
    x_out = np.array(x_out)
    x_bar = np.mean(x_out, axis=0)
    # CI for HiGrad Estimator
    HiGrad22_Estimator = []
    Sigma_inv = np.array([[0.48214286, -0.16071429, 0., 0.], [-0.16071429, 0.48214286, 0., 0.], [0., 0., 0.48214286, -0.16071429], [0., 0., -0.16071429, 0.48214286]])
    for ii in range(d):
        mu_x = x_out[:,ii]
        mu_bar = np.ones(4) * x_bar[ii]
        HiGrad22_Estimator.append(np.sqrt( ((112/9) * (mu_x - mu_bar)@Sigma_inv@(mu_x - mu_bar)) / 48))
    tq = t.ppf(0.975, 3)
    CI_radius = tq * np.array(HiGrad22_Estimator)
    return x_bar, CI_radius

# SGD bootstrap loop
# compute bootstrap confidence interval
def bootstrap_LogR_CI_std(x_0, n, R, a_n_history, b_n_history, eta, alpha):
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
            x_n = x_prev + eta_n * sigmoid(-b_n * (a_n @ x_prev)) * b_n * a_n
            x_prev = x_n
            # recording
            x_history.append(x_n)
        bootstrap_output_history.append(x_n)

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

def main_logR_loop(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LogR_O(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)
    CI_radius = bootstrap_LogR_CI(x_0, n, R, a_n_history, b_n_history, eta, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_std(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LogR_std_O(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)
    CI_radius = bootstrap_LogR_CI_std(x_0, n, R, a_n_history, b_n_history, eta, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_plug_in(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LogR_plug_in(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_RS(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LogR_RS(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_BM(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LogR_BM(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_OB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_logR_OB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha)
    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_COB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_logR_COB(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha)
    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_logR_loop_HiGrad22(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    CI_center, CI_radius = run_SGD_logR_HiGrad22(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(CI_center[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(CI_center))]

    return mean_Len, std_Len, cover, CI_radius*2, CI_center

def main_logR_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
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

    Asy_cov = np.eye(d)  # asymptotic covariance matrix

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'LogR_Result_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

def main_logR_experiments_parallel_std(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_std)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'LogR_Result_std_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

def main_logR_experiments_parallel_plug_in(d, n, eta, alpha, x_star, x_0, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_plug_in)(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'LogR_Result_PI_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: N.A. \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

def main_logR_experiments_parallel_RS(d, n, eta, alpha, x_star, x_0, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=10)(delayed(main_logR_loop_RS)(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'logR_Result_RS_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

def main_logR_experiments_parallel_BM(d, n, eta, alpha, x_star, x_0, M_ratio, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    M = int(np.floor(n ** (M_ratio)))-1
    N = int(np.floor(n**(1-alpha)/(M+1)))
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_BM)(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'logR_Result_BM_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t M ratio: {M_ratio} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return


def main_logR_experiments_parallel_OB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_OB)(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'logR_Result_OB{B}_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t B: {B} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

def main_logR_experiments_parallel_COB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_COB)(seed, x_star, x_0, n, B, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'logR_Result_COB{B}_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t B: {B} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.close()

    return

def main_logR_experiments_parallel_HiGrad22(d, n, eta, alpha, x_star, x_0, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
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

    Asy_cov = np.eye(d)  # asymptotic covariance matrix

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=1)(delayed(main_logR_loop_HiGrad22)(seed, x_star, x_0, n, eta, var_epsilon, mean_a , cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])

    for seed in range(1, 1 + num_trials):
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))

    f = open(f'logR_Result_HiGrad22_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: N.A. \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')

    f.close()

    return

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    d = 1  # d = 5,20,100,200
    n = int(1e5)  # sample size
    eta = 1e-2
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    x_star = np.linspace(0, 1, d)  # optimal solution
    x_0 = np.zeros(d)  # initial guess
    R = 2  # number of bootstrap
    num_trials = 500

    for R in [2,5,10]:
        main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)
