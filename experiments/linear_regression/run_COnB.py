from main import main_experiments_parallel_COB
import numpy as np
import time

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    n = int(1e5)  # sample size
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    num_trials = 10

    for B in [3]:
        for d in [1000, 2000]:
            start = time.time()
            for eta in [0.02]:
                for cov_a_str in ['identity']:
                    x_star = np.linspace(0, 1, d)  # optimal solution
                    x_0 = np.zeros(d)  # initial guess
                    main_experiments_parallel_COB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials)
            end = time.time()
            tmp_time = (end - start) / 10
            f = open(f"time_COnB{B}.txt", "a")
            print(f'd={d}; time used = {tmp_time:.2f}')
            f.write(f'd={d}; time used = {tmp_time:.2f}\n')
            f.close()
