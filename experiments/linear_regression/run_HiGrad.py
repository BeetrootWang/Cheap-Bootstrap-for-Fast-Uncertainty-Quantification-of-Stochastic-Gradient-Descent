from main import main_experiments_parallel_HiGrad22
from main_logR import main_logR_experiments_parallel_HiGrad22
import numpy as np
import time

if __name__ == "__main__":
    var_epsilon = 1
    n = int(1e5)
    alpha = 0.501
    num_trials = 10
    for d in [1000, 2000]:
        start = time.time()
        for eta in [0.02]:
            for cov_a_str in ['identity']:
                x_star = np.linspace(0,1,d)
                x_0 = np.zeros(d)
                main_experiments_parallel_HiGrad22(d, n, eta, alpha, x_star, x_0, var_epsilon, cov_a_str, num_trials)
        end = time.time()
        tmp_time = (end - start) / 10
        f = open(f"time_HiGrad.txt", "a")
        print(f'd={d}; time used = {tmp_time:.2f}')
        f.write(f'd={d}; time used = {tmp_time:.2f}\n')
        f.close()
