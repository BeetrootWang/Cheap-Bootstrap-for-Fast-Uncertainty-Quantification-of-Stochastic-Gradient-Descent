"""
Timing experiment for COfB and COnB in high dimensions (d = 1000, 2000).

Measures wall-clock time per trial for each method at large d to demonstrate
computational efficiency. Results are written to time_COfB{B}.txt and
time_COnB{B}.txt.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'cheap_bootstrap'))
from methods import run_experiment_COfB, run_experiment_COnB
import numpy as np
import time

if __name__ == '__main__':
    var_epsilon = 1
    n = int(1e5)
    alpha = 0.501
    num_trials = 10

    for B in [3]:
        for d in [1000, 2000]:
            x_star = np.linspace(0, 1, d)
            x_0 = np.zeros(d)

            # COfB timing
            start = time.time()
            for eta in [0.02]:
                for cov_a_str in ['identity']:
                    run_experiment_COfB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials)
            end = time.time()
            tmp_time = (end - start) / num_trials
            print(f'COfB  d={d}; avg time per trial = {tmp_time:.2f}s')
            with open(f'time_COfB{B}.txt', 'a') as f:
                f.write(f'd={d}; avg time per trial = {tmp_time:.2f}s\n')

            # COnB timing
            start = time.time()
            for eta in [0.02]:
                for cov_a_str in ['identity']:
                    run_experiment_COnB(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials)
            end = time.time()
            tmp_time = (end - start) / num_trials
            print(f'COnB  d={d}; avg time per trial = {tmp_time:.2f}s')
            with open(f'time_COnB{B}.txt', 'a') as f:
                f.write(f'd={d}; avg time per trial = {tmp_time:.2f}s\n')
