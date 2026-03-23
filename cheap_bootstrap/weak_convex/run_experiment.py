from main_ridge import main_experiments_ridge_COfB, main_experiments_ridge_COnB
import numpy as np
import time
import os
import csv


if __name__ == "__main__":
    var_epsilon = 1
    n = int(1e5)
    num_trials = 10
    kappa_list = [10, 100, 1000]
    lambda_list = [0, 1e-4, 1e-3, 1e-2]
    # eta_list = list(np.linspace(0.05, 0.5, 8))
    # eta_list = [0.02, 0.05, 0.1, 0.2, 0.5]
    eta_list = [0.1]
    alpha = 0.501
    sgd_seed = 1
    init_mode = "provided"
    r_boot = 5
    B = 5
    d_list = [20]
    target_cov = 0.95

    os.makedirs("results", exist_ok=True)

    def select_best_eta(run_fn):
        best = None
        for eta in eta_list:
            metrics = run_fn(eta)
            gap = abs(metrics["mean_coverage"] - target_cov)
            candidate = {
                "eta": eta,
                "mean_coverage": metrics["mean_coverage"],
                "avg_len": metrics["avg_len"],
                "std_len": metrics["std_len"],
                "gap": gap,
            }
            if best is None:
                best = candidate
                continue
            best_good = best["avg_len"] <= 10
            cand_good = candidate["avg_len"] <= 10
            if cand_good and not best_good:
                best = candidate
                continue
            if best_good and not cand_good:
                continue
            if best_good and cand_good:
                if candidate["gap"] < best["gap"] or (
                    candidate["gap"] == best["gap"] and candidate["avg_len"] < best["avg_len"]
                ):
                    best = candidate
                continue
            if candidate["avg_len"] < best["avg_len"]:
                best = candidate
        return best

    summary_paths = {
        "COfB": os.path.join("results", "summary_ridge_COfB.csv"),
        "COnB": os.path.join("results", "summary_ridge_COnB.csv"),
    }
    headers = [
        "method",
        "d",
        "n",
        "kappa",
        "lambda_reg",
        "eta_best",
        "mean_coverage",
        "avg_len",
        "std_len",
        "gap_to_0p95",
        "alpha",
        "num_trials",
        "r_boot",
        "B",
    ]

    for method, path in summary_paths.items():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    for d in d_list:
        x_star = np.linspace(0, 1, d)
        x_0 = np.zeros(d)
        for lambda_reg in lambda_list:
            for kappa in kappa_list:
                start = time.time()
                best_c_of_b = select_best_eta(
                    lambda eta: main_experiments_ridge_COfB(
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
                        sgd_seed=sgd_seed,
                        init_mode=init_mode,
                    )
                )
                end = time.time()
                print(
                    f"COfB d={d} kappa={kappa} lambda={lambda_reg} "
                    f"eta={best_c_of_b['eta']} coverage={best_c_of_b['mean_coverage']:.4f} "
                    f"avg_len={best_c_of_b['avg_len']:.4f} time={(end - start):.2f}s"
                )
                with open(summary_paths["COfB"], "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(
                        {
                            "method": "COfB",
                            "d": d,
                            "n": n,
                            "kappa": kappa,
                            "lambda_reg": lambda_reg,
                            "eta_best": best_c_of_b["eta"],
                            "mean_coverage": best_c_of_b["mean_coverage"],
                            "avg_len": best_c_of_b["avg_len"],
                            "std_len": best_c_of_b["std_len"],
                            "gap_to_0p95": best_c_of_b["gap"],
                            "alpha": alpha,
                            "num_trials": num_trials,
                            "r_boot": r_boot,
                            "B": "",
                        }
                    )

                start = time.time()
                best_c_on_b = select_best_eta(
                    lambda eta: main_experiments_ridge_COnB(
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
                        sgd_seed=sgd_seed,
                        init_mode=init_mode,
                    )
                )
                end = time.time()
                print(
                    f"COnB d={d} kappa={kappa} lambda={lambda_reg} "
                    f"eta={best_c_on_b['eta']} coverage={best_c_on_b['mean_coverage']:.4f} "
                    f"avg_len={best_c_on_b['avg_len']:.4f} time={(end - start):.2f}s"
                )
                with open(summary_paths["COnB"], "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(
                        {
                            "method": "COnB",
                            "d": d,
                            "n": n,
                            "kappa": kappa,
                            "lambda_reg": lambda_reg,
                            "eta_best": best_c_on_b["eta"],
                            "mean_coverage": best_c_on_b["mean_coverage"],
                            "avg_len": best_c_on_b["avg_len"],
                            "std_len": best_c_on_b["std_len"],
                            "gap_to_0p95": best_c_on_b["gap"],
                            "alpha": alpha,
                            "num_trials": num_trials,
                            "r_boot": "",
                            "B": B,
                        }
                    )
