"""
Parse and display results from the sparse regression experiments (COfB and COnB).

Reads result files produced by sparse_regression.py and prints a LaTeX-formatted
table with coverage rates and average CI lengths for each method and covariance
structure.

Usage:
    python result_analysis.py
"""

import re


def get_best_result(result_dir, method_name, s, B):
    """
    Parse result file and return the configuration with the best coverage for T*.

    Parameters
    ----------
    result_dir  : str – directory containing result files
    method_name : str – filename stem, e.g. 's3_High_Result_COfB3_identity'
    s           : int – sparsity level
    B           : int – number of bootstrap replications / perturbed runs

    Returns
    -------
    dict with keys: cov, cov_std, len, len_std, T_cov, T_cov_std, T_len, T_len_std,
                    Tc_cov, Tc_cov_std, Tc_len, Tc_len_std
    """
    filename = f"{result_dir}/{method_name}.txt"
    re_4 = re.compile(r"\D*([\d.]*)\D*([\d.]*)\D*([\d.]*)\D*([\d.]*)")
    re_5 = re.compile(r"\D*([\d.]*)\D*([\d.]*)\D*([\d.]*)\D*([\d.]*)\D*([\d.]*)")

    best = None
    cnt = 0

    with open(filename, 'r') as f:
        for line in f:
            if "----->" in line:
                cnt = 0
            if cnt == 1:
                res = re_4.match(line)
                CurCov = float(res.group(1)) * 100
                CurCov_std = float(res.group(2)) * 100
                CurLen = float(res.group(3))
                CurLen_std = float(res.group(4))
            elif cnt == 2:
                res = re_4.match(line)
                T_Cov = float(res.group(1)) * 100
                T_Cov_std = float(res.group(2)) * 100
                T_Len = float(res.group(3))
                T_Len_std = float(res.group(4))
            elif cnt == 3:
                res = re_4.match(line)
                Tc_Cov = float(res.group(1)) * 100
                Tc_Cov_std = float(res.group(2)) * 100
                Tc_Len = float(res.group(3))
                Tc_Len_std = float(res.group(4))
            elif cnt == 4:
                candidate = {
                    "cov": CurCov, "cov_std": CurCov_std,
                    "len": CurLen, "len_std": CurLen_std,
                    "T_cov": T_Cov, "T_cov_std": T_Cov_std,
                    "T_len": T_Len, "T_len_std": T_Len_std,
                    "Tc_cov": Tc_Cov, "Tc_cov_std": Tc_Cov_std,
                    "Tc_len": Tc_Len, "Tc_len_std": Tc_Len_std,
                }
                if best is None:
                    best = candidate
                elif abs(T_Cov - 95) < abs(best["T_cov"] - 95) and T_Len < 100:
                    best = candidate
            cnt += 1

    return best


if __name__ == "__main__":
    methods = ["COfB", "COnB"]
    cov_list = ["identity", "toeplitz", "equi"]
    result_dir = "results"

    for cov in cov_list:
        best_result = {}
        for s in [3, 15]:
            for method in methods:
                for B in [3, 10]:
                    method_name = f"s{s}_High_Result_{method}{B}_{cov}"
                    key = f"{method} B={B} s={s}"
                    try:
                        best_result[key] = get_best_result(result_dir, method_name, s, B)
                    except FileNotFoundError:
                        print(f"[skip] {method_name}.txt not found")

        print(f"\n{'=' * 60}")
        print(f"Covariance: {cov}")
        print(f"{'=' * 60}")
        for method in methods:
            for B in [3, 10]:
                print("\\hline")
                print(f"\\multirow{{2}}{{*}}{{{method} $B={B}$}}")
                for s in [3, 15]:
                    key = f"{method} B={B} s={s}"
                    if key not in best_result or best_result[key] is None:
                        print(f"  s={s}: no result")
                        continue
                    r = best_result[key]
                    print(
                        f"& $\\in T^*$ & ${r['T_cov']:.2f}$ (${r['T_cov_std']:.2f}$) "
                        f"& ${r['T_len']:.2f}$ (${r['T_len_std']:.2f}$) \\\\"
                    )
                    print(
                        f"& $\\notin T^*$ & ${r['Tc_cov']:.2f}$ (${r['Tc_cov_std']:.2f}$) "
                        f"& ${r['Tc_len']:.2f}$ (${r['Tc_len_std']:.2f}$) \\\\"
                    )
