# Process the result of numerical experiments
import numpy as np
cnt = 0
target_n = int(1e5)
method_name = 'OB'
for Cov in ["identity", "toeplitz", "equi"]:
    print(Cov)
    for B in [100, 200]:
        result_str = f'{method_name} ($B={B}$) '
        for target_d in [5,20,200]:
            best_cov = 0 # Corresponds to R = 3,5,10,50
            best_cov_var = 0
            best_len = 1e5
            best_len_var = 0
            filename = f'./results_jan_2023/logR_Result_{method_name}{B}_{target_d}_{Cov}.txt'
            # print(filename)
            f = open(filename, 'r')
            R_or_Ratio = 'something wrong'
            for line in f:
                if '-->' in line:
                    cnt = 1
                    if R_or_Ratio!='something wrong' and 100*abs(cov-0.95) + Len < 100*abs(best_cov - 0.95) + best_len:
                        best_cov = cov
                        best_cov_var = cov_var / np.sqrt(target_n)
                        best_len = Len
                        best_len_var = Len_var
                elif cnt == 1:
                    cnt = 2
                    # This line contains
                    # Coverage rate and average length
                    tmp = line.split(" ")
                    cov = float(tmp[3])
                    cov_var = float(tmp[5].strip('()'))
                    Len = float(tmp[8])
                    Len_var = float(tmp[10].strip('()'))
                elif cnt == 2:
                    cnt = 3
                    tmp = line.split()
                    d = int(tmp[1])
                    n = int(tmp[3])
                    if 'BM' in filename:
                        if n!= target_n:
                            continue
                        R_or_Ratio = tmp[5+1]
                        eta_0 = float(tmp[7+1])
                        alpha = float(tmp[9+1])
                        num_trials = int(tmp[12+1])
                    else:
                        if n!= target_n:
                            cov=0
                            Len=1e5
                            continue
                        R_or_Ratio = tmp[5]
                        eta_0 = float(tmp[7])
                        alpha = float(tmp[9])
                        num_trials = int(tmp[12])
            if R_or_Ratio != 'something wrong' and 100 * abs(cov - 0.95) + Len < 100 * abs(
                    best_cov - 0.95) + best_len:
                best_cov = cov
                best_cov_var = cov_var / np.sqrt(target_n)
                best_len = Len
                best_len_var = Len_var
            result_str = result_str + f'& ${best_cov*100:.2f}$ (${best_cov_var*100:.2f}$) & ${best_len*100:.2f}$ (${best_len_var*100:.2f}$)'
            f.close()

        result_str = result_str+' \\\\ \hline'
        print(result_str)