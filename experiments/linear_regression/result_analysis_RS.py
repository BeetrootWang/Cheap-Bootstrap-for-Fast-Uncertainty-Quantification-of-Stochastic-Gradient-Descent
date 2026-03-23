# Process the result of numerical experiments
import numpy as np
cnt = 0
for Cov in ['identity', 'toeplitz', 'equi']:
    result_str = 'RS '
    for target_d in [5,20,200]:
        target_n = int(1e5)
        best_cov = 0 # Corresponds to R = 3,5,10,50
        best_cov_var = 0
        best_len = 1e5
        best_len_var = 0
        filename = f'./logR_Result_RS_{target_d}_{Cov}.txt'
        # print(filename)
        f = open(filename, 'r')
        R_or_Ratio = 'something wrong'
        for line in f:
            # print(line)
            if '-->' in line:
                cnt = 1
                if R_or_Ratio!='something wrong' and abs(cov-0.95) + Len < abs(best_cov - 0.95) + best_len:
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
                # print(cov, Len, best_cov, best_len)
                if abs(cov - 0.95) + 1 * Len < abs(best_cov - 0.95) + 1 * best_len:
                    best_cov = cov
                    best_cov_var = cov_var / np.sqrt(target_n)
                    best_len = Len
                    best_len_var = Len_var
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
                    # R_or_Ratio = tmp[5]
                    eta_0 = float(tmp[5])
                    alpha = float(tmp[7])
                    num_trials = int(tmp[10])

        result_str = result_str + f'& ${best_cov*100:.2f}$ (${best_cov_var*100:.2f}$) & ${best_len*100:.2f}$ (${best_len_var*100:.2f}$)'
        f.close()

    result_str = result_str+'\\\\ \hline'
    print(Cov)
    print(result_str)