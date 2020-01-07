import config as c
import numpy as np
from scipy.io import loadmat

if __name__ == "__main__":
    path = c.save_dir_for_SE + 'tr/tr_'
    num = 3340

    total_cnt = 0
    for i in range(1, num+1):
        print('calculating number:', i)
        data = loadmat(path + str(i) + '.mat')['beam_spec']
        data_trans = data.reshape(-1, c.n_freq)
        total_cnt += data_trans.shape[0]
        if i == 1:
            sum_X_temp = np.sum(np.abs(data_trans), axis=0)
            std_X_temp = np.sum(np.abs(data_trans) ** 2, axis=0)
        else:
            sum_X_temp += np.sum(np.abs(data_trans), axis=0)
            std_X_temp += np.sum(np.abs(data_trans) ** 2, axis=0)
    X_mean = sum_X_temp / total_cnt
    X_std = np.sqrt(std_X_temp / total_cnt - X_mean ** 2)
    print(X_mean.shape[0])
    np.save('abs_spec_mean.npy', X_mean)
    np.save('abs_spec_std.npy', X_std)

