import os
import numpy as np
from AMF_part.AMF2 import AMF


def load_X():
    X = []
    print(os.getcwd())
    dir = "../checkpoints/cifar-10/end2end/"
    # for file_name in ["features___1", "features___2", "features___4"]:
    for file_name in ["features1_1", "features2_1", "features3_1", "features4_1"]:
        temp_X = np.load(dir + file_name)[:].T
        X.append(temp_X)
    Y = np.load(dir + "features1_2")[:]
    return X, Y


if __name__ == '__main__':
    X, Y = load_X()
    n_cluster = 10
    beta_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    k_list = [50, 75, 100, 150, 200, 250, 300, 500]
    for k in k_list:
        for beta in beta_list:
            print("==============",k,  beta)
            w, S, F = AMF(k=100, beta=10, X=X, Y=Y, cluster=n_cluster)
        # clustering(1, X[i].T, n_cluster, Y, SC=False)