import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from avf import avf
from sklearn.cluster import KMeans
from metrics import cal_clustering_metric
import mkl
import faiss

mkl.get_max_threads()
np.set_printoptions(suppress=True)  # 控制输出精度


def AMF(k, beta, X, Y, cluster=10):
    """
    Args:
        k: 邻居数
        beta: 超参数
        X: 多视图数据X(v)∈d_v*n(v=1,2...l),默认视图数l=6, X为list,里面的每个元素为二维数组
        cluster: 聚类的簇的数量
    Returns: w, S, F
    """
    # 1 初始化w和F
    _, n = X[0].shape
    multi_view_num = len(X)
    w, F = init_w_F(multi_view_num, n, cluster)
    print("Init w and F")
    print("w", w)

    # 循环十次左右收敛
    for iter in range(10):
        print("iter------------------------------------------------------", iter)

        # 2 更新d_ij 公式23/24
        print("claculate_d...")
        d, d_list = claculate_d(X, F, w, n, beta, iter)
        # np.save('distance1.npy', d)
        # # print("hhh")


        # 3 更新统一相似度S 公式22
        print("claculate_S...")
        S = claculate_S(d, n, k)


        # 消融实验
        print("聚类...")
        clustering(S, _, cluster, Y, SC=True)
        # break

        # 4 构造领接矩阵D and 拉普拉斯矩阵L
        print("构造领接矩阵D and 拉普拉斯矩阵L...")
        D = np.zeros((n, n))
        for i in range(0, n):
            D[i, i] = np.sum(S[i, :]) + np.sum(S[:, i])
        L = D - (S + S.T) / 2


        # 5 更新f_v 公式23/24
        print("计算fv...")
        f = np.zeros(len(X))
        for i in range(len(X)):
            d_list[i] = d_list[i] / np.max(d_list[i], axis=1)
            f[i] = np.sum(d_list[i] * S)
        print("fv_list", f)

        # 6 用算法1,输入f_v,更新w
        print("更新w...")
        f = f * np.array([1000, 100, 10, 1])
        w, p = avf(f)
        print("w", w)

        # 7 用公式15更新最小值
        print("更新F...")
        Eigenvalues, Eigenvalue_vectors = np.linalg.eig(L)
        # print(Eigenvalues, Eigenvalue_vectors)
        F = Eigenvalue_vectors[:, -cluster:]

        # print("聚类...")
        # clustering(S, _, cluster, Y, SC=True)

    return w, S, F


def calculate_F(L, c):
    # 7 用公式15更新最小值
    Eigenvalues, Eigenvalue_vectors = np.linalg.eig(L)
    F = Eigenvalue_vectors[:, :c]
    return F


def init_w_F(multi_view_num, n, c):
    # 初始化w
    w = multi_view_num * [1]
    w = w/np.sum(w)  # 满足w的和为1
    # 初始化F
    H = np.random.rand(n, c)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    F = np.dot(u, vh)  # 满足F^TF=I,得到的单位矩阵阶数为c
    return w, F


def claculate_S(d, n, k):
    sort_d = np.sort(d, axis=1)
    d_k = sort_d[:, k:k + 1]
    d_k_matrix = np.repeat(d_k, n, axis=1)
    d_sum = np.sum(sort_d[:, :k], axis=1, keepdims=True)
    d_sum_matrix = np.repeat(d_sum, n, axis=1)
    S = (d_k_matrix - d) / (k * d_k_matrix - d_sum_matrix)
    S[S < 0] = 0
    return S


def claculate_d(X, F, w, n, beta, index):
    d_list = []
    d = np.zeros((n, n))
    for v in range(len(X)):
        d_temp = faiss_calculate_only_distance(X[v].T, n)
        d_list.append(d_temp)
        d += d_temp / (1 - w[v])
    if len(X) == 1:
        d = faiss_calculate_only_distance(X[0].T, n)
        return d, d_list
    if index == 1000:
        d = faiss_calculate_only_distance(X[-1].T, n)
    d += beta * faiss_calculate_only_distance(F, n)
    return d, d_list


def clustering(weights, embedding, n_clusters, labels, SC=True):
    if not SC:
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        print(prediction[:20])
        acc, nmi = cal_clustering_metric(labels, prediction)
        print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
        print('')


    if SC:
        weights = (weights + weights.T) / 2
        # 根据数据集构造相似矩阵 W，度矩阵 D;
        degree_ = np.power(np.sum(weights, axis=1), -0.5)
        degree = np.diag(degree_)
        # 计算拉普拉斯矩阵L，并标准化D-1/2 L D-1/2;
        L = np.dot(np.dot(degree, weights), degree)
        # 计算D-1/2 L D-1/2最小的k个特征值对应的特征向量f;
        _, vectors = np.linalg.eig(L)
        indicator = vectors[:, :n_clusters]
        # 将各个特征向量f组成的矩阵按行标准化,组成特征矩阵 F∈Rn×k;
        indicator = indicator / np.repeat(np.linalg.norm(indicator, ord=2, axis=1, keepdims=True), n_clusters, axis=1)
        km = KMeans(n_clusters=n_clusters).fit(indicator)
        prediction = km.predict(indicator)
        acc, nmi = cal_clustering_metric(labels, prediction)
        print('SC --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end='')
        print('')
    return acc, nmi


def faiss_calculate_only_distance(features, k):
    # mine the topk nearest neighbors for every sample
    # 挖掘每个样本的topk最近邻
    n, dim = features.shape
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    distances, indices = index.search(features, k)  # Sample itself is included

    distances_new = np.zeros((n, n))
    for i in range(n):
        for j in range(k):
            distances_new[i][indices[i][j]] = distances[i][j]
    return distances_new


if __name__ == '__main__':
    pass