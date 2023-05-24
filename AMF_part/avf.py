import numpy as np


def avf(f_list):
    # 输入f为视图损失函数
    # 返回:w为list, k为整数

    # np.insert(arr, obj, values, axis)
    # arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入。
    f_list = np.insert(f_list, 0, 0, 0)
    f_list_temp = f_list
    f_list = sorted(np.sqrt(f_list))
    n = len(f_list)

    k = n-1
    f_sum_p = 0

    for p in range(1, n-1):
        f_sum_p = np.sum(f_list[1: p+1])
        if 1 + f_sum_p / f_list[p+1] <= p < 1 + f_sum_p / f_list[p]:
            k = p
            break

    if k == n-1:
        f_sum_p = np.sum(f_list[1:])

    weight = np.zeros(n)

    for i in range(1, n):
        weight[i] = max(0.0, 1 - (k - 1) * np.sqrt(f_list_temp[i]) / f_sum_p)

    return weight[1:], k