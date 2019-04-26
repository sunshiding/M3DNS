import numpy as np


def cal_single_instance(x, y, p):
    miss = 0
    for i in range(x.shape[0]):
        if x[i] >= p and y[i] == 0:
            miss += 1
        if x[i] < p and y[i] == 1:
            miss += 1
    return miss


def hamming_loss(x, y, thread=0.5):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the hamming auc
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    miss_label = 0
    for i in range(n):
        miss_label += cal_single_instance(x[i], y[i], thread)
    hl = miss_label * 1.0 / (n * d)
    return hl
