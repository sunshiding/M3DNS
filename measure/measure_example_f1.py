import numpy as np


def cal_single_instance(x, y, p):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(x.shape[0]):
        if x[i] >= p:
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 1:
                fn += 1
            else:
                tn += 1
    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) <= 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    return f1


def example_f1(x, y, thread=0.5):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the micro f1
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    f1 = 0
    for i in range(n):
        f1 += cal_single_instance(x[i], y[i], thread)
    f1 /= n
    return f1
