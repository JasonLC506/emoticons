import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean


def cumulate(y, L, K):
    x = np.zeros(L*K, dtype=np.float16).reshape([L, K])
    for i in range(L):
        x[i] = np.sum(y[:i, :], axis=0)
    return x

x = np.arange(6).reshape([2,3])
y = np.arange(12).reshape([4,3])
print x, y
print np.inner(x[1], y[2])
a = np.zeros(10)
print a.shape[0]
print np.log(10.0)
print np.exp(2.0)
print math.exp(2.00000)
