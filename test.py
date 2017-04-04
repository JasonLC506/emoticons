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

mu = np.array([ 0.00389246,  0.03770252, -0.04320661], dtype = np.float64 )
phi = np.exp(mu)
print phi/np.sum(phi)