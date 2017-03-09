import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean

# NCLASS = 6
# NRANK = 6
# a = np.array([[1,2,3],[2,3,4]])
# m = np.mean(a, axis=0)
# print a-m
# print m in a
# print [1,2,3] in a
#
# b = 3*2.4/3.5 * a - a
# print b

class A:
    def __init__(self, k):
        self.k = k

a = A(1)
b = A(2)
c = A(3)
L = [a,b]
for item in L:
    print item.k
L.append(c)
c.k = 4
for item in L:
    print item.k

