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

c = 10.0
print np.log(c)
b = [2,3,1,4]
b.insert(4,5)
print b

d = [b]
d.append([])
print b, d, b in d, [] in d