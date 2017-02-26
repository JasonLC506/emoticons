import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean

NCLASS = 6
NRANK = 6
a = np.array([[1,2,3],[2,3,4]])
m = np.mean(a, axis=0)
print a-m
print m in a
print [1,2,3] in a
