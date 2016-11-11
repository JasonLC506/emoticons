import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math
from scipy.stats import kendalltau
from matplotlib import pyplot as plt

NCLASS = 6
NRANK = 6

def nConstraints():
    N_per = math.factorial(NCLASS)/math.factorial(NCLASS-NRANK)
    N_constr = [0 for i in range(N_per)]
    rank_per = rankPer()

def rankPer():
    N_per = math.factorial(NCLASS) / math.factorial(NCLASS - NRANK)
    rank_per = [[] for i in range(N_per)]


perf = np.arange(2*3).reshape([2,3])
print perf.ndim
