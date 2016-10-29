import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math
from scipy.stats import kendalltau

NCLASS = 6
NRANK = 6

def nConstraints():
    N_per = math.factorial(NCLASS)/math.factorial(NCLASS-NRANK)
    N_constr = [0 for i in range(N_per)]
    rank_per = rankPer()

def rankPer():
    N_per = math.factorial(NCLASS) / math.factorial(NCLASS - NRANK)
    rank_per = [[] for i in range(N_per)]



rank1 = [[1,2,3,4]]
rank2 = [[2,1,4,-1]]
print KendallTau(rank1, rank2)

