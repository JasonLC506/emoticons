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

def nConstraints():
    N_per = math.factorial(NCLASS)/math.factorial(NCLASS-NRANK)
    N_constr = [0 for i in range(N_per)]
    rank_per = rankPer()

def rankPer():
    N_per = math.factorial(NCLASS) / math.factorial(NCLASS - NRANK)
    rank_per = [[] for i in range(N_per)]

### test ###
rank_test = [[0,1,2,3] for i in range(9)]
rank_test.append([0,3,1,2])
rank_pred = [[0,1,2,3] for i in range(10)]
# recall_pair, Nsamp_pair = recallPair(rank_pred, rank_test)
# recall_pair = np.array(recall_pair)
# recall_pair_mask = np.ma.masked_invalid(recall_pair)
# g_mean_pair = gmean(recall_pair_mask, axis=None)
# print recall_pair
# print recall_pair_mask
# print Nsamp_pair
# print g_mean_pair
print perfMeasure(rank_pred, rank_test, rankopt = True)