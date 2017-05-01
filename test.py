import numpy as np
#from logRegFeatureEmotion import *
#from sklearn.model_selection import KFold
#import itertools
#import math
#from scipy.stats import kendalltau
from matplotlib import pyplot as plt
#from scipy.stats.mstats import gmean
#import DecisionTreeWeight_Bordar as dtb
#from readSushiData import readSushiData
import sys
import copy
from scipy.optimize import curve_fit

# def cumulate(y, L, K):
#     x = np.zeros(L*K, dtype=np.float16).reshape([L, K])
#     for i in range(L):
#         x[i] = np.sum(y[:i, :], axis=0)
#     return x

# x, y = readSushiData()

# m = np.array([9,2,1]).reshape([1,3])
# print np.repeat(m, 3, axis=0)



# def crossValidateTest(x, y, cv=5, alpha=0.0, rank_weight=False, stop_criterion_mis_rate=None, stop_criterion_min_node=1,
#                   stop_criterion_gain=0.0, prune_criteria=0):
#
#
#     results = {"perf": []}
#
#     # cross validation #
#     np.random.seed(1100)
#     kf = KFold(n_splits=cv, shuffle=True, random_state=0)  ## for testing fixing random_state
#     for train, test in kf.split(x):
#         x_train = x[train, :]
#         y_train = y[train, :]
#         x_test = x[test, :]
#         y_test = y[test, :]
#
#         y_pred_single = dtb.DecisionTree().nodeResult(y_train,None)
#         print "simple Bordar aggregation result: ", y_pred_single
#         y_pred = np.repeat(y_pred_single.reshape([1,y_pred_single.shape[0]]), y_test.shape[0], axis=0)
#         results["perf"].append(LogR.perfMeasure(y_pred, y_test, rankopt=True))
#
#     for key in results.keys():
#         item = np.array(results[key])
#         mean = np.nanmean(item, axis=0)
#         std = np.nanstd(item, axis=0)
#         results[key] = [mean, std]
#
#     return results
#
# print crossValidateTest(x,y)

a = np.ones([10,3,3])
b = a.tolist()
print b