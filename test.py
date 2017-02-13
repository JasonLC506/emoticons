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



# ### test ###
# # rank_test = [[0,1,2,3] for i in range(9)]
# # rank_test.append([0,3,1,2])
# # rank_pred = [[0,1,2,3] for i in range(10)]
# # # recall_pair, Nsamp_pair = recallPair(rank_pred, rank_test)
# # # recall_pair = np.array(recall_pair)
# # # recall_pair_mask = np.ma.masked_invalid(recall_pair)
# # # g_mean_pair = gmean(recall_pair_mask, axis=None)
# # # print recall_pair
# # # print recall_pair_mask
# # # print Nsamp_pair
# # # print g_mean_pair
# # print perfMeasure(rank_pred, rank_test, rankopt = True)
#
# def crossValidate(x,y, method = "logReg",cv=5, alpha = None):
#     #  error measure
#     results = []
#     if method == "logReg":
#         results = {"perf":[], "coef":[], "interc":[]}
#     elif method == "dT":
#         results = {"alpha": [], "perf":[]}
#
#     # cross validation #
#     np.random.seed(1100)
#     kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
#     for train,test in kf.split(x):
#         x_train = x[train,:]
#         y_train = y[train,:]
#         x_test = x[test,:]
#         y_test = y[test,:]
#
#
#         result = simpleCount(y_train,x_test)
#
#         # performance measure
#         if method == "logReg":
#             y_pred = result
#             results["perf"].append(perfMeasure(y_pred,y_test))
#
#     for key in results.keys():
#         item = np.array(results[key])
#         mean = np.nanmean(item, axis = 0)
#         std = np.nanstd(item, axis = 0)
#         results[key] = [mean, std]
#
#     return results
#
# def simpleCount(y_train,x_test):
#     if type(y_train) != list:
#         y_train.tolist()
#     Nsamp = len(x_test)
#     Nclass = len(y_train[0])
#     scores = [0 for i in range(Nclass)]
#     for y in y_train:
#         for j in range(Nclass):
#             scores[j] += y[j]
#     total = sum(scores)
#     for j in range(Nclass):
#         scores[j]=scores[j]*1.0/total
#     return np.array([scores for s in range(Nsamp)])
#
if __name__ == "__main__":
    x,y= dataClean("data/nytimes_Feature_linkemotion.txt")
    # print "number of samples: ", x.shape[0]
    print y.shape[1]
#
#     ### test ####
#     feature_name = "No feature"
#     X_non =np.ones([y.shape[0],1]).astype("float")
#     result = crossValidate(X_non,y)
#     # result = crossValidate(x,y)
#     # feature_name = "all"
#     print "------%s feature -----" % feature_name
#     print result
#     # # write2result #
#     # file = open("result_washington.txt","a")
#     # file.write("number of samples: %d\n" % x.shape[0])
#     # file.write("------%s feature -----\n" % feature_name)
#     # file.write("NONERECALL: %f\n" % NONERECALL)
#     # file.write("CV: %d\n" % 5)
#     # file.write(str(result)+"\n")
#     # file.close()
