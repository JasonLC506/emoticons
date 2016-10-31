import numpy as np
import logRegFeatureEmotion as LogR
import cPickle
from matplotlib import pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
from queryAlchemy import emotion_list as EL
import ast
import math
import itertools
from DecisionTree import *
from datetime import datetime
from scipy.stats.mstats import gmean

def dataSimulated(Nsamp, Nfeature, Nclass, Noise):

    np.random.seed(seed=10)
    x = np.random.random(Nsamp*Nfeature)
    x = x.reshape([Nsamp,Nfeature])
    y = (x + Noise * np.random.random(Nsamp*Nclass).reshape([Nsamp,Nclass]))/(1+Noise)*Nappear
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] *= weight[j]
    y = y.astype(int)
    y = np.array(y)
    return x,y


def crossValidateSimple(x,y, method = "logReg",cv=5, alpha = None):
    #  error measure
    results = {"perf":[]}
    # cross validation #
    np.random.seed(1100)
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
    for train,test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        # performance measure
        if method == "logReg":
            y_pred = np.zeros(x_test.shape)
            for i in range(y_pred.shape[0]):
                for j in range(y_pred.shape[1]):
                    y_pred[i, j] = x_test[i,j] * weight[j]
            results["perf"].append(LogR.perfMeasure(y_pred,y_test))

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":
    Imbalance = 1.0
    weight = np.exp(-np.arange(6,dtype="float")*Imbalance)
    weight = weight/np.sum(weight)

    Nappear = 500

    noise_level = np.linspace(0.1,10.0, num=50).tolist()
    perf = []
    for noise in noise_level:
        x,y = dataSimulated(10000, 6, 6, noise)
        result = LogR.crossValidate(x,y)
        performance = result["perf"][0] # only mean of the performance results
        perf.append(performance)
    perf = np.array(perf)
    file = open("ImbalancedTest.txt","w")
    cPickle.dump(perf,file)
    file.close()
    ref_var = [0,23]
    tar_var = [22]
    for var1 in ref_var:
        for var2 in tar_var:
            fig = plt.figure()
            plt.plot(perf[:,var1],perf[:,var2])
            fig.savefig("ImbalancedTest_%d_%d.png" % (var1, var2))