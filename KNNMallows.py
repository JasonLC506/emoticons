"""
KNN Mallows model
implementing Cheng, W., Huhn, J. and Hullermeier, E., 2009, June. Decision tree and instance-based learning for label ranking [1]
"""

from KNNPlackettLuce import KNN
from DecisionTreeMallows import MM
from DecisionTreeMallows import rankO2New
from DecisionTreeMallows import rankN2Old
import numpy as np
from sklearn.model_selection import KFold
import logRegFeatureEmotion as LogR
from datetime import datetime
from datetime import timedelta
from readSushiData import readSushiData

class KNNMallows(KNN):
    def aggregate(self, neighbors):
        return Mallows().fit(self.y[neighbors,:]).predict()


class Mallows(object):
    """
    input are rankings in old form
    """
    def __init__(self):
        self.Nclass = 0
        self.theta = 0.0
        self.median = None

    def fit(self, y_s, max_iteration = 100, iter_out = True):
        # data structure and transformation #
        self.Nclass = y_s.shape[1]
        ranks = y_s.tolist()
        ranks = map(rankO2New, ranks)
        self.theta, self.median, iter = MM(ranks, max_iter = max_iteration, iter_out = iter_out)
        print "converge in ", iter
        return self

    def predict(self):
        return np.array(rankN2Old(self.median))


def crossValidate(x, y, cv=5, K=None):
    """
    :param y: N*L ranking vectors
    :return:
    """
    results = {"perf": []}

    ## cross validation ##
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(x):
        x_train = x[train, :]
        y_train = y[train, :]
        x_test = x[test, :]
        y_test = y[test, :]

        y_pred = KNNMallows(K=K).fit(x_train, y_train).predict(x_test)
        # print y_pred ### test
        results["perf"].append(LogR.perfMeasure(y_pred, y_test, rankopt=True))
        # print results["perf"][-1]

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":
    newsfiles = ["nytimes","wsj","washington"]
    for news in newsfiles:
        datafile = "data/" + news + "_Feature_linkemotion.txt"
        K = 20

        x,y = LogR.dataClean(datafile)
        y = np.array(map(LogR.rankOrder, y.tolist()))
        # x, y = readSushiData()

        start = datetime.now()
        result = crossValidate(x,y,K=K)
        duration = datetime.now() - start

        print duration.total_seconds()
        print result
    
        with open("results/result_KNNMallows.txt", "a") as f:
            f.write("K = %d\n" % K)
            f.write("data = %s\n" % datafile)
            f.write("time = %f\n" % duration.total_seconds())
            f.write(str(result)+"\n")