"""
General KNN class, with default KNN_Plackett-Luce model, referring to PlackettLuce.py
"""

import numpy as np
from biheap import BiHeap
from PlackettLuce import PlackettLuce
from sklearn.model_selection import KFold
import logRegFeatureEmotion as LogR
from datetime import datetime
from datetime import timedelta
from readSushiData import readSushiData

class KNN(object):
    def __init__(self, K):
        self.K = K
        self.x = None
        self.y = None

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train
        return self

    def predict(self, x_test):
        ## batch and single predict, default feature dimension is 1 ##
        if x_test.ndim > 1:
            y_pred = []
            for samp in range(x_test.shape[0]):
                y_pred.append(self.predict(x_test[samp]))
            return np.array(y_pred)

        neighbors = self.neighborhood(x_test)
        y_pred = self.aggregate(neighbors)
        return y_pred

    def probcal(self, x_test, y_test):
        ## only single sample calculation ##
        neighbors = self.neighborhood(x_test)
        prob = PlackettLuce().fit(self.y[neighbors,:]).probability(y_test)
        return prob

    def neighborhood(self, x_test):
        """
        return neighborhood index of x_test, given self.x and self.K
        """
        neighbors = []
        queue = BiHeap().buildheap([],minimal=False,key=1, identifier=0)
        for samp in range(self.x.shape[0]):
            dist = self.distance(x_test, self.x[samp])
            queue.insert([samp, dist])
            if queue.length > self.K:
                _ = queue.pop()
        for i in range(self.K):
            samp, dist = queue.pop()
            neighbors.append(samp)
        return np.array(neighbors)

    def aggregate(self, neighbors):
        """
        calculate predicted result
        :param neighbors: neighbor points index
        :return: aggregated result of given neighborhood
        default [1]
        """
        return PlackettLuce().fit(self.y[neighbors,:]).predict()

    def distance(self, x1, x2):
        """
        calculate distance between two points in feature space
        default: Cartesian distance square
        """
        return np.sum(np.power((x1-x2),2.0))


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

        y_pred = KNN(K=K).fit(x_train, y_train).predict(x_test)
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
    datafile = "data/sushi data"
    K = 20

    # x,y = LogR.dataClean(datafile)
    # y = np.array(map(LogR.rankOrder, y.tolist()))
    x, y = readSushiData()
    # x,y = x[:1000, :], y[:1000, :]

    start = datetime.now()
    result = crossValidate(x,y,K=K)
    duration = datetime.now() - start

    print duration.total_seconds()
    print result

    with open("results/result_KNNPL.txt", "a") as f:
        f.write("K = %d\n" % K)
        f.write("data = %s\n" % datafile)
        f.write("time = %f\n" % duration.total_seconds())
        f.write(str(result)+"\n")