from AnomalyDataPrep import anomalyDataPrep
from logRegFeatureEmotion import dataClean
from DecisionTreeWeight import label2Rank
# from KNNPlackettLuce import KNN
from KNNSMP import KNNSMP as KNN
import numpy as np
import sys

K_KNN = 50

def traintest(x_train, y_train, x_test, y_test, anomaly_threshold = "median"):
    """
    input format is the same as the output of anomalyDataPrep function
    """
    model = KNN(K = K_KNN)

    ## fit ##
    model.fit(x_train, y_train)

    ## calculate anomaly score for both non-perturbed and perturbed set ##
    anomaly_scores = []
    for p in range(len(y_test)):
        y_test_part = y_test[p]
        x_test_part = x_test[p]
        Nsamp_test = y_test_part.shape[0]
        anomaly_score_part = np.zeros(Nsamp_test, dtype=np.float64)
        for isamp in range(Nsamp_test):
            prob = model.probcal(x_test_part[isamp],y_test_part[isamp])
            anomaly_score_part[isamp] = 1.0 - prob ## using 1-probability as anomaly score
        anomaly_scores.append(anomaly_score_part)

    ## mark anomaly ##
    anomaly = []
    if anomaly_threshold == "median":
        anomaly_score_whole = np.concatenate((anomaly_scores[0],anomaly_scores[1]))
        threshold = np.median(anomaly_score_whole)
    else:
        raise ValueError("only supporting median threshold currently")
    for p in range(len(anomaly_scores)):
        anomaly_score_part = anomaly_scores[p]
        Nsamp_part = anomaly_score_part.shape[0]
        anomaly_part = np.zeros(Nsamp_part, dtype=np.float64)
        for isamp in range(Nsamp_part):
            if anomaly_score_part[isamp] >= threshold:
                anomaly_part[isamp] = 1
        anomaly.append(anomaly_part)

    ## performance computation ##
    perf = performance(anomaly)

    return perf


def performance(anomaly):
    recalls = []
    for p in range(len(anomaly)):
        if p == 0:
            # non-perturbed data #
            recall_part = 1 - np.mean(anomaly[p])
        elif p == 1:
            # perturbed data #
            recall_part = np.mean(anomaly[p])
        else:
            raise ValueError("so many parts of test data?")
        recalls.append(recall_part)
    return recalls


def multitest(x, y, Ntest = 10):
    results = {"perf":[]}
    for itest in range(Ntest):
        x_train, y_train, x_test, y_test = anomalyDataPrep(x,y)
        results["perf"].append(traintest(x_train, y_train, x_test, y_test))

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":
    K_KNN = int(sys.argv[2])
    news = sys.argv[1]
    # K_KNN = 10
    # news = "nytimes"
    result_file = "results/anomaly_KNN_SMP.txt"
    x, y = dataClean("data/" + news + "_Feature_linkemotion.txt")
    y = label2Rank(y)
    results = multitest(x, y)
    print results
    with open(result_file, "a") as f:
        f.write("news: %s\n" % news)
        f.write("K_KNN: %d\n" % K_KNN)
        f.write(str(results)+"\n")
