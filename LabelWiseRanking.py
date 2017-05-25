from sklearn import linear_model
from sklearn.model_selection import KFold
import logRegFeatureEmotion as LogR
from DecisionTree import label2Rank
import numpy as np
from munkres import Munkres
from readSyntheticData import readSyntheticData
import sys

def rank2position(y_train):
    """
    :param y_train: np.array(np.int32) n*d with ranking vector each position as ranking position
    :return: y_train_pos, valid_samp, miss
    """
    y_train.tolist()
    Nsamp = len(y_train)
    Nclass = len(y_train[0])
    y_train_pos = [[-1 for label in range(Nclass)] for samp in range(Nsamp)]
    valid_samp = [[False for label in range(Nclass)] for samp in range(Nsamp)]
    for samp in range(Nsamp):
        ranking = y_train[samp]
        for pos in range(Nclass):
            if ranking[pos] < 0:
                break
            else:
                label = ranking[pos]
                y_train_pos[samp][label] = pos
                valid_samp[samp][label] = True
    y_train_pos = np.asarray(y_train_pos, dtype=np.int32)
    valid_samp = np.asarray(valid_samp, dtype=bool)

    miss = [[True for p in range(Nclass)] for label in range(Nclass)]
    for samp in range(Nsamp):
        for label in range(Nclass):
            if valid_samp[samp,label]:
                miss[label][y_train_pos[samp,label]] = False

    return y_train_pos, valid_samp, miss


def complete(posprob, miss):
    Nsamp = posprob.shape[0]
    a = posprob.tolist()
    # print type(a), a
    # print "miss", miss
    Nclass = len(miss)
    for i in range(Nclass):
        if miss[i]:
            if i < len(a[0]):
                for samp in range(Nsamp):
                    a[samp].insert(i,0.0)
            else:
                for samp in range(Nsamp):
                    a[samp].append(0.0)
    # print a
    return np.array(a)


def posprobloss(labelposprob):
    """
    here spearman footrule loss for ranking is used
    :param labelposprob: np.ndarray
    :return:
    """
    # print "probmatrix", labelposprob ### test
    Nclass = labelposprob.shape[0]
    labelposloss = [[-1 for pos in range(Nclass)] for label in range(Nclass)]
    for label in range(Nclass):
        for pos in range(Nclass):
            loss = 0.0
            for prob_pos in range(Nclass):
                loss += (abs(prob_pos - pos)*labelposprob[label,prob_pos])
            labelposloss[label][pos] = loss
    # print "lossmatrix",labelposloss ### test
    return np.array(labelposloss, dtype=np.float32)


def aggregate(labelposloss):
    ## using Hungarian algorithm ##
    # print labelposloss ### test
    matrix = labelposloss.tolist()
    Nclass = len(matrix)
    m = Munkres()
    indexs = m.compute(matrix)

    ## transform to rank ##
    ranking = [-1 for i in range(Nclass)]
    for label, pos in indexs:
        ranking[pos] = label
    # print ranking ### test
    return ranking


def labelWiseRanking(x_train, y_train, x_test):
    """
    label ranking
    :param x_train:
    :param y_train: ranking data
    :param x_test:
    :return: ranking data
    """
    y_train_pos, valid_samp, miss = rank2position(y_train)
    # print "miss", miss
    # print "y_train_pos", y_train_pos.shape
    # print "valid_samp", valid_samp.shape
    Nsamp_test = x_test.shape[0]
    Nclass = y_train_pos.shape[1]
    posproblist_labels = [[] for label in range(Nclass)]

    ## pos prob predict for each label ##
    for label in range(Nclass):
        posprob,_a,_b = LogR.logRegFeatureEmotion(x_train[valid_samp[:,label],:], y_train_pos[valid_samp[:,label], label], x_test)
        # print type(posprob), posprob
        posprob_complete = complete(posprob, miss[label])
        # print "completed", type(posprob_complete), posprob_complete
        posproblist_labels[label] = list(posprob_complete)
    # print "posproblist_labels", posproblist_labels
    posproblist_labels = np.array(posproblist_labels, dtype=np.float32)
    # print "posproblist_labels", type(posproblist_labels), posproblist_labels

    ## aggregation ##
    rankings = [[] for samp in range(Nsamp_test)]
    for samp in range(Nsamp_test):
        labelposprob = posproblist_labels[:,samp,:]
        # print "posproblabels", type(posproblabels), posproblabels, posproblabels.shape
        labelposloss = posprobloss(labelposprob)
        rankings[samp] = aggregate(labelposloss)
    # print rankings### test
    return rankings


def simulatedtest():
    x = np.arange(1000).reshape([100,10])
    y = [[2,1,-1] for i in range(80)]
    for _ in range(10):
        y.append([1,2,0])
        y.append([0,2,1])
    y = np.array(y, dtype=np.int32)
    print labelWiseRanking(x,y,x)


def crossValidate(x,y,cv=5):
    results = {"perf":[]}
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train, test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        y_pred = labelWiseRanking(x_train, y_train, x_test)
        results["perf"].append(LogR.perfMeasure(y_pred, y_test, rankopt=True))
    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        results[key] = [mean, std]
    return results


if __name__ == "__main__":
    # x,y = LogR.dataClean("data/nytimes_Feature_linkemotion.txt")
    # y = label2Rank(y)
    # dataset = "bodyfat"
    dataset = sys.argv[1]
    x, y = readSyntheticData("data/synthetic/" + dataset)
    result = crossValidate(x,y)

    file = open("results/result_LWR_synthetic.txt","a")
    file.write("dataset: synthetic %s\n" % dataset)
    file.write("number of samples: %d\n" % x.shape[0])
    file.write("NONERECALL: %f\n" % LogR.NONERECALL)
    file.write("CV: %d\n" % 5)
    file.write(str(result)+"\n")
    file.close()
    print result