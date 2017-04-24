"""
synthetic anomaly data based on Song, X., Wu, M., Jermaine, C. and Ranka, S., 2007. Conditional anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 19(5). [1]
"""
from logRegFeatureEmotion import dataClean
import numpy as np
from DecisionTreeWeight import KendalltauSingle
from DecisionTreeWeight import label2Rank

TEST_PROP = 0.2
PERTURB_PROP = 0.5
PERTURB_POOL_SIZE = 50
PERTURB_POOL_PROP = 0.25

def anomalyDataPrep(x, y, test_prop = TEST_PROP, perturb_prop = PERTURB_PROP):
    x_train, y_train, x_test, y_test = datadivide(x, y, test_prop)
    x_np, y_np, x_p, y_p = perturb(x_test, y_test, perturb_prop)
    return x_train, y_train, [x_np, x_p], [y_np, y_p]

def perturb(x, y, perturb_prop = 0.5, perturb_pool_size = PERTURB_POOL_SIZE, perturb_pool_prop = PERTURB_POOL_PROP):
    x_np, y_np, x_tp, y_tp = datadivide(x, y, perturb_prop)
    # print "pre-perturb ", y_tp ### test
    ## perturb x_p, y_p ##
    Nsamp = x_tp.shape[0]
    x_p = x_tp # keep contextual attributes
    y_p = []
    perturb_pool = min([perturb_pool_size, int(perturb_pool_prop * Nsamp)])
    for isamp in range(Nsamp):
        origin = y_tp[isamp]
        farest = None
        dist_farest = None
        _a, _b, _c, pool = datadivide(x_tp, y_tp, None, test_size = perturb_pool)
        for icandidate in range(pool.shape[0]):
            candidate = pool[icandidate]
            dist = distance(origin, candidate)
            if dist_farest is None or dist_farest < dist:
                dist_farest = dist
                farest = candidate
        if farest is None:
            raise ValueError("cannot find farest candidate")
        else:
            y_p.append(farest)
    y_p = np.array(y_p)
    # print "post-perturb ", y_p ### test
    return x_np, y_np, x_p, y_p

def datadivide(x, y, test_prop, test_size=None):
    Nsamp = x.shape[0]
    samples = np.arange(Nsamp)
    np.random.shuffle(samples) # randomise
    if test_size is None:
        Nsamp_train = Nsamp - int(Nsamp * test_prop)
    else:
        Nsamp_train = Nsamp - test_size
    Nsamp_test = Nsamp - Nsamp_train

    x_train = x[samples[:Nsamp_train]]
    y_train = y[samples[:Nsamp_train]]
    x_test = x[samples[Nsamp_train:]]
    y_test = y[samples[Nsamp_train:]]

    return x_train, y_train, x_test, y_test


def distance(rank_true, rank_candidate):
    ## using - kendall's tau as distance measure, the larger the farther ##
    return - KendalltauSingle(rank_true, rank_candidate)


if __name__ == "__main__":
    x, y = dataClean("data/nytimes_Feature_linkemotion.txt")
    y = label2Rank(y)
    x_train, y_train, x_test, y_test = anomalyDataPrep(x[:40,:], y[:40,:])
    print "-------- training data --------"
    print x_train
    print y_train
    print "-------- non-perturbed data ---------"
    print x_test[0]
    print y_test[0]
    print "-------- perturbed data -----------"
    print x_test[1]
    print y_test[1]
