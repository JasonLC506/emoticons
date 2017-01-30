import logRegFeatureEmotion as LogR
import numpy as np
from queryAlchemy import emotion_list
from logRegFeatureEmotion import emoticon_list
from sklearn.model_selection import KFold

def score2pair(x,y, k = 2, Abstention = False):
    """
    k is the enhance factor for preference pairs over no-prefer pairs
    the bigger k is, the smaller the effect of no-prefer pairs,
    kind of inverse of laplace smooth

    y is the score data
    """
    # consider not appearing emoticons as ranked lowest #
    Nsamp = y.shape[0]
    Nclass = y.shape[1]
    Nfeature = x.shape[1]
    # no-prefer pairs are included #
    y_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
    x_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
    # for case no-prefer pairs are excluded#
    x_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
    y_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
    # i,j value = 1 if emoticon i ranks higher than j, 0 otherwise #
    for samp in range(Nsamp):
        for i in range(Nclass):
            for j in range(i+1, Nclass):
                if y[samp,i] > y[samp,j]:
                    for _ in range(k):
                        y_enlarge[i][j].append(1)
                        x_enlarge[i][j].append(x[samp])
                    y_list[i][j].append(1)
                    x_list[i][j].append(x[samp])
                elif y[samp,i] < y[samp, j]:
                    for _ in range(k):
                        y_enlarge[i][j].append(0)
                        x_enlarge[i][j].append(x[samp])
                    y_list[i][j].append(0)
                    x_list[i][j].append(x[samp])
                else: # y[samp,i] == y[samp,j]
                    for value in [0,1]:
                        y_enlarge[i][j].append(value)
                        x_enlarge[i][j].append(x[samp])

    if Abstention:
        # include no-preference pair #
        return x_enlarge, y_enlarge
    else:
        # exclude no-preference pair #
        return x_list, y_list


def pairpref(x_train,y_train, x_test):
    """
    x_train, y_train in the form of output of score2pair
    x_test should be ordinary Nsamp*Nfeature
    """
    Nclass = len(y_train)
    Nsamp_test = len(x_test)
    pair_pref_prob = np.zeros(shape=[Nsamp_test, Nclass, Nclass])
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            ## handle extreme cases ##
            if len(y_train[i][j]) == 0:
                pair_pref_prob[:,i,j] = 0.5*np.ones(Nsamp_test, dtype="float")
                continue
            if 0 not in y_train[i][j]:
                pair_pref_prob[:,i,j] = np.ones(Nsamp_test, dtype="float")
                continue
            if 1 not in y_train[i][j]:
                pair_pref_prob[:,i,j] = np.zeros(Nsamp_test, dtype="float")
                continue
            prob, coef, intercept = LogR.logRegFeatureEmotion(x_train[i][j], y_train[i][j], x_test)
            pair_pref_prob[:, i, j] = prob[:,1]
    # fill pair_pref_prob matrix #
    for i in range(Nclass):
        for j in range(0, i):
            pair_pref_prob[:,i,j] = 1 - pair_pref_prob[:,j,i]
    return pair_pref_prob


def pair2rank(pair_pref_prob_samp):
    # list(Nclass * Nclass)#
    # borda count #
    ppps = pair_pref_prob_samp
    Nclass = len(ppps)
    for i in range(Nclass):
        if ppps[i][i]!=0:
            print "warning: self comparison", i, ppps[i][i]
            ppps[i][i] = 0
    rscore = map(sum, ppps)
    # print "rscore", rscore
    rank = LogR.rankOrder(rscore)
    return rank


def rankPairPref(x_train, y_train, x_test):
    """
    x_train, y_train in the form of output of score2pair
    x_test should be ordinary Nsamp*Nfeature
    """
    pair_pref_prob = pairpref(x_train, y_train, x_test)
    ppp = pair_pref_prob.tolist()
    ranks = map(pair2rank, ppp)
    return ranks


def crossValidate(x,y, cv = 5, Abstention = True, Inverse_laplace = 4):
    results = {"perf":[]}
    np.random.seed(2016)
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0)
    for train, test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        # score input for y to pairwise input #
        x_tr, y_tr = score2pair(x_train,y_train, k=Inverse_laplace, Abstention=Abstention)
        # train and predict ranks for test data #
        ranks = rankPairPref(x_tr, y_tr, x_test)
        # transform test score data to rank
        y_te = map(LogR.rankOrder, y_test.tolist())

        results["perf"].append(LogR.perfMeasure(y_pred=ranks,y_test=y_te,rankopt=True))

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


def simulatedtest():
    x = np.array([[1,2,3],[3,2,1]])
    y = np.array([[10,8,6,5,0,0],[7,9,0,0,0,0]])
    x_enlarge, y_enlarge = score2pair(x,y,k=4,Abstention = True)
    x_list, y_list = score2pair(x,y,k=4, Abstention=False)
    print "abstention:"
    for i in range(y.shape[1]):
        for j in range(i+1, y.shape[1]):
            print i,j,":"
            print x_enlarge[i][j]
            print y_enlarge[i][j]
    print "not abstention"
    for i in range(y.shape[1]):
        for j in range(i+1, y.shape[1]):
            print i,j,":"
            print x_list[i][j]
            print y_list[i][j]
    x_test = x
    ranks = rankPairPref(x_enlarge, y_enlarge, x_test)
    print ranks


if __name__ == "__main__":
    x,y = LogR.dataClean("data/foxnews_Feature_linkemotion.txt")
    print "number of samples: ", x.shape[0]
    Absention = False
    Inverse_laplace = 64
    result = crossValidate(x, y, cv=5, Abstention=Absention, Inverse_laplace=Inverse_laplace)
    print result
    # write2result #
    file = open("result_rpp_foxnews.txt","a")
    file.write("number of samples: %d\n" % x.shape[0])
    file.write("NONERECALL: %f\n" % LogR.NONERECALL)
    file.write("CV: %d\n" % 5)
    file.write("Abstention %s\n" % str(Absention))
    file.write("Inverse_laplace %d\n" % Inverse_laplace)
    file.write(str(result)+"\n")
    file.close()
