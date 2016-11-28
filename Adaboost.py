import numpy as np
import math
from datetime import datetime
from datetime import timedelta
from sklearn.model_selection import KFold
from DecisionTreeWeight import DecisionTree
from DecisionTreeWeight import label2Rank
from DecisionTreeWeight import rank2Weight
from DecisionTreeWeight import dataSimulated
import logRegFeatureEmotion as LogR

ITER_MAX = 10
stop_criterion_mis_rate = 0.1
output = 1
cost = "init_weight"
def adaboost(x_train, y_train, x_test = None, y_test = None, output = output, iter_max = ITER_MAX, cost = None):

    Nsamp = y_train.shape[0]

    classifiers = []

    # initialize weights #
    if cost is None:
        weight = 1.0/Nsamp
        weights_init = np.array([weight for i in range(Nsamp)], dtype = np.float32)
        weights = weights_init
    elif cost == "init_weight":
        weights_init = rank2Weight(y_train)
        weights = weights_init
    else:
        raise(ValueError, "unsupported cost type")

    start = datetime.now() # timer
    for iter in range(iter_max):
        # base classifier, decisiontree for now #
        tree = DecisionTree().buildtree(x_train, y_train, weights, stop_criterion_mis_rate = stop_criterion_mis_rate)
        # tree.printtree()
        # training result #
        compare_results = [False for i in range(Nsamp)]# whether correctly predicted
        for i in range(Nsamp):
            y_pred = tree.predict(x_train[i])
            # print y_pred, y_train[i]
            cmp_result = not tree.diffLabel(y_pred, y_train[i])
            compare_results[i] = cmp_result
        compare_results = np.array(compare_results, dtype = np.bool)

        # updating weight for classifier, wc#
        weight_sum_cor = np.sum(weights[compare_results == True])
        weight_sum_dis = np.sum(weights[compare_results == False])
        # ### test ###
        # print "iter: ", iter
        # # print "wc: ", wc
        # print "cor: ", weight_sum_cor, " dis: ", weight_sum_dis
        # print compare_results
        # print weights
        # print weights[compare_results == False]
        if weight_sum_cor < weight_sum_dis:                                 # the classifier is too weak for boosting
            raise(ValueError,"too weak classifier")
        if weight_sum_dis == 0:                                             # already perfect classifier
            Warning("perfect classifier")
            break
        wc = 0.5 * (math.log(weight_sum_cor) - math.log(weight_sum_dis))
        print "wc", wc ### test
        # updating weights #
        weights = weightsUpdate(weights, compare_results, wc)

        # add classifier to classifier list #
        classifiers.append([wc, tree])

        # realtime output #
        if output is not None and (iter+1) % output == 0:
            y_pred = predict(x_test,classifiers)
            performance = LogR.perfMeasure(y_pred,y_test,rankopt = True)
            print "iter: ", iter+1
            print performance
            # Nsamp_test = x_test.shape[0]
            # compare_results = [False for i in range(Nsamp_test)]  # whether correctly predicted
            # for i in range(Nsamp_test):
            #     # print y_pred, y_train[i]
            #     cmp_result = not tree.diffLabel(y_pred[i], y_test[i])
            #     compare_results[i] = cmp_result
            # compare_results = np.array(compare_results, dtype=np.bool)
            # print "correctly classified", np.sum(np.ones(Nsamp_test)[compare_results==True])/(1.0*Nsamp_test)
            duration = datetime.now()-start
            start = datetime.now()
            print "time for %d iters: %f" % (output, duration.total_seconds())
    return classifiers


def weightsUpdate(weights, compare_results, wc):
    Nsamp = len(weights)
    new_weights = np.array([weights[i] for i in range(Nsamp)], dtype=np.float32)
    if type(compare_results) ==np.ndarray:
        compare_results = compare_results.tolist()
    cmp_results = map(lambda x: 1 if x else -1, compare_results)
    total_weight = 0.0
    for i in range(Nsamp):
        new_weights[i] *= math.exp(-wc*cmp_results[i])
        total_weight += new_weights
    new_weights = new_weights/total_weight
    return new_weights


def predict(x_test, classifiers):
    if x_test.ndim == 2:
        y_pred = []
        for samp in range(x_test.shape[0]):
            y_pred.append(predict(x_test[samp], classifiers))
        return np.array(y_pred)
    else:
        y_s = []
        w_s = map(lambda x: x[0], classifiers)
        for i in range(len(classifiers)):
            y_s.append(classifiers[i][1].predict(x_test))
        y_s = np.array(y_s)
        w_s = np.array(w_s)
        y_pred = DecisionTree().nodeResult(y_s, w_s) # using the same combine method as that for tree node result
        return y_pred


def crossValidate(x,y, cv=5, nocross = False, cost = None, iter_max = ITER_MAX):

    results = {"perf": []}

    # cross validation #
    np.random.seed(1100)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)  ## for testing fixing random_state
    for train, test in kf.split(x):
        x_train = x[train, :]
        y_train = y[train, :]
        x_test = x[test, :]
        y_test = y[test, :]

        # training and predict

        # if alpha == None:
        #     ## nested select validate and test ##
        #     # print "start searching alpha:", datetime.now() ### test
        #     alpha_sel, perf = DTme.hyperParometer(x_train, y_train)
        #     # print "finish searching alpha:", datetime.now(), alpha ### test
        # else:
        #     alpha_sel = alpha
        classifiers = adaboost(x_train,y_train,x_test,y_test, iter_max= iter_max, cost = cost)

        # performance measure
        y_pred = predict(x_test, classifiers)
        results["perf"].append(LogR.perfMeasure(y_pred, y_test, rankopt=True))

        if nocross:
            break

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


if __name__ == "__main__":
    x,y = LogR.dataClean("data/posts_Feature_Emotion.txt")
    y = label2Rank(y)
    # x,y = dataSimulated(100,3,5)
    # for j in range(1,6):
    #     stop_criterion_mis_rate = 0.22 - 0.04*j
    #     for m in range(10):
    #         ITER_MAX = 10 + m*10
    result = crossValidate(x,y, nocross = False, iter_max=ITER_MAX, cost = cost)
    print result
    with open("result_boost.txt","a") as f:
        f.write("Nsamp: %d\n" % x.shape[0])
        f.write("iter_max "+str(ITER_MAX)+"\n")
        f.write("stop misclassification rate %f\n" %stop_criterion_mis_rate)
        f.write("cost: initial weight v_0\n")
        f.write(str(result)+"\n")
