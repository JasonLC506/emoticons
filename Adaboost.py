import numpy as np
import math
import DecisionTree as DTme

def adaboost(x_train, y_train, iter_max = 100):
    Nsamp = y_train.shape[0]

    classifiers = []

    # initialize weights #
    weight = 1.0/Nsamp
    weights = np.array([weight for i in range(Nsamp)], dtype = np.float32)

    for iter in range(iter_max):
        # base classifier, decisiontree for now #
        tree = buildtree(x_train, y_train, weights)

        # training result #
        compare_results = [False for i in range(Nsamp)]
        for i in range(Nsamp):
            y_pred = predict(x_train[i], tree)
            cmp_result = resultCompare(y_pred, y_train[i])
            compare_results[i] = cmp_result

        # updating weight for classifier, wc#
        weight_sum_cor = np.sum(weights[compare_results == True])
        weight_sum_dis = np.sum(weights[compare_results == False])
        if weight_sum_cor < weight_sum_dis:                                 # the classifier is too weak for boosting
            raise(ValueError,"too weak classifier")
        if weight_sum_dis == 0:                                             # already perfect classifier
            Warning("perfect classifier")
            break
        wc = 0.5 * (math.log(weight_sum_cor) - math.log(weight_sum_dis))

        # updating weights #
        weights = weightsUpdate(weights, compare_results, wc)

        # add classifier to classifier list #
        classifiers.append([wc, tree])

    return classifiers


def weightsUpdate(weights, compare_results, wc):
    Nsamp = len(weights)
    new_weights = np.array([weights[i] for i in range(Nsamp)], dtype=np.float32)
    cmp_results = map(lambda x: 1 if x else -1, compare_results)
    total_weight = 0.0
    for i in range(Nsamp):
        new_weights[i] *= math.exp(-wc*cmp_results[i])
        total_weight += new_weights
    new_weights = new_weights/total_weight
    return new_weights


def buildtree(x_train, y_train, weights):
    """
    build weighted tree
    besides weights for samples, the stop condition should be modified
    :param x_train:
    :param y_train:
    :param weights:
    :return:
    """
    pass


def predict(obs, tree):
    pass


def resultCompare(y_pred, y_train):
    """
    Compare predicted and true labels
    :param y_pred: single result
    :param y_train: single result
    :return: True or False for now (dependent on boosting algorithm)
    """
    pass