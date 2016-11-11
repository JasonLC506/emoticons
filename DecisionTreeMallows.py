import DecisionTree as DTme
from datetime import datetime
from datetime import timedelta
from scipy.stats import rankdata
from scipy.optimize import ridder
import random
import math
import numpy as np

"""
rankform = [[highest_possible_rank, lowest_possible_rank] for emoticon in emoticon_list]
"""

def MM(ranks, max_iter = 10):
    """
    The modified MM algorithm proposed for incomplete ranks
    :param ranks: y in ranks in given nodes
    :return: the deviation theta and the node result (median)
    """
    if type(ranks)==np.ndarray:
        ranks = ranks.tolist()
    median = MMInit(ranks)
    flag_cvg = False
    for iter in range(max_iter):
        ranks_cplt = []
        for rank in ranks:
            ranks_cplt.append(MMExt(rank, median))
        median_new = MMBC(ranks_cplt)
        if median_new == median:
            flag_cvg = True
            break
        else:
            median = median_new
    if not flag_cvg:
        print "warning: MM fails to converge"
    theta = MMMallowTheta(ranks_cplt, median)
    return theta, median


def MMInit(ranks):
    # using set of incomplete ranks finding initial median rank for MM algorithm #
    # modified gneralized Borda Count for ranks with abstention in the end #
    # the abstention in the middle is rare and eliminated according label id order #
    Nclass = len(ranks[0])
    init_rank = []
    rscore = [0.0 for i in range(Nclass)]
    for rank in ranks:
        for label in range(Nclass):
            score_label = Nclass - float(rank[label][0] + rank[label][1])/2.0 # key formula #
            rscore[label] += score_label
    init_rank = score2rank(rscore, cplt=True)
    return init_rank


def MMExt(rank, median):
    # Given median rank, find most probable consistent extensions for input incomplete rank #
    # robust for complete rank input #
    Nclass = len(rank)
    ext_rank = [rank[i] for i in range(Nclass)]
    for r in range(Nclass):
        # for each rank position #
        compete = []
        for label in range(Nclass):
            if ext_rank[label][0] == r and ext_rank[label][1] == r:
                break # rank position r is complete
            elif ext_rank[label][0] == r: # abstention
                compete.append(label)
        if len(compete) == 1:
            ext_rank[compete[0]][1] = ext_rank[compete[0]][0] # lowest = highest
        elif len(compete) > 1:
            highest_label = compete[0]
            for label in compete:
                if median[label][0] < median[highest_label][0]: # rank higher than highest label
                    highest_label = label
            for label in compete:
                if label == highest_label:
                    ext_rank[label][1] = ext_rank[label][0]
                else:
                    ext_rank[label][0] += 1
    return ext_rank


def MMBC(ranks_cplt):
    # using set of complete ranks finding median rank using Borda Count#
    # using the easiest non-weighted Borda Count #
    return MMInit(ranks_cplt)


def MMMallowTheta(ranks_cplt, median):
    """
    :return: the MLE theta parameter in Mallows distribution
    """
    try:
        theta = ridder(MallowsThetaDev, 1e-5, 1e+5, args=(ranks_cplt, median))
        return theta
    except ValueError, e:
        print "!!!Not well chosen median"
        raise e



def MallowsThetaDev(theta, ranks_cplt, median):
    Nsamp = len(ranks_cplt)
    Nclass = len(ranks_cplt[0])
    distances = [discordant(ranks_cplt[i], median) for i in range(Nsamp)]
    distances = np.array(distances, dtype=np.float16)
    dev = np.mean(distances)
    thetadev = Nclass*math.exp(-theta)/(1-math.exp(-theta))-sum([j*math.exp(-j*theta)/(1-math.exp(-j*theta)) for j in range(1,Nclass+1)])

    return (thetadev - dev)



def discordant(rank, rank_ref):
    # the number of discordant pairs #
    dis = 0
    Nclass = len(rank)
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            if (rank[i][0] - rank[j][0])*(rank_ref[i][0]-rank_ref[j][0]) < 0:
                dis += 1
    return dis


def score2rank(rscore, cplt=False):
    """
    :param rscore: the vector of score with each dimension for each label
    :param cplt: True for only complete rank output, False for general output
    :return: rank in rankform
    """
    Nclass = len(rscore)
    rscore_minus = map(lambda x: -x, rscore)
    rank_min = rankdata(rscore_minus, method = "min")-1
    rank_max = rankdata(rscore_minus, method = "max")-1
    rank = [[rank_min[i],rank_max[i]] for i in range(Nclass)]
    if cplt:
        random_prior = [i for i in range(Nclass)]
        random.shuffle(random_prior)
        position_taken = []
        for label in random_prior:
            for position in range(rank[label][0], rank[label][1]+1):
                if position not in position_taken:
                    position_taken.append(position)
                    rank[label][0] = position
                    rank[label][1] = position
                    break
    return rank


def bestSplit(x, y, samples, feature, min_node=1):
    """

    :param x: features
    :param y: np.ndarray of ranks in rankform
    :param samples: np.array
    :param feature: current feature considered
    :param min_node: minimum number of samples in a node
    :return:
    """
    min_var = -1
    best_split = 0
    best_sets = []
    best_sets_result = []

    Nsamp = len(samples)

    temp = [(x[samples[i],feature],samples[i]) for i in range(Nsamp)]
    dtype = [("value", float),("index", int)]
    x_ord = np.sort(np.array(temp,dtype=dtype), order = "value")

    old_value = x_ord[0][0]
    left_size =1
    right_size = Nsamp-1
    left_samps = np.array([x_ord[0][1]])
    right_samps = np.array([x_ord[s][1] for s in range(1, Nsamp)])
    for i in range(1,Nsamp-1): # avoid 0 split
        value = x_ord[i][0]
        if value != old_value:
            # a valid split #
            left_result = MM(y[left_samps])
            right_result = MM(y[right_samps])
            variance = (left_size*left_result[0]+right_size*right_result[0])/(1.0*Nsamp)
            if min_var < 0 or min_var > variance:
                min_var = variance
                best_sets = [left_samps, right_samps]
                best_split = [feature, value] # >= split
                best_sets_result = [left_result, right_result]

        np.append(left_samps, x_ord[i][1])
        right_samps = np.delete(right_samps, 0)
        left_size += 1
        right_size += -1
        old_value = value
    return min_var, best_split, best_sets, best_sets_result


def buildtree(x,y, samples, min_node=1, result_cur = None):
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples) != np.ndarray:
        samples = np.array(samples)
    if len(samples) == 0:
        return DTme.decisionnode()

    if result_cur is None:
        result_cur = MM(y[samples])

    if len(samples)<= min_node:
        return DTme.decisionnode(result=result_cur[1])
    # find best split
    best_gain = 0.0
    best_split = []
    best_sets = []
    best_sets_result = []

    N_feature = x.shape[1]

    for feature in range(N_feature):
        # nlogn selection
        min_var, split, sets, sets_result = bestSplit(x,y,samples,feature)
        gain = result_cur[0] - min_var
        if gain > best_gain and len(sets[0]) * len(sets[1]) > 0:
            best_gain = gain
            best_split = split
            best_sets = sets
            best_sets_result = sets_result

    if best_gain>0:
        tb = buildtree(x,y, best_sets[0], min_node = min_node, result_cur = best_sets_result[0])
        fb = buildtree(x,y, best_sets[1], min_node = min_node, result_cur = best_sets_result[1])
        return DTme.decisionnode(feature = best_split[0], value = best_split[1], result = result_cur[1],
                            tb = tb, fb = fb,
                            gain = (tb.gain+fb.gain+best_gain), size_subtree = (tb.size+fb.size))
    else:
        return DTme.decisionnode(result = result_cur[1])





