from DecisionTree import *

"""
rank = [[highest_possible_rank, lowest_possible_rank] for emoticon in emoticon_list]
"""

def MM(ranks, max_iter = 10):
    """
    The modified MM algorithm proposed for incomplete ranks
    :param ranks: y in ranks in given nodes
    :return: the deviation theta and the node result (median)
    """
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
    theta = MMMallowDev(ranks, median)
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
                


    return ext_rank


def MMBC(ranks_cplt):
    # using set of complete ranks finding median rank using Borda Count#
    rank_agg = []
    return rank_agg


def MMMallowDev(ranks, median):
    pass


def score2rank(rscore, cplt=False):
    pass