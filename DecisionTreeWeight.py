"""
Using gini_rank as split and also pruning
pruning hyperparameter validation using G-mean
"""
import numpy as np
import logRegFeatureEmotion as LogR
from sklearn.model_selection import KFold
from functools import partial
from scipy.stats.mstats import gmean
from datetime import datetime
from stats import pairwise

from logRegFeatureEmotion import NONERECALL


class DecisionTree(object):
    """
    binary tree
    """
    def __init__(self, feature=-1, value=None, result = None, tb=None, fb=None, gain=0.0, mis_rate = None, size_subtree = 1):
        self.feature = feature
        self.value = value
        self.result = result
        self.tb = tb
        self.fb = fb
        # pruning #
        self.mis_rate = mis_rate
        self.gain = gain
        self.size = size_subtree
        self.alpha = -1.0

    # independent functions #
    def buildtree(self, x_train, y_train, weights = None, samples = None, stop_criterion_mis_rate = None,
                  stop_criterion_min_node = 1, stop_criterion_gain = 0.0):
        if samples is None:
            samples = np.arange(y_train.shape[0])
        if weights is None: # not weighted tree
            weights = np.ones(y_train.shape[0], dtype=np.float32)
        Nsamp = len(samples)
        if Nsamp == 0:
            raise (ValueError, "tree node with no samples")

        ## to be writen to current node ##
        self.result = self.nodeResult(y_train[samples], weights[samples])
        self.mis_rate = self.misRate(y_train[samples], weights[samples], self.result)

        ## check if stop ##
        if stop_criterion_mis_rate is not None and self.mis_rate < stop_criterion_mis_rate or \
            stop_criterion_min_node is not None and Nsamp <= stop_criterion_min_node:
            # print "mis_rate: ", self.mis_rate
            return self

        ## find split ##
        # split_criterion for current node #
        split_criterion = self.splitCriterion(y_train[samples], weights[samples])
        # find best split among all binary splits over any feature #
        best_cri, best_split, best_sets = self.bestSplit(x_train, y_train, weights, samples)
        # calculate the gain of split in terms of split_criterion #
        gain = self.splitGain(split_criterion, best_cri)

        ## split or stop ##
        if gain > stop_criterion_gain: # split
            children = [DecisionTree() for c in range(2)]
            for c in range(2):
                children[c].buildtree(samples = best_sets[c],
                                          x_train = x_train, y_train = y_train, weights = weights,
                                          stop_criterion_mis_rate = stop_criterion_mis_rate,
                                          stop_criterion_min_node = stop_criterion_min_node,
                                          stop_criterion_gain = stop_criterion_gain)
            self.tb = children[0]
            self.fb = children[1]
            self.feature = best_split[0]
            self.value = best_split[1]
            self.size = self.tb.size + self.fb.size
            self.gain = self.pruneGain(gain, len(best_sets[0]), len(best_sets[1]))
            self.size = self.tb.size + self.fb.size
            self.alpha = self.gain / (self.size - 1)
            return self
        else:
            return self

    def printtree(self, indent=""):
        if self.tb is None:
            print(indent + str(self.result))
        else:
            print(indent + str(self.feature) + ">=" + str(self.value) + "?")
            print(indent + "T->\n")
            self.tb.printtree(indent + "   ")
            print(indent + "F->\n")
            self.fb.printtree(indent + "   ")

    def predict(self, x_test, alpha = 0):
        if x_test.ndim == 2: # batch predict
            y_pred = []
            for samp in range(x_test.shape[0]):
                y_pred.append(self.predict(x_test[samp], alpha))
            return np.array(y_pred)
        else: # individual predict
            if self.tb is None:
                return self.result
            if self.alpha >= 0:
                if self.alpha < alpha:
                    return self.result
            value = x_test[self.feature]
            if value >= self.value:
                return self.tb.predict(x_test,alpha)
            else:
                return self.fb.predict(x_test,alpha)

    # dependent functions #
    def nodeResult(self, y_s, w_s):
        """
        calculate the predict result for the node
        :param y_s: labels of samples in the tree node
        :param w_s: weights of corresponding samples
        :return: type(y)
        """
        n_rc = self.nRankClass(y_s, w_s, np.arange(y_s.shape[0]))
        Ranks = y_s.shape[1]
        result = []
        for rank in range(Ranks):
            # assign ranking #
            # current rank with highest priority, when zero appearance, from highest rank to lowest #
            priority = [i for i in range(Ranks) if i != rank]
            priority.insert(0, rank)
            flag = False
            for i in priority:
                n_class_cur = n_rc[i]
                while not flag:
                    max_value = max(n_class_cur)
                    if max_value == 0:
                        break  # all are 0 in current rank
                    emoti = n_class_cur.index(max_value)
                    if emoti not in result:
                        result.append(emoti)
                        flag = True  # find best emoticon
                    else:
                        n_class_cur[emoti] = 0
                if flag:
                    break
            if not flag:
                result.append(-1)
        return np.array(result)

    def misRate(self, y_s, w_s, result):
        """
        calculate the misclassification rate in the node
        :param y_s: labels of samples in the tree node
        :param w_s: the weights of corresponding samples
        :param result: predicted result for the node
        :return: float
        """
        Norm = np.sum(w_s)
        mis = 0.0
        for i in range(y_s.shape[0]):
            if self.diffLabel(result, y_s[i]):
                mis += w_s[i]
        return (mis/Norm)

    def diffLabel(self, y_pred, y):
        """
        check if two labels are different
        current for two ranks, False if all emoticons in y is recalled by y_pred
        :param y_pred: predicted rank
        :param y: true rank
        :return: True or False
        """
        for rank in range(len(y)):
            emoti = y[rank]
            if emoti >= 0:
                if y_pred[rank]!= emoti:
                    return True
            else:# not present emoticons follow
                return False
        return False

    def splitCriterion(self, y_s, w_s):
        """
        calculate the split_criterion of the tree node,
        current default is gini_rank
        :param y_s:
        :param w_s:
        :return: float
        """
        samples = np.arange(y_s.shape[0])
        n_rc = self.nRankClass(y_s,w_s,samples)
        gini = self.giniRank_e(n_rc)
        return gini

    def bestSplit(self, x, y, weights, samples):
        """
        find best split among all binary splits over any feature
        current for weighted gini for rank
        :param x: whole train features
        :param y: whole train labels
        :param weights: whole train weights
        :param samples: samples in current node
        :return: best_cri(float weighted sum of split split_criterion), best_split([feature_index, split_value])
                best_sets([samps1, samps2] the samples in left and right children)
        """

        Nsamp = len(samples)
        Ranks = y.shape[1]
        Nclass = Ranks  # full rank
        Nfeature = x.shape[1]

        min_gini = [np.nan for f in range(Nfeature)]
        best_split = [None for f in range(Nfeature)]
        best_sets = [[] for f in range(Nfeature)]
        gini_s = [[0,0] for f in range(Nfeature)]

        for feature in range(Nfeature):
            min_gini_sub = -1
            temp = [(x[samples[i], feature], samples[i]) for i in range(Nsamp)]
            dtype = [("value", float), ("index", int)]
            x_ord = np.sort(np.array(temp, dtype=dtype), order="value")

            n_rc = [[[0.0 for i in range(Nclass)] for j in range(Ranks)] for i in range(2)]
            # n_rc[0] results for tb; n_rc[1] results for fb

            n_rc[0] = self.nRankClass(y, weights, samples)

            j = 0
            old_value = x_ord[0][0]
            for i in range(Nsamp - 1):
                value = x_ord[i][0]
                if value == old_value:
                    n_rc[0] = self.nRankClassChange(n_rc[0], y[x_ord[i][1], :], weights[x_ord[i][1]], "del")
                    n_rc[1] = self.nRankClassChange(n_rc[1], y[x_ord[i][1], :], weights[x_ord[i][1]], "add")
                    if x_ord[i + 1][0] > value:
                        j = i + 1
                        old_value = x_ord[i + 1][0]
                        gini_tb = self.giniRank_e(n_rc[0])
                        gini_fb = self.giniRank_e(n_rc[1])
                        gini = gini_tb + gini_fb
                        # print "current gini", gini_tb, gini_fb
                        # print "current sets", [[y[n,:] for n in range(j,Nsamp)], [y[m,:] for m in range(j)]]
                        if min_gini_sub < 0 or min_gini_sub >= gini:
                            min_gini_sub = gini
                            best_split_sub = j
                            gini_s_sub = [gini_tb, gini_fb]
            if min_gini_sub >= 0:
                min_gini[feature] = min_gini_sub
                best_split[feature] = x_ord[best_split_sub][0]
                best_sets[feature] = [[x_ord[i][1] for i in range(best_split_sub,Nsamp)],
                                      [x_ord[j][1]for j in range(best_split_sub)]]
                gini_s[feature] = gini_s_sub
        gini_min = min(min_gini)
        feature_min = min_gini.index(gini_min)
        best_split = best_split[feature_min]
        best_sets = best_sets[feature_min]
        gini_s_sub = gini_s[feature_min]
        return gini_min, [feature_min, best_split], best_sets

    def nRankClass(self, y, weights, samples):
        if type(y) != np.ndarray:
            y = np.array(y)
        Ranks = y.shape[1]
        Nclass = Ranks
        n_rc = [[0.0 for i in range(Nclass)] for j in range(Ranks)]
        for rank in range(Ranks):
            for sample in samples:
                emoti = int(y[sample, rank])
                if emoti >= 0:
                    n_rc[rank][emoti] += weights[sample]
        return n_rc

    def nRankClassChange(self, n_rc, y_rank, weight, method):
        Ranks = len(n_rc)
        for rank in range(Ranks):
            emoti = int(y_rank[rank])
            if emoti < 0:
                break
            if method == "del":
                factor = -1
            elif method == "add":
                factor = +1
            else:
                raise(ValueError, "not supporting other change")
            n_rc[rank][emoti] = n_rc[rank][emoti] + factor * weight
            if n_rc[rank][emoti] < 0:
                print "wrong delete"
        return n_rc

    def giniRank_e(self, n_rc):
        Ranks = len(n_rc)
        Nclass = len(n_rc[0])
        gini_rank = 0.0
        for rank in range(Ranks):
            gini = 0.0
            n = sum(n_rc[rank])
            if n == 0:
                gini_rank += gini * n
            else:
                gini = sum([n_rc[rank][i] * (n - n_rc[rank][i]) for i in range(Nclass)]) * 1.0 / n
                gini_rank += gini
        return gini_rank

    def splitGain(self, cri_cur, cri_split):
        """
        calculate the gain of split in terms of split_criterion
        currently for variance type, the smaller the better
        :param cri_cur: criterion for current node
        :param cri_split: combined criterion for children
        :return: float
        """
        return (cri_cur - cri_split)

    def pruneGain(self, split_cri_gain, tb_Nsamp, fb_Nsamp):
        """
        calculate the gain compared to complete split of the substree rooted at current node
        for now, gain is concerned about misclassification rate
        :param split_cri_gain: gain of current split in terms of split_criterion
        :return: float
        """
        Nsamp = tb_Nsamp + fb_Nsamp
        split_mis_rate = (tb_Nsamp * self.tb.mis_rate + fb_Nsamp * self.fb.mis_rate)/Nsamp
        return (self.mis_rate - split_mis_rate)


def crossValidate(x,y, cv=5, alpha = 0, rank_weight = False, stop_criterion_mis_rate = None, stop_criterion_min_node = 1,
                  stop_criterion_gain = 0.0):

    results = {"alpha": [], "perf": []}

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
        if rank_weight:
            weights = rank2Weight(y_train)
        else:
            weights = None
        tree = DecisionTree().buildtree(x_train,y_train, weights,
                                        stop_criterion_mis_rate= stop_criterion_mis_rate,
                                        stop_criterion_min_node = stop_criterion_min_node,
                                        stop_criterion_gain=stop_criterion_gain)

        # performance measure
        alpha_sel, y_pred = alpha, tree.predict(x_test, alpha)
        results["perf"].append(LogR.perfMeasure(y_pred, y_test, rankopt=True))
        results["alpha"].append(alpha_sel)
        print alpha_sel, "alpha"

    for key in results.keys():
        item = np.array(results[key])
        mean = np.nanmean(item, axis=0)
        std = np.nanstd(item, axis=0)
        results[key] = [mean, std]

    return results


def label2Rank(y):
    y = map(LogR.rankOrder, y)
    y = np.array(y, dtype=int)
    return y


def rank2Weight(y_s):
    """

    :param y_s: old rank form
    :return: weights np.ndarray
    """
    Nsamp = y_s.shape[0]
    Nclass = y_s.shape[1]
    weights = np.ones(Nsamp,dtype=np.float32)
    paircomp, paircomp_sub = rankPairwise(y_s)
    for samp in range(Nsamp):
        rank = y_s[samp]
        emoti_list = [emoti for emoti in range(Nclass)]
        for i in range(Nclass-1):
            emoti = int(rank[i])
            if emoti < 0:# following rank positions contain only -1
                break
            emoti_list.remove(emoti)
            for emoti_cmp in emoti_list:
                n_big = paircomp[emoti][emoti_cmp]   # including comparison with missing emoticons
                n_small = paircomp[emoti_cmp][emoti]
                if n_big > 0 and n_small > 0: # valid only when both > 0
                    weight_new = (1.0 * n_small)/n_big
                    if weights[samp] < weight_new:
                        weights[samp] = weight_new
    return weights


def rank2Weight_cost(y_s, cost_level = np.arange(1.0,0.4,-0.1,dtype=np.float16)):
    """

    :param y_s: old rank form
    :param cost_level: levels of cost to be chosen
    :return: weights np.ndarray
    """
    Nsamp = y_s.shape[0]
    Nclass = y_s.shape[1]
    print cost_level, cost_level[-1] ### test
    weights = np.ones(Nsamp,dtype=np.float32)*cost_level[-1]
    paircomp, paircomp_sub = rankPairwise(y_s)
    for samp in range(Nsamp):
        rank = y_s[samp]
        emoti_list = [emoti for emoti in range(Nclass)]
        for i in range(Nclass-1):
            emoti = int(rank[i])
            if emoti < 0:# following rank positions contain only -1
                break
            emoti_list.remove(emoti)
            for emoti_cmp in emoti_list:
                n_big = paircomp[emoti][emoti_cmp]   # including comparison with missing emoticons
                n_small = paircomp[emoti_cmp][emoti]
                if n_big > 0 and n_small > 0: # valid only when both > 0
                    weight_new_score = (1.0 * n_small)/n_big
                    cost = int(- np.log2(weight_new_score))
                    if cost < 0:
                        weight_new = cost_level[0]
                    elif cost > (len(cost_level)-1):
                        weight_new = cost_level[-1]
                    else:
                        weight_new = cost_level[cost]
                    if weights[samp] < weight_new:
                        weights[samp] = weight_new
    print "cost stats, mean, std, min, max ", np.mean(weights), np.std(weights), np.min(weights), np.max(weights) ### test

    return weights


def rankPairwise(y_s):
    Nclass = y_s.shape[1]
    # first and second dimension is the indices of emoticon pairs, the first value is # posts first emoticon rank higher
    # than the second, vise versa
    paircomp = [[0 for i in range(Nclass)] for j in range(Nclass)] # take not appearing emoticons to rank the lowest
    paircomp_sub = [[0 for i in range(Nclass)] for j in range(Nclass)] # excluding not appearing emoticons
    y_s = y_s.tolist()
    for post in y_s:
        emoti_list = [emoti for emoti in range(Nclass)]
        for i in range(Nclass-1):
            emoti = int(post[i])
            if emoti < 0:
                break
            emoti_list.remove(emoti)
            for emoti_cmp in emoti_list:
                paircomp[emoti][emoti_cmp] += 1
                if emoti_cmp in post:
                    paircomp_sub[emoti][emoti_cmp] += 1
    return paircomp, paircomp_sub


def dataSimulated(Nsamp, Nfeature, Nclass):
    np.random.seed(seed=10)
    x = np.arange(Nsamp*Nfeature,dtype="float").reshape([Nsamp,Nfeature])
    x += np.random.random(x.shape)*10
    y = np.random.random(Nsamp*Nclass).reshape([Nsamp, Nclass])
    y *= 2
    y = y.astype(int)
    y = map(LogR.rankOrder,y)
    y = np.array(y)
    return x,y

if __name__ == "__main__":
    ### test ###
    # x,y = dataSimulated(8, 3, 6)
    # print x
    # print y
    # Nsamp = x.shape[0]
    # weight = 1.0/Nsamp
    # weights = np.array([weight for i in range(Nsamp)], dtype = np.float32)
    # print weights
    # tree = DecisionTree().buildtree(x,y,weights, stop_criterion_mis_rate=0.4)
    # tree.printtree()
    # for i in range(x.shape[0]):
    #     y_pred = tree.predict(x[i])
    #     print y_pred, y[i]



    x,y = LogR.dataClean("data/nytimes_Feature_linkemotion.txt")
    y = label2Rank(y)
    # weights = rank2Weight(y)
    # paircmp, _ = rankPairwise(y)
    # print "\n".join(map(str,paircmp))
    # # print weights
    # print np.max(weights), np.min(weights), np.mean(weights), np.std(weights)
    # weights = rank2Weight(y)
    # print np.max(weights), np.min(weights), np.mean(weights), np.std(weights)

    result = crossValidate(x, y, stop_criterion_mis_rate=0.0, rank_weight = False)
    # write2result #
    file = open("result_dt_nytimes.txt","a")
    file.write("number of samples: %d\n" % x.shape[0])
    file.write("NONERECALL: %f\n" % NONERECALL)
    file.write("CV: %d\n" % 5)
    file.write(str(result)+"\n")
    file.close()
    print result
