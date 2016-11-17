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



    def buildtree(self, x_train, y_train, weights = None, samples = None, stop_criterion_mis_rate = None,
                  stop_criterion_min_node = None, stop_criterion_gain = 0.0):
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


    def nodeResult(self, y_s, w_s):
        """
        calculate the predict result for the node
        :param y_s: labels of samples in the tree node
        :param w_s: weights of corresponding samples
        :return: type(y)
        """
        pass


    def misRate(self, y_s, w_s, result):
        """
        calculate the misclassification rate in the node
        :param y_s: labels of samples in the tree node
        :param w_s: the weights of corresponding samples
        :param result: predicted result for the node
        :return: float
        """
        pass


    def splitCriterion(self, y_s, w_s):
        """
        calculate the split_criterion of the tree node,
        current default is gini_rank
        :param y_s:
        :param w_s:
        :return: float
        """
        pass


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
                        if min_gini_sub < 0 or min_gini >= gini:
                            min_gini_sub = gini
                            best_split_sub = j
            if min_gini_sub >= 0:
                min_gini[feature] = min_gini_sub
                best_split[feature] = x_ord[best_split_sub][0]
                best_sets[feature] = [[x_ord[i][1] for i in range(best_split_sub,Nsamp)],
                                      [x_ord[j][1]for j in range(best_split_sub)]]
        gini_min = min(min_gini)
        feature_min = min_gini.index(gini_min)
        best_split = best_split[feature_min]
        best_sets = best_sets[feature_min]
        return gini_min, [feature_min, best_split], best_sets


    def nRankClass(self,y,weights, samples):
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
            if n < 1:
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


dt = DecisionTree()