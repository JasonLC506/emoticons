"""
Discrete output GMM prediction as general with constraint that covariance matrix must be diagonal
Default prediction of label ranking via GMM of preference matrix, referred to SMPrank.py
"""

import numpy as np
from SMPrank import SmpRank
from Math_Gaussian import Gaussian
import copy

MAX_ITERATION = 100

class GMM(object):
    def __init__(self):
        # model parameters #
        self.K = 0
        self.L = 0  # size of label set in ranking
        self.prior = None
        self.mu = None
        self.sigma = None   # diagonal terms in the same format of mu
        self.sigma_inv = None   # inv of covariance matrix for efficiency

        # prediction output #
        self.y_pref = None
        self.y_rank = None

        # intermediate paras for prediction #
        self.q = None   # E-step weight in EM
        self.w = None   # weight of square in M-step in EM
        self.llh = None # here used in prediction as predicted instance llh
        self.probcomp = None    # current prob for each component
        self.Y = None   # for test in M-step in EM

    def setparas(self, prior, mu, sigma):
        self.K = prior.shape[0]
        self.L = mu.shape[1]
        self.prior = prior
        self.mu = mu
        self.sigma = sigma
        self.sigma_inv = 1.0 / self.sigma
        return self

    def predict(self, max_iteration = MAX_ITERATION):
        self.initialize()
        self.llh = self.llhcal()
        llh_old = self.llh
        y_rank_old = self.y_rank
        print "initial with ranking ", y_rank_old
        for iter in range(max_iteration):
            self.Estep()
            # Q = self.Qcal()
            self.Mstep()
            # Q_new = self.Qcal()
            ### test for Q increase Mstep, which is not necessary due to approx TOUGH algo ###
            # try:
            #     assert int(1000*Q_new) >= int(1000*Q) # check Mstep
            # except AssertionError, e:
            #     print Q_new
            #     print Q
            #     print y_rank_old
            #     print self.y_rank
            #     print "Y matrix", self.Y
            #     agree_old = self.check_agree(self.Y, y_rank_old)
            #     agree_new = self.check_agree(self.Y, self.y_rank)
            #     print "agree old ", agree_old
            #     print "agree new ", agree_new
            #     raise e
            ### check ###
            self.llh = self.llhcal()
            if llh_old > self.llh:
                print "llh_old", llh_old
                print "llh", self.llh
                self.llh = llh_old
                self.y_rank = y_rank_old
                print "TOUGH aggregation fails, early stop at iter ", iter+1
                print "with ranking ", self.y_rank
                break   # end iteration when TOUGH fails
            # print "at the end of iter ", iter + 1
            # print "Q before Mstep ", Q
            # print "Q after Mstep ", Q_new
            # print "llh_old ", llh_old
            # print "llh_new ", self.llh
            # print "previous rank", y_rank_old
            # print "current rank", self.y_rank
            llh_old = self.llh
            # check converge #
            if self.rankequal(y_rank_old, self.y_rank):
                print "early converge at iter", iter+1
                print "with ranking ", self.y_rank
                break
            y_rank_old = self.y_rank
        return self.y_rank

    def initialize(self):
        # use SMPrank TOUGH aggregation as initialization #
        self.y_pref = np.tensordot(self.prior, self.mu, axes=(0,0))
        smp = SmpRank(K=1)
        self.y_rank = np.array(smp.aggregate(self.y_pref), dtype=np.int16)
        self.y_pref = smp.rank2pair(self.y_rank)    # keep y_pref and y_rank consistent
        return self

    def llhcal(self):
        self.probcomp = np.zeros(self.K, dtype=np.float64)
        for i in range(self.K):
            self.probcomp[i] = self.prior[i] * Gaussian(self.y_pref, self.mu[i], self.sigma[i], diagonal_variance=True)
        return np.log(np.sum(self.probcomp))

    def Estep(self):
        self.probcomp = np.zeros(self.K, dtype=np.float64)
        for i in range(self.K):
            self.probcomp[i] = self.prior[i] * Gaussian(self.y_pref, self.mu[i], self.sigma[i], diagonal_variance=True)
        self.q = self.probcomp/np.sum(self.probcomp)
        return self

    def Mstep(self):
        w_comp = np.zeros([self.K, self.L, self.L], dtype = np.float64)
        for i in range(self.K):
            w_comp[i] = self.sigma_inv[i] * self.q[i]
        mu_weighted = np.multiply(w_comp, self.mu)
        self.w = np.sum(w_comp, axis = 0)
        mu_mean = np.divide(np.sum(mu_weighted, axis = 0), self.w)
        # print "mu_mean", mu_mean
        Y = - 0.5 * (np.multiply(self.w, np.power((1-mu_mean), 2)) + np.transpose(np.multiply(self.w, np.power(mu_mean, 2))))
        Y_min = np.amin(Y)
        Y = Y - Y_min   # shift to all positive value
        for ind in range(Y.shape[0]):
            Y[ind,ind] = -1    # abandon diagonal terms
        self.Y = Y
        smp = SmpRank(K=1)
        self.y_rank = np.array(smp.aggregate(copy.deepcopy(self.Y)), dtype=np.int16)
        self.y_pref = smp.rank2pair(self.y_rank)
        return self

    def Qcal(self):
        self.probcomp = np.zeros(self.K, dtype=np.float64)
        for i in range(self.K):
            self.probcomp[i] = self.prior[i] * Gaussian(self.y_pref, self.mu[i], self.sigma[i], diagonal_variance=True)
        return np.inner(self.q, np.log(self.probcomp))

    def rankequal(self, rank1, rank2):
        equal = True
        for ind in range(rank1.shape[0]):
            if rank1[ind] != rank2[ind]:
                equal = False
                return equal
        return equal

    def check_agree(self, Y, rank):
        agree = 0.0
        for i in range(rank.shape[0]):
            for j in range(i+1, rank.shape[0]):
                pr = rank[i]
                po = rank[j]
                if pr >= 0 and po >= 0:
                    agree += Y[pr, po]
        return agree

if __name__ == "__main__":
    np.random.seed(2016)
    K = 3
    prior = np.random.random(K)
    L = 6
    mu = np.random.random(K*L*L).reshape([K,L,L])
    for i in range(L):
        mu[:,i,i] = 0.0
    sigma = np.random.random(K*L*L).reshape([K,L,L])
    print mu
    print "-------------------"
    print sigma
    gmm = GMM().setparas(prior, mu, sigma)
    print gmm.predict()