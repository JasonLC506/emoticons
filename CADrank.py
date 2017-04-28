"""
Implemented based on Grbovic, M., Djuric, N. and Vucetic, S., 2013, August. Multi-Prototype Label Ranking with Novel Pairwise-to-Total-Rank Aggregation. In IJCAI. [1]
in terms of preference matrix Gaussian and aggregation method
where the variance of Gaussian here is trainable scalar rather than fixed in [1].
The Main structure of two GMMs mapping is based on Song, X., Wu, M., Jermaine, C. and Ranka, S., 2007. Conditional anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 19(5). [2]
"""

import numpy as np
from logRegFeatureEmotion import dataClean
from DecisionTreeWeight import label2Rank
from SMPrank import SmpRank

THRESHOLD = 0.001
MAX_ITERATION = 100

class CADrank(object):
    def __init__(self, Nu, Nv):
        # Model prefixed parameters #
        self.Nu = int(Nu)    # number of Gaussians for feature space
        self.Nv = int(Nv)    # number of Gaussians for target space

        # data parameters #
        self.Du = 0     # dim of feature space
        self.Nclass = 0 # number of labels for ranking
        self.Nsamp = 0  # number data samples, for fitting temporarily

        # model parameters #
        self.pu = np.zeros(self.Nu, dtype=np.float64)    # prior prob for Gaussians for feature space
        self.mu_u = None                            # Gaussian means for feature space
        self.sigma_u = None                         # Gaussian variance matrices for feature space
        self.mu_v = None                            # Gaussian means for target space, here is in preference matrix format
        self.sigma_v = None                         # Gaussian variance scalars for target space
        self.map_uv = np.zeros([self.Nu, self.Nv], dtype=np.float64) # mapping prob from feature Gaussian to target Gaussian

        # intermediate parameters #
        self.llh = 0        # current log-likelihood
        self.b = None       # weighted parameter used in [2]

    def fit(self, x, y, threshold = THRESHOLD, max_iteration = MAX_ITERATION):
        """
        input x: feature variables np.ndarray([Nsamp, Du])
        input y: target variables, here preference matrices np.ndarray([Nsamp, Nclass, Nclass])
        """
        ## set data parameters ##
        self.Nsamp = x.shape[0]
        self.Du = x.shape[1]
        self.Nclass = y.shape[1]

        ## initialize ##
        self.initialize(x, y)
        self.llh, pxu, pyv = self.llhcal(x, y)

        ## EM ##
        llh_old = self.llh
        for iteration in range(max_iteration):
            ### test ###
            print "---------  before iter %d ---------" % iteration
            print "llh_old: ", llh_old
            # E-step #
            self.Estep(pxu, pyv)    # updating weighted paramters self.b with intermediate paras

            ### test ###
            # llh_pyv = np.sum(np.multiply(np.sum(self.b, axis=1), np.log(pyv)))
            # llh_map = np.sum(np.multiply(np.sum(self.b, axis=0), np.log(self.map_uv)))
            # pxu_temp = np.zeros([self.Nsamp, self.Nu], dtype=np.float64)
            # for isamp in range(self.Nsamp):
            #     for iu in range(self.Nu):
            #         pxu_temp[isamp, iu] = Gaussian(x[isamp], self.mu_u[iu], self.sigma_u[iu])
            # llh_pxu_temp = np.sum(np.multiply(np.sum(self.b, axis=2), np.log(pxu_temp)))
            # llh_pu = np.sum(np.multiply(np.sum(self.b, axis=(0,2)), np.log(self.pu)))

            # M-step #
            self.Mstep(x, y)    # updating model parameters
            # calculate new llh #
            self.llh, pxu, pyv = self.llhcal(x, y)

            ### test ###
            # converge with precision limit reached #
            llh_pyv_new = 0.0
            b_yv = np.sum(self.b, axis=1)
            for isamp in range(self.Nsamp):
                for iv in range(self.Nv):
                    prob_yv = pyv[isamp,iv]
                    if prob_yv == 0.0:
                        if b_yv[isamp,iv] == 0.0:
                            llh_pyv_new += 0.0
                        else:
                            print "b", b_yv[isamp, iv]
                            print "pyv", pyv[isamp,iv]
                            print "y", y[isamp]
                            print "mu_v", self.mu_v[iv]
                            print "sigma_v", self.sigma_v[iv]
                            print "stop with precision limit at iter ", iteration
                            return self
                    else:
                        llh_pyv_new += (b_yv[isamp,iv] * np.log(prob_yv))
            # # llh_pyv_new = np.sum(np.multiply(np.sum(self.b, axis=1), np.log(pyv)))
            # llh_map_new = np.sum(np.multiply(np.sum(self.b, axis=0), np.log(self.map_uv)))
            # pxu_temp = np.zeros([self.Nsamp, self.Nu], dtype=np.float64)
            # for isamp in range(self.Nsamp):
            #     for iu in range(self.Nu):
            #         pxu_temp[isamp, iu] = Gaussian(x[isamp], self.mu_u[iu], self.sigma_u[iu])
            # llh_pxu_temp_new = np.sum(np.multiply(np.sum(self.b, axis=2), np.log(pxu_temp)))
            # llh_pu_new = np.sum(np.multiply(np.sum(self.b, axis=(0, 2)), np.log(self.pu)))
            # try:
            #     assert llh_pyv_new >= llh_pyv
            #     assert llh_map_new >= llh_map
            #     assert llh_pxu_temp_new >= llh_pxu_temp
            #     assert llh_pu_new >= llh_pu
            # except AssertionError,e:
            #     print "------ Assertion error in M-step ------"
            #     print "pyv", pyv
            #     print "sigma_v", self.sigma_v
            #     print llh_pyv_new, llh_pyv
            #     print llh_map_new, llh_map
            #     print llh_pxu_temp_new, llh_pxu_temp
            #     print llh_pu_new, llh_pu
            #     raise e

            # converge check #
            if llh_old + threshold > self.llh:
                print "early converged at ", iteration
                print "resulting llh ", self.llh
                break
            llh_old = self.llh

        return self

    def predict(self, x_test):
        # ## batch predict ##
        # if x_test.dim != 1:
        #     N = x_test.shape[0]
        #     y_rank_pred = [[] for i in range(N)]
        #     for samp in range(N):
        #         y_rank_pred[samp] = self.predict(x_test[samp])
        #     return np.array(y_rank_pred, dtype = np.int8)

        ## weighted average of preference matrices ##
        pxu = self._pxucal(x_test) # shape[Nsamp_test, self.Nu]
        pv = np.dot(pxu, self.map_uv)


    def initialize(self, x, y):
        # pu #
        self.pu = np.random.random(self.Nu)
        self.pu = self.pu / np.sum(self.pu)     # distribution sum to 1
        # mu_u #
        self.mu_u = np.random.random(self.Nu * self.Du).reshape([self.Nu, self.Du])
        # sigma_u #
        self.sigma_u = variance(x, self.mu_u, np.ones([x.shape[0], self.Nu])) # uniform prior to set initial variance
        # mu_v #
        self.mu_v = np.random.random(self.Nv * self.Nclass * self.Nclass).reshape([self.Nv, self.Nclass, self.Nclass])
        for i in range(self.Nclass):
            for j in range(i,self.Nclass):
                for iv in range(self.Nv):
                    if i != j:
                        self.mu_v[iv, i, j] = 1.0 - self.mu_v[iv, j, i]     # sum to one
                    else:
                        self.mu_v[iv, i, j] = 0.0   # empty for diagonal
        # sigma_v #
        self.sigma_v = np.ones(self.Nv, dtype=np.float64)
        for iv in range(self.Nv):
            self.sigma_v[iv] = float(np.mean(np.power(np.linalg.norm(
                y - self.mu_v[iv],
                axis = (1,2)
            ), 2)))
        # self.map_uv #
        self.map_uv = np.random.random(self.Nu * self.Nv).reshape([self.Nu, self.Nv])
        map_uv_sum = np.sum(self.map_uv, axis=1, keepdims=True)
        self.map_uv[:] = self.map_uv[:]/map_uv_sum[:]
        return self

    def llhcal(self, x, y):
        # p(x[k] belongs to u[iu]) #
        pxu = self._pxucal(x)
        # p(y[k]|v[iv]) #
        pyv = np.zeros([self.Nsamp, self.Nv], dtype = np.float64)
        for isamp in range(self.Nsamp):
            for iv in range(self.Nv):
                pyv[isamp, iv] = Gaussian(y[isamp], self.mu_v[iv], self.sigma_v[iv], scalar_variance = True)
        # map already calculated #
        # llh #
        core = np.zeros(self.Nsamp, dtype = np.float64)
        for isamp in range(self.Nsamp):
            for iu in range(self.Nu):
                core[isamp] += pxu[isamp, iu] * np.inner(self.map_uv[iu], pyv[isamp])
        llh = np.sum(np.log(core)) / self.Nsamp

        return llh, pxu, pyv

    def _pxucal(self, x):
        pxu = np.zeros([self.Nsamp, self.Nu], dtype = np.float64)
        for isamp in range(self.Nsamp):
            pxu_partial_sum = 0.0
            for iu in range(self.Nu):
                pxu[isamp, iu] = Gaussian(x[isamp], self.mu_u[iu], self.sigma_u[iu]) * self.pu[iu]
                pxu_partial_sum += pxu[isamp, iu]
            pxu[isamp,:] = pxu[isamp,:] / pxu_partial_sum
        return pxu

    def Estep(self, pxu, pyv):
        # directly calculated from intermediate parameters to reduce redundancy #
        self.b = np.zeros([self.Nsamp, self.Nu, self.Nv], dtype = np.float64)
        for isamp in range(self.Nsamp):
            b_samp_sum = 0.0
            for iu in range(self.Nu):
                for iv in range(self.Nv):
                    self.b[isamp, iu, iv] = pxu[isamp, iu] * pyv[isamp, iv] * self.map_uv[iu, iv]
                    b_samp_sum += self.b[isamp, iu, iv]
            self.b[isamp,:,:] = self.b[isamp,:,:] / b_samp_sum
        return self

    def Mstep(self, x, y):
        ## update parameters ##
        b_part_sum_02 = np.sum(self.b, axis = (0,2))
        b_part_sum_01 = np.sum(self.b, axis = (0,1))
        # pu #
        self.pu = b_part_sum_02 / np.sum(self.b)
        # mu_u #
        self.mu_u = np.transpose(np.sum(np.dot(np.transpose(x), np.transpose(self.b, axes = (1,0,2))), axis = 2))
        for iu in range(self.Nu):
            self.mu_u[iu,:] = self.mu_u[iu,:] / b_part_sum_02[iu]
        # sigma_u #
        self.sigma_u = variance(x, self.mu_u, weights = np.sum(self.b, axis = 2))
        # mu_v #
        self.mu_v = np.transpose(np.sum(np.dot(np.transpose(y, axes = (1,2,0)), np.transpose(self.b, axes = (1,0,2))), axis = 2),
                                 axes = (2,0,1))
        for iv in range(self.Nv):
            self.mu_v[iv,:,:] = self.mu_v[iv,:,:] / b_part_sum_01[iv]
        # sigma_v #
        var_yv = np.zeros([self.Nsamp, self.Nv], dtype = np.float64)
        for isamp in range(self.Nsamp):
            for iv in range(self.Nv):
                var_yv[isamp,iv] = np.power(np.linalg.norm(y[isamp] - self.mu_v[iv]),2)
        self.sigma_v = np.zeros(self.Nv, dtype= np.float64)
        for iv in range(self.Nv):
            self.sigma_v[iv] = np.inner(np.sum(self.b, axis = 1)[:,iv], var_yv[:,iv]) / b_part_sum_01[iv] / np.power(self.Nclass,2)
        # map_uv #
        self.map_uv = np.sum(self.b, axis = 0)
        for iu in range(self.Nu):
            self.map_uv[iu,:] = self.map_uv[iu,:] / b_part_sum_02[iu]

        return self


def variance(x, mu_s, weights):
    """
    calculate MLE variance estemation for multivariate Gaussian model
    input x: data np.ndarray([Nsamp, Du])
    input mu_s, set of means, np.ndarray([Nu, Du])
    weights: weights for each data and each mean np.ndarray([Nsamp, Nu])
    """
    Nsamp = x.shape[0]
    Nu = mu_s.shape[0]
    Du = x.shape[1]
    assert Du == mu_s.shape[1]
    var = np.zeros([Nu, Du, Du], dtype=np.float64)
    weight_sum = np.sum(weights, axis=0)
    for isamp in range(Nsamp):
        for iu in range(Nu):
            diff = x[isamp] - mu_s[iu]
            var[iu] += (np.outer(diff, diff) * weights[isamp, iu] / weight_sum[iu])
    return var

def Gaussian(x, mean, var, scalar_variance = False):
    y = x.flatten()
    u = mean.flatten()
    if scalar_variance:
        prob = np.exp(-0.5 * np.inner((y-u),(y-u)) / var) \
               / np.sqrt(np.power(2 * np.pi * var, u.shape[0]))
    else:
        prob = np.exp(-0.5 * np.inner((y- u), np.inner(np.linalg.inv(var), (y - u)))) \
           / np.sqrt(np.power(2 * np.pi, u.shape[0]) * abs(np.linalg.det(var)))
    return prob


def crossValid(x, y, cv = 5):
    pass


if __name__ == "__main__":
    np.random.seed(2017)
    x, y = dataClean("data/washington_Feature_linkemotion.txt")
    y = label2Rank(y)
    smp = SmpRank(K=1)
    y_pref = np.array(map(smp.rank2pair, y.tolist()), dtype=np.float64)
    cad = CADrank(Nu = 10, Nv = 20).fit(x,y_pref)
