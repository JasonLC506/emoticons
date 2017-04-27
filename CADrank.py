"""
Implemented based on Grbovic, M., Djuric, N. and Vucetic, S., 2013, August. Multi-Prototype Label Ranking with Novel Pairwise-to-Total-Rank Aggregation. In IJCAI. [1]
in terms of preference matrix Gaussian and aggregation method
where the variance of Gaussian here is trainable scalar rather than fixed in [1].
The Main structure of two GMMs mapping is based on Song, X., Wu, M., Jermaine, C. and Ranka, S., 2007. Conditional anomaly detection. IEEE Transactions on Knowledge and Data Engineering, 19(5). [2]
"""

import numpy as np

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
        Nsamp = x.shape[0]
        self.Du = x.shape[1]
        self.Nclass = y.shape[1]

        ## initialize ##
        self.initialize(x, y)
        self.llh = self.llhcal(x, y)

        ## EM ##
        llh_old = self.llh
        for iteration in range(max_iteration):
            # E-step #
            self.Estep(x, y)    # updating weighted paramters self.b
            # M-step #
            self.Mstep(x, y)    # updating model parameters
            # calculate new llh #
            self.llh = self.llhcal(x, y)
            # converge check #
            if llh_old + threshold > self.llh:
                print "early converged at ", iteration
                print "resulting llh ", self.llh
                break

        return self

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
            self.sigma_v[iv] = float(np.mean(np.linalg.norm(
                y - self.mu_v[iv],
                axis = (1,2)
            )))
        # self.map_uv #
        self.map_uv = np.random.random(self.Nu * self.Nv).reshape([self.Nu, self.Nv])
        map_uv_sum = np.sum(self.map_uv, axis=1, keepdims=True)
        self.map_uv[:] = self.map_uv[:]/map_uv_sum[:]
        return self

    def llhcal(self, x, y):
        return 0

    def Estep(self, x, y):
        return self

    def Mstep(self, x, y):
        return self


def variance(x, mu_s, weights):
    """
    input x: data np.ndarray([Nsamp, Du])
    input mu_s, set of means, np.ndarray([Nu, Du])
    weights: weights for each data and each mean np.ndarray([Nsamp, Nu])
    """
    Nsamp = x.shape[0]
    Nu = mu_s.shape[0]
    Du = x.shape[0]
    assert Du == mu_s.shape[1]
    var = np.zeros([Nu, Du, Du], dtype=np.float64)
    weight_sum = np.sum(weights, axis=0)
    for isamp in range(Nsamp):
        for iu in range(Nu):
            var[iu] += (np.outer(x[isamp], mu_s[iu]) * weights[isamp, iu] / weight_sum[iu])
    return var

