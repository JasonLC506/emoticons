"""
Implementing Wang, Ting, Dashun Wang, and Fei Wang. "Quantifying herding effects in crowd wisdom." SIGKDD 2014 as [1]
batch fitting with common theta matrix and f function
"""
import numpy as np
import math
import random
from Heard import Heard
from Heard import cumulate

class HeardBatch(Heard):
    def __init__(self):
        # model parameters #
        self.M = 0 # batch size
        self.mu = [] # intrinsic popularity of labels, length == K, size of label set
        self.f = [] # fitted herding amplitude function versus timestamp, length == L, longest sequence in training
        self.theta = [[]] # herding amplitude between each pair of labels, theta[i][j] is herding from j to i
        self.K = 0
        self.L = [] # length of training sequences
        self.fCONSTANT = True # model selection whether f is a constant function

        # hyperparameter #
        self.lamda = 1.0
        self.threshold = 0.001

        # intermediate parameter #
        self.llh_minus = 0.0
        self.llh_minus_lamda = 0.0
        self.phi = [[]] # shape[L,K]
        self.beta = [[]] # shape[L,K]

        # for iteration #
        self.mu_old = []
        self.f_old = []
        self.theta_old = [[]]
        self.phi_old = [[]]
        self.beta_old = [[]]
        self.c_old = [] # no need in parameter

        # for prediction #
        self.state_endoftrain = []

    def fit(self, y_s, lamda = 1.0, threshold = 0.001, max_iteration =100, f_constant = True):
        """

        :param y_s: list of label sequences, [np.ndarray([L[i],K]) for i in range(M)]
        :param lamda:
        :param threshold:
        :param max_iteration:
        :param f_constant:
        :return:
        """
        self.lamda = lamda
        self.threshold = threshold
        self.M = len(y_s)
        self.K = y_s[0].shape[1]
        self.fCONSTANT = f_constant
        self.L = map(lambda item: item.shape[0], y_s)

        # compute cumulative sequences #
        x_s = map(cumulate, y_s)
        self.state_endoftrain = map(lambda item: np.sum(item, axis=0)/item.shape[0], y_s)

        # initialize parameters #
        self.initialize(y_s)
        self.intermediate(x_s, y_s)

        # calculate initial surrogate function value #
        self.setold()

    def initialize(self, y_s):
        # mu #
        self.mu = map(self._mu_initialize, y_s)
        # f #
        self.f = np.zeros(max(self.L)) # longest of all L
        # theta #
        self.theta = (np.random.random([self.K, self.K]) - 0.5) * 0.1
        return self

    def _mu_initialize(self, y):
        mu = np.zeros(y.shape[1], dtype=np.float64)
        x_N = np.sum(y, axis=0)
        for i in range(y.shape[1]):
            if x_N[i] > 0:
                mu[i] = np.log(x_N[i] * 1.0 / y.shape[0])  # remove the denominator in [1]
            else:
                mu[i] = 0.0  # for label not present, treat it as appearing once
        return mu

    def intermediate(self, x_s, y_s):
        self.phi = [np.zeros([self.L[m], self.K], dtype=np.float64) for m in range(self.M)]
        self.beta = [np.zeros([self.L[m], self.K], dtype=np.float64) for m in range(self.M)]
        for m in range(self.M):
            L = self.L[m]
            mu = self.mu[m]
            x = x_s[m]
            y = y_s[m]
            for i in range(L):
                z = 0.0
                for k in range(self.K):
                    self.phi[m][i,k] = mu[k] + self.f[i] * np.inner(self.theta[k],x[i])
                    z += np.exp(self.phi[m][i,k])
                for k in range(self.K):
                    self.beta[m][i,k] = np.exp(self.phi[m][i,k]) / z
        return self

    def setold(self):
        self.mu_old = self.mu
        self.f_old = self.f
        self.theta_old = self.theta
        self.phi_old = self.phi