"""
Implementing Wang, Ting, Dashun Wang, and Fei Wang. "Quantifying herding effects in crowd wisdom." SIGKDD 2014 as [1]
"""
import numpy as np
import math

class Heard(object):
    def __init__(self):
        # model parameters #
        self.mu = [] # intrinsic popularity of labels, length == K, size of label set
        self.f = [] # fitted herding amplitude function versus timestamp, length == L, longest sequence in training
        self.theta = [[]] # herding amplitude between each pair of labels, theta[i][j] is herding from j to i
        self.K = 0
        self.L = 0 # length of training sequence

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


    def fit(self, y, lamda = 1.0, threshold = 0.001, max_interation = 100):
        """
        fit HEARD model by data y
        :param y: label sequence data, np.ndarray([L,K])
        :param lamda:
        :param threshold:
        :return:
        """
        self.lamda = lamda
        self.threshold = threshold
        self.K = y.shape[1]
        self.L = y.shape[0]

        # compute cumulative sequence #
        x = self.cumulate(y)

        # initialize parameters #
        self.initialize(y)
        self.intermediate(x, y)

        # calculate initial surrogate function value #
        self.setold()
        Q = self.Qcalculate(y) ### waiting for test of Q calculation
        self.llh_minus_lamda = Q

        llh_minus_lamda_old = self.llh_minus_lamda

        # iterative optimization #
        for iteration in range(max_interation):
            # check converging, correctness test #
            if iteration > 0:
                if self.llh_minus_lamda > llh_minus_lamda_old - threshold:
                    if self.llh_minus_lamda > llh_minus_lamda_old:
                        raise ValueError("wrong iteration")
                    else:
                        print "converge"
                        break
                else:
                    llh_minus_lamda_old = self.llh_minus_lamda
            # parameter updates #
            self.update(x, y)

            ### check the update process ###
            Q_updated = self.Qcalculate(y)
            assert Q_updated <= llh_minus_lamda_old

            # calculate new likelihood #
            self.setold()
            Q = self.Qcalculate(y) ### waiting for test of Qcalculation
            self.llh_minus_lamda = Q

        return self


    def cumulate(self, y):
        x = np.zeros(self.L * self.K, dtype=np.float16).reshape([self.L, self.K])
        for i in range(1, self.L):
            x[i] = y[i-1] + x[i-1]
        return x

    def initialize(self, y):
        # self.mu #
        self.mu = np.zeros(self.K, dtype=np.float16)
        x_N = np.sum(y, axis=0)
        for i in range(self.K):
            if x_N[i] > 0:
                self.mu[i] = np.log(x_N[i] * 1.0) # remove the denominator in [1]
            else:
                self.mu[i] = 0.0 # for label not present, treat it as appearing once
        # self.f #
        self.f = np.zeros(self.L)
        # self.theta #
        self.theta = np.random.random([self.K, self.K]) - 0.5
        return self

    def intermediate(self, x, y):
        self.phi = np.zeros([self.L, self.K], dtype=np.float16)
        self.beta = np.zeros([self.L, self.K], dtype=np.float16)
        for i in range(self.L):
            z = 0.0
            for k in range(self.K):
                self.phi[i,k] = self.mu[k] + self.f[i] * np.inner(self.theta[k], x[i])
                z += self.phi[i,k]
            for k in range(self.K):
                self.beta[i,k] = self.phi[i,k] / z
        return self

    def setold(self):
        self.mu_old = self.mu
        self.f_old = self.f
        self.theta_old = self.theta
        self.phi_old = self.phi
        self.beta_old = self.beta

        self.c_old = np.zeros(self.L, dtype=np.float16)
        for i in range(self.L):
            sum_phi2 = 0.0
            sum_betaphi = 0.0
            sum_phi = 0.0
            sum_expphi = 0.0
            for k in range(self.K):
                sum_phi2 += math.pow(self.phi_old[i,k], 2.0)
                sum_betaphi += (self.beta_old[i,k] * self.phi_old[i,k])
                sum_phi += self.phi_old[i,k]
                sum_expphi += math.exp(self.phi_old[i,k])
            self.c_old[i] = sum_phi2 - sum_betaphi - math.pow(sum_phi, 2.0)/self.K + np.log(sum_expphi)

        return self

    def Qcalculate(self, y):
        sum_a = 0.0
        sum_b = 0.0
        sum_theta2 = 0.0
        sum_mu2 = 0.0
        for k in range(self.K):
            sum_mu2 += math.pow(self.mu[k], 2.0)
        for i in range(self.L):
            sum_a_i = 0.0
            sum_b_i1 = 0.0
            sum_b_i2 = 0.0
            for k in range(self.K):
                sum_a_i += (math.pow(self.phi[i,k], 2.0) + (self.beta_old[i,k] - 2.0*self.phi_old[i,k] - y[i,k]) * self.phi[i,k])
                sum_b_i1 += (self.phi[i,k] - 2.0*self.phi_old[i,k])
                sum_b_i2 += self.phi[i,k]
                sum_theta2 += (math.pow(self.phi[i,k], 2.0))
            sum_a += sum_a_i
            sum_b += (sum_b_i1 * sum_b_i2)
        sum_complexity = sum_theta2 + sum_mu2 + funcComplexity(self.f)
        sum_c = np.sum(self.c_old)

        Q = sum_a/self.L - sum_b/self.K/self.L + sum_complexity*self.lamda/2.0 + sum_c/self.L
        return Q

    def update(self, x, y):
        pass

def funcComplexity(f):
    """
    calculate complexity of discrete function based on sum of square of first derivative
    :param f: array of function value
    :return:
    """
    L = f.shape[0]
    complex = 0.0
    for i in range(1,L):
        complex += math.pow((f[i]-f[i-1]), 2.0)
    return complex