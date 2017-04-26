"""
Implementing Wang, Ting, Dashun Wang, and Fei Wang. "Quantifying herding effects in crowd wisdom." SIGKDD 2014 as [1]
batch fitting with common theta matrix and f function
"""
import numpy as np
import math
import random
from Heard import Heard
from Heard import cumulate
from Heard import funcComplexity
from Heard import discreteODE
from Heard import synthetic2
# from matplotlib import pyplot as plt
import copy
from MonteCarloTimeSeries import MonteCarloTimeSeries
from datetime import datetime
from datetime import timedelta


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
        self.llh_minus_lamda = self.Qcalculate(y_s)
        llh_minus_lamda_test = self.llhMinusLamdacheck(y_s)
        try:
            assert int(self.llh_minus_lamda*1000) == int(llh_minus_lamda_test*1000)
        except AssertionError, e:
            print "Q ", self.llh_minus_lamda
            print "llh", llh_minus_lamda_test
            self.printmodel()
            raise e

        llh_minus_lamda_old = self.llh_minus_lamda

        # iterative optimization #
        for iteration in range(max_iteration):
            # check converging, correctness test #
            if iteration > 0:
                if self.llh_minus_lamda > llh_minus_lamda_old - threshold:
                    if int(self.llh_minus_lamda * 1000) > int(llh_minus_lamda_old * 1000):
                        raise ValueError("wrong iteration")
                    else:
                        print "converge at iter ", iteration
                        self.printmodel()
                        break
                else:
                    llh_minus_lamda_old = self.llh_minus_lamda
            # parameter updates #
            self.update(x_s, y_s)
            self.intermediate(x_s, y_s)

            ### check the update process ###
            Q_updated = self.Qcalculate(y_s)
            try:
                assert Q_updated <= llh_minus_lamda_old + 0.001
            except AssertionError, e:
                print "Q_min: ", Q_updated
                print "llh_minus_lamda_old: ", llh_minus_lamda_old
                print self.printmodel()
                raise e

            # calculate new likelihood #
            self.setold()
            self.llh_minus_lamda = self.Qcalculate(y_s)

        return self

    def predict(self, time_target, Nsamp = 2000):
        state_targets = []
        if type(time_target) != list:
            time_targets = [time_target for m in range(self.M)]
        else:
            time_targets = time_target
        for m in range(self.M):
            MC = MonteCarloTimeSeries(state_init = self.state_endoftrain[m], time_init = self.L[m], mu = self.mu[m],
                                      theta = self.theta, f = self.fExtend(self.f, time_targets[m]), Nsamp = Nsamp)
            start = datetime.now()
            state_targets.append(MC.predict(time_targets[m]))
            duration = (datetime.now() - start).total_seconds()
            print "MC simulation takes %f seconds" % duration
        return state_targets

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
        self.mu_old = copy.deepcopy(self.mu)
        self.f_old = copy.deepcopy(self.f)
        self.theta_old = copy.deepcopy(self.theta)
        self.phi_old = copy.deepcopy(self.phi)
        self.beta_old = copy.deepcopy(self.beta)

        self.c_old = [np.zeros(self.L[m], dtype=np.float64) for m in range(self.M)]
        for m in range(self.M):
            phi_old = self.phi_old[m]
            beta_old = self.beta_old[m]
            L = self.L[m]
            for i in range(L):
                sum_phi2 = 0.0
                sum_betaphi = 0.0
                sum_phi = 0.0
                sum_expphi = 0.0
                for k in range(self.K):
                    sum_phi2 += math.pow(phi_old[i, k], 2.0)
                    sum_betaphi += (beta_old[i, k] * phi_old[i, k])
                    sum_phi += phi_old[i, k]
                    sum_expphi += np.exp(phi_old[i, k])
                self.c_old[m][i] = sum_phi2 - sum_betaphi - math.pow(sum_phi, 2.0) / self.K + np.log(sum_expphi)
        return self

    def Qcalculate(self, y_s):
        # complexity term #
        sum_theta2 = np.sum(np.power(self.theta,2.0))
        sum_mu2 = sum([np.sum(np.power(self.mu[m],2.0)) for m in range(self.M)]) # add all mu without scaling down
        sum_a = 0.0
        sum_b = 0.0
        for m in range(self.M):
            L = self.L[m]
            phi = self.phi[m]
            phi_old = self.phi_old[m]
            beta_old = self.beta_old[m]
            y = y_s[m]
            for i in range(L):
                sum_a_i = 0.0
                sum_b_i1 = 0.0
                sum_b_i2 = 0.0
                for k in range(self.K):
                    sum_a_i += (math.pow(phi[i,k], 2.0) + (beta_old[i, k] - 2.0 * phi_old[i, k] - y[i, k]) * phi[i, k])
                    sum_b_i1 += (phi[i, k] - 2.0 * phi_old[i, k])
                    sum_b_i2 += phi[i, k]
                sum_a += sum_a_i
                sum_b += (sum_b_i1 * sum_b_i2)
        sum_complexity = sum_theta2 + sum_mu2 + funcComplexity(self.f)
        sum_c = sum([np.sum(self.c_old[m]) for m in range(self.M)])
        sum_L = sum(self.L)

        Q = sum_a/sum_L - sum_b/self.K/sum_L + sum_complexity*self.lamda/2.0 + sum_c/sum_L
        return Q

    def llhMinusLamdacheck(self, y_s):
        llh = 0.0
        for m in range(self.M):
            L = self.L[m]
            beta = self.beta[m]
            y = y_s[m]
            for i in range(L):
                for k in range(self.K):
                    llh += (y[i,k] * np.log(beta[i,k]))
        llh = llh / sum(self.L)
        sum_theta2 = np.sum(np.power(self.theta,2.0))
        sum_mu2 = sum([np.sum(np.power(self.mu[m],2.0)) for m in range(self.M)])
        sum_complexity = sum_theta2 + sum_mu2 + funcComplexity(self.f)

        llh_minus_lamda = -llh + sum_complexity * self.lamda / 2.0
        return llh_minus_lamda

    def update(self, x_s, y_s):
        # updating f #
        A = np.zeros(self.f.shape[0], dtype = np.float64)
        B = np.zeros(self.f.shape[0], dtype = np.float64)
        for m in range(self.M):
            x = x_s[m]
            y = y_s[m]
            L = self.L[m]
            theta_old = self.theta_old
            mu_old = self.mu_old[m]
            phi_old = self.phi_old[m]
            beta_old = self.beta_old[m]
            sum_L = sum(self.L)
            for i in range(L):
                sum_A1 = 0.0
                sum_AB2 = 0.0
                sum_B1 = 0.0
                for k in range(self.K):
                    prod_k_i = np.inner(theta_old[k], x[i])
                    sum_A1 += math.pow(prod_k_i, 2.0)
                    sum_AB2 += prod_k_i
                    sum_B1 += (2 * mu_old[k] - 2 * phi_old[i, k] + beta_old[i, k] - y[i, k]) * prod_k_i
                A[i] += sum_A1/sum_L - math.pow(sum_AB2, 2.0)/(sum_L * self.K)
                B[i] += sum_B1/sum_L + sum_AB2*(np.sum(phi_old[i])-np.sum(mu_old))*2.0/(sum_L*self.K)
        self.f = discreteODE(A, B, self.lamda, f0 = 0.0, f1 = 0.0)

        ## constant f ##
        if self.fCONSTANT:
            self.f = np.ones(self.f.shape[0], dtype=np.float64)

        # updating mu #
        sum_L = sum(self.L)
        for m in range(self.M):
            L = self.L[m]
            y = y_s[m]
            beta_old = self.beta_old[m]
            mu_old = self.mu_old[m]
            for k in range(self.K):
                sum_mu = 0.0
                for i in range(L):
                    sum_mu += (y[i,k] - beta_old[i,k])
                self.mu[m][k] = (self.K*sum_mu + 2*sum_L*(self.K-1)*mu_old[k]) / (2*sum_L*(self.K-1) + sum_L*self.K*self.lamda)

        # updating theta #
        sum_L = sum(self.L)
        for k in range(self.K):
            for k_ in range(self.K):
                sum_theta1 = 0.0
                sum_theta2 = 0.0
                sum_theta3 = 0.0
                for m in range(self.M):
                    L = self.L[m]
                    x = x_s[m]
                    y = y_s[m]
                    beta_old = self.beta_old[m]
                    theta_old = self.theta_old
                    for i in range(L):
                        sum_theta1 += (self.f[i] * x[i, k_] * (y[i, k] - beta_old[i, k]))
                        sum_theta2 += (math.pow(self.f[i], 2.0) * math.pow(x[i, k_], 2.0) * theta_old[k, k_])
                        sum_theta3 += (math.pow(self.f[i], 2.0) * math.pow(x[i, k_], 2.0))
                self.theta[k,k_] = (self.K*sum_theta1 + 2*(self.K-1)*sum_theta2) / (2*(self.K-1)*sum_theta3 + sum_L*self.K*self.lamda)
        return self

    def printmodel(self):
        print "------ fit model ------"
        print "llh_minus_lamda"
        print self.llh_minus_lamda, "\n"
        print "mu"
        print self.mu, "\n"
        print "theta"
        print self.theta, "\n"
        # plt.plot(self.f)
        # plt.show() ### test
        return self


if __name__ == "__main__":
    L = 1000
    M = 10
    time_init = 200
    time_target = 900
    np.random.seed(2017)
    mu = [np.array([-0.01*m, 0.01*m]) for m in range(M)]
    theta = np.array([[1.0,-1.0],[-1.0,1.0]])
    f = np.ones(L)
    y_s = []
    for m in range(M):
        y_s.append(synthetic2(L-m, mu=mu[m], theta=theta, f=f))
    # y = synthetic2(L, mu=np.array([1.0,1.0]), theta=np.array([[1.0,-1.0],[-1.0,1.0]]), f=np.ones(L))
    print map(lambda item: np.sum(item, axis=0, dtype=np.float64), y_s)
    y_s_cumulate = map(cumulate, y_s)
    # for m in range(M):
    #     y = y_s[m]
    #     y_cumulate = y_s_cumulate[m]
    #     for d in range(y.shape[1]):
    #         plt.plot(y_cumulate[:,d], label="%d, m=%d" % (d, m))
    # plt.legend()
    # plt.show()
    heard = HeardBatch().fit(map(lambda item: item[:time_init,:], y_s), lamda=1.0, f_constant=False)
    heard.printmodel()
    states_predicted = heard.predict(time_target)
    for m in range(M):
        print "init", y_s_cumulate[m][time_init]
        print "predicted", states_predicted[m]
        print "true", y_s_cumulate[m][time_target]