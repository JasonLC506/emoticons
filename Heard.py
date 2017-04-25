"""
Implementing Wang, Ting, Dashun Wang, and Fei Wang. "Quantifying herding effects in crowd wisdom." SIGKDD 2014 as [1]
"""
import numpy as np
import math
import random
# from matplotlib import pyplot as plt
from MonteCarloTimeSeries import MonteCarloTimeSeries
from datetime import datetime
from datetime import timedelta
import copy

class Heard(object):
    def __init__(self):
        # model parameters #
        self.mu = [] # intrinsic popularity of labels, length == K, size of label set
        self.f = [] # fitted herding amplitude function versus timestamp, length == L, longest sequence in training
        self.theta = [[]] # herding amplitude between each pair of labels, theta[i][j] is herding from j to i
        self.K = 0
        self.L = 0 # length of training sequence
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

    def fit(self, y, lamda = 1.0, threshold = 0.001, max_iteration = 100, f_constant = True):
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
        self.fCONSTANT = f_constant

        # compute cumulative sequence #
        x = cumulate(y)
        self.state_endoftrain = np.sum(y,axis=0)/self.L

        # initialize parameters #
        self.initialize(y)
        self.intermediate(x, y)

        # calculate initial surrogate function value #
        self.setold()
        self.llh_minus_lamda = self.Qcalculate(y) ### tested of Q calculation
        # llh_minus_lamda = self.llhMinusLamdacheck(y)
        # ### test ###
        # try:
        #     assert llh_minus_lamda == Q ### test
        # except AssertionError,e:
        #     print "llh_minus_lamda", llh_minus_lamda
        #     print "Q", Q
        #     self.printmodel()
        #     print "llh not consistent with Q"
        # self.llh_minus_lamda = Q

        llh_minus_lamda_old = self.llh_minus_lamda

        # iterative optimization #
        for iteration in range(max_iteration):
            # check converging, correctness test #
            # self.printmodel() ### test
            if iteration > 0:
                if self.llh_minus_lamda > llh_minus_lamda_old - threshold:
                    if int(self.llh_minus_lamda*1000) > int(llh_minus_lamda_old*1000):
                        print "old llh ", llh_minus_lamda_old
                        print "current llh", self.llh_minus_lamda
                        raise ValueError("wrong iteration")
                    else:
                        print "converge"
                        self.printmodel()
                        break
                else:
                    llh_minus_lamda_old = self.llh_minus_lamda
            # parameter updates #
            self.update(x, y)
            self.intermediate(x, y)

            ### check the update process ###
            Q_updated = self.Qcalculate(y)
            try:
                assert Q_updated <= llh_minus_lamda_old + 0.001
            except AssertionError, e:
                print "Q_min: ", Q_updated
                print "llh_minus_lamda_old: ", llh_minus_lamda_old
                print self.printmodel()
                raise e

            # calculate new likelihood #
            self.setold()
            self.llh_minus_lamda = self.Qcalculate(y) ### waiting for test of Qcalculation
            llh_minus_lamda = self.llhMinusLamdacheck(y)
            # ### test ###
            # try:
            #     assert llh_minus_lamda == Q  ### test
            # except AssertionError, e:
            #     print "llh_minus_lamda", llh_minus_lamda
            #     print "Q", Q
            #     self.printmodel()
            #     print "llh not consistent with Q"
            # self.llh_minus_lamda = Q

        return self

    def predict(self, time_target, Nsamp = 2000):
        MC = MonteCarloTimeSeries(state_init = self.state_endoftrain, time_init = self.L, mu = self.mu, theta = self.theta,
                                  f = self.fExtend(self.f, time_target), Nsamp = Nsamp)
        start = datetime.now()
        state_target = MC.predict(time_target)
        duration = (datetime.now() - start).total_seconds()
        print "MC simulation takes %f seconds" % duration
        return state_target

    def fExtend(self, f, time_target):
        if self.fCONSTANT:
            return np.ones(time_target+1, dtype=np.float64)
        else:
            raise ValueError("not supporting general f function currently")

    def initialize(self, y):
        # self.mu #
        self.mu = np.zeros(self.K, dtype=np.float64)
        x_N = np.sum(y, axis=0)
        for i in range(self.K):
            if x_N[i] > 0:
                self.mu[i] = np.log(x_N[i] * 1.0 / self.L) # remove the denominator in [1]
            else:
                self.mu[i] = 0.0 # for label not present, treat it as appearing once
        # self.f #
        self.f = np.zeros(self.L)
        # self.theta #
        self.theta = (np.random.random([self.K, self.K]) - 0.5) * 0.1
        return self

    def intermediate(self, x, y):
        self.phi = np.zeros([self.L, self.K], dtype=np.float64)
        self.beta = np.zeros([self.L, self.K], dtype=np.float64)
        for i in range(self.L):
            z = 0.0
            for k in range(self.K):
                self.phi[i,k] = self.mu[k] + self.f[i] * np.inner(self.theta[k], x[i])
                z += np.exp(self.phi[i,k])
            for k in range(self.K):
                self.beta[i,k] = np.exp(self.phi[i,k]) / z
        return self

    def setold(self):
        self.mu_old = copy.deepcopy(self.mu)
        self.f_old = copy.deepcopy(self.f)
        self.theta_old = copy.deepcopy(self.theta)
        self.phi_old = copy.deepcopy(self.phi)
        self.beta_old = copy.deepcopy(self.beta)

        self.c_old = np.zeros(self.L, dtype=np.float64)
        for i in range(self.L):
            sum_phi2 = 0.0
            sum_betaphi = 0.0
            sum_phi = 0.0
            sum_expphi = 0.0
            for k in range(self.K):
                sum_phi2 += math.pow(self.phi_old[i,k], 2.0)
                sum_betaphi += (self.beta_old[i,k] * self.phi_old[i,k])
                sum_phi += self.phi_old[i,k]
                sum_expphi += np.exp(self.phi_old[i,k])
            self.c_old[i] = sum_phi2 - sum_betaphi - math.pow(sum_phi, 2.0)/self.K + np.log(sum_expphi)

        return self

    def Qcalculate(self, y):
        sum_a = 0.0
        sum_b = 0.0
        sum_theta2 = 0.0
        sum_mu2 = 0.0
        for k in range(self.K):
            sum_mu2 += math.pow(self.mu[k], 2.0)
            for k_ in range(self.K):
                sum_theta2 += (math.pow(self.theta[k, k_], 2.0))
        for i in range(self.L):
            sum_a_i = 0.0
            sum_b_i1 = 0.0
            sum_b_i2 = 0.0
            for k in range(self.K):
                sum_a_i += (math.pow(self.phi[i,k], 2.0) + (self.beta_old[i,k] - 2.0*self.phi_old[i,k] - y[i,k]) * self.phi[i,k])
                sum_b_i1 += (self.phi[i,k] - 2.0*self.phi_old[i,k])
                sum_b_i2 += self.phi[i,k]
            sum_a += sum_a_i
            sum_b += (sum_b_i1 * sum_b_i2)
        sum_complexity = sum_theta2 + sum_mu2 + funcComplexity(self.f)
        sum_c = np.sum(self.c_old)

        # ### test ###
        # print "llh in Q: ", sum_a/self.L - sum_b/self.K/self.L + sum_c/self.L
        # # print "sum_complexity in Q: ", sum_complexity

        Q = sum_a/self.L - sum_b/self.K/self.L + sum_complexity*self.lamda/2.0 + sum_c/self.L
        return Q

    def llhMinusLamdacheck(self, y):
        llh = 0.0
        for i in range(self.L):
            for k in range(self.K):
                llh += (y[i,k]*np.log(self.beta[i,k]))
        llh = llh / self.L
        sum_mu2 = 0.0
        sum_theta2 = 0.0
        for k in range(self.K):
            sum_mu2 += math.pow(self.mu[k], 2.0)
            for k_ in range(self.K):
                sum_theta2 += math.pow(self.theta[k,k_],2.0)
        sum_complexity = sum_theta2 + sum_mu2 + funcComplexity(self.f)
        ### test ###
        # print "llh in llh: ", llh
        # print "sum_complexity in llh: ", sum_complexity

        llh_minus_lamda = -llh + sum_complexity * self.lamda/2.0
        return llh_minus_lamda

    def update(self, x, y):
        # updating f #
        A = np.zeros(self.L, dtype=np.float64)
        B = np.zeros(self.L, dtype=np.float64)
        for i in range(self.L):
            sum_A1 = 0.0
            sum_AB2 = 0.0
            sum_B1 = 0.0
            for k in range(self.K):
                prod_k_i = np.inner(self.theta_old[k],x[i])
                sum_A1 += math.pow(prod_k_i, 2.0)
                sum_AB2 += prod_k_i
                sum_B1 += (2*self.mu_old[k] - 2*self.phi_old[i,k] + self.beta_old[i,k] - y[i,k]) * prod_k_i
            A[i] = sum_A1/self.L - math.pow(sum_AB2,2.0)/(self.L*self.K)
            B[i] = sum_B1/self.L + sum_AB2*(np.sum(self.phi_old[i])-np.sum(self.mu_old))*2.0/(self.L*self.K)
        self.f = discreteODE(A ,B, self.lamda, f0 = 0.0, f1 = 0.0)

        ### constant f ###
        if self.fCONSTANT:
            self.f = np.ones(self.L, dtype=np.float64)

        # updating mu #
        for k in range(self.K):
            sum_mu = 0.0
            for i in range(self.L):
                sum_mu += (y[i,k] - self.beta_old[i,k])
            self.mu[k] = (self.K*sum_mu + 2*self.L*(self.K-1)*self.mu_old[k]) / (2*self.L*(self.K-1) + self.L*self.K*self.lamda)
        # updating theta #
        for k in range(self.K):
            for k_ in range(self.K):
                sum_theta1 = 0.0
                sum_theta2 = 0.0
                sum_theta3 = 0.0
                for i in range(self.L):
                    sum_theta1 += (self.f[i]*x[i,k_]*(y[i,k] - self.beta_old[i,k]))
                    sum_theta2 += (math.pow(self.f[i],2.0)*math.pow(x[i,k_],2.0)*self.theta_old[k,k_])
                    sum_theta3 += (math.pow(self.f[i],2.0)*math.pow(x[i,k_],2.0))
                self.theta[k,k_] = (self.K*sum_theta1 + 2*(self.K-1)*sum_theta2) / (2*(self.K-1)*sum_theta3 + self.L*self.K*self.lamda)

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


def cumulate(y):
    """
    cumulating differential label sequence into distribution evolution
    :param y: np.ndarray([L,K]), one-hot vector for each row
    :return: np.ndarray([L,K])
    """
    L,K = y.shape
    x = np.zeros(L * K, dtype=np.float64).reshape([L, K])
    for i in range(1, L):
        x[i] = (y[i - 1] + x[i - 1] * (i - 1)) * 1.0 / i
    return x


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


def discreteODE(A, B, c, f0, f1):
    """
    solve discreteODE of form 2A[i]*f[i] + B[i] - c*f"[i]=0 | \forall i
    or minimize \sum A[i](f[i]^2) + \sum B[i]f[i] + c/2*\sum f'[i]^2
    :param f0: boundary condition
    :param f1: boundary condition
    """
    L = A.shape[0]
    f = np.zeros(L,dtype=np.float64)
    if c == 0.0:
        for i in range(1,L):
            if A[i] == 0.0:
                print "A", A
                raise ValueError("divided by 0")
        f[1:] = np.divide(B[1:],A[1:])/2.0
    else:
        # set boundary condition #
        f[0] = f0
        f[1] = f1
        # iterative #
        for i in range(2,L):
            f[i] = 2*(1.0+A[i-1]/c)*f[i-1] - f[i-2] + B[i]/c
    return f


def synthetic(L,mu,theta,f):
    K = mu.shape[0]
    # no herding #
    f = np.ones(L, dtype=np.float64)
    x = np.zeros([L,K], dtype=np.float64)
    y = np.zeros([L,K], dtype=np.float64)
    for i in range(L):
        beta = betacal(mu, f, theta, x, i)
        rnd = random.random()
        beta_sum = 0.0
        for k in range(K):
            beta_sum += beta[k]
            if rnd <= beta_sum:
                y[i,k] = 1.0
                break
        if np.sum(y[i])<1:
            print rnd, beta_sum
    return y


def synthetic2(L, mu, theta, f):
    random.seed(2037)
    print "mu"
    print mu
    print "theta"
    print theta
    K = mu.shape[0]
    x = np.zeros([L,K], dtype = np.float64)
    y = np.zeros([L,K], dtype = np.float64)
    for i in range(L):
        beta = betacal(mu, f, theta, x, i)
        rnd = random.random()
        beta_sum = 0.0
        for k in range(K):
            beta_sum += beta[k]
            if rnd <= beta_sum:
                y[i,k] = 1.0
                break
        if np.sum(y[i])<1:
            print rnd, beta_sum
        if i < L - 1:
            x[i+1] = (x[i]*i + y[i])/(i+1)
    return y


def betacal(mu, f, theta, x, i):
    phi = mu + f[i]*np.inner(theta,x[i])
    return np.exp(phi)/np.sum(np.exp(phi))


if __name__ == "__main__":
    L = 1000
    time_init = 200
    time_target = 900
    np.random.seed(2017)
    y = synthetic2(L, mu=np.array([1.0,1.0]), theta=np.array([[1.0,-1.0],[-1.0,1.0]]), f=np.ones(L))
    print np.sum(y, axis=0, dtype=np.float64)
    y_cumulate = cumulate(y)
    # for d in range(y.shape[1]):
    #     plt.plot(y_cumulate[:,d], label="%d" % d)
    # plt.legend()
    # plt.show()
    heard = Heard().fit(y, lamda=0.0, f_constant=True)
    heard.printmodel()
    # state_predicted = heard.predict(time_target)
    # print "init", y_cumulate[time_init]
    # print "predicted", state_predicted
    # print "true", y_cumulate[time_target]
