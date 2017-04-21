"""
multivariate time series extropolate via Monte Carlo simulation
default Dashun KDD'14 HEARD model [1]
"""
import numpy as np
from Heard import *
from datetime import datetime
from datetime import timedelta

class MonteCarloTimeSeries(object):
    def __init__(self, state_init, time_init, mu, theta, f, Nsamp = 26610, seed = 2017):
        ## model dependent parameter ##
        self.state_init = state_init
        self.para_mu = mu
        self.para_theta = theta
        self.para_f = f
        self.para_K = state_init.shape[0]

        ## MC dependent parameter ##
        self.time_init = time_init # the time stamp of initial state
        self.time_target = 0
        self.SEED = seed
        self.Nsamp = Nsamp # default as in [1]

    def predict(self, time_target):
        # init random seed #
        np.random.seed(self.SEED)

        self.time_target = time_target
        state_targets = []
        for samp in range(self.Nsamp):
            # start = datetime.now()
            state_target = self.predict_single(time_target)
            # print "round ", samp, "takes ", (datetime.now()-start).total_seconds(), "with state", state_target
            state_targets.append(state_target)
        state_targets = np.array(state_targets, dtype=np.float64)
        return np.mean(state_targets, axis=0)

    def predict_single(self, time_target):
        time_cur = self.time_init
        state_cur = self.state_init
        while time_cur < time_target:
            beta = self.transition(state_cur=state_cur, time_cur=time_cur)
            step = self.stepSample(beta)
            state_next = self.stateUpdate(state_cur, time_cur, step)
            state_cur = state_next
            time_cur += 1
        return state_cur

    def transition(self, state_cur, time_cur):
        ## model dependent ##
        phi = self.para_mu + self.para_f[time_cur]*np.inner(self.para_theta, state_cur)
        phi_exp = np.exp(phi)
        beta = phi_exp / np.sum(phi_exp)
        return beta

    def stepSample(self, prob):
        prob_sum = 0.0
        rnd = np.random.random()
        step = np.zeros(self.para_K, dtype=np.float64)
        for k in range(self.para_K):
            prob_sum += prob[k]
            if prob_sum >= rnd:
                step[k] += 1
                return step
        print prob
        raise ValueError("wrong prob distribution")

    def stateUpdate(self, state_cur, time_cur, step):
        ## model dependent ##
        state_next = (state_cur * time_cur + step) / (time_cur + 1)
        return state_next

if __name__ == "__main__":
    L = 1000
    y = synthetic2(L, mu=np.array([1.0,1.0]), theta=np.array([[1.0,-1.0],[-1.0,1.0]]), f=np.zeros(L))
    print np.sum(y, axis=0, dtype=np.float64)
    y_cumulate = cumulate(y)
    for d in range(y.shape[1]):
        plt.plot(y_cumulate[:,d], label="%d" % d)
    plt.legend()
    plt.show()
    time_init = 200
    time_target = 500
    heard = Heard().fit(y[:time_init,:], lamda=5.0, f_constant=True)
    heard.printmodel()
    start = datetime.now()
    MC = MonteCarloTimeSeries(state_init = y_cumulate[time_init,:], time_init = time_init, mu = heard.mu, theta= heard.theta, f = np.ones(L), Nsamp=2000)
    state_predicted = MC.predict(time_target)
    print "it takes ", (datetime.now() - start).total_seconds()
    print "init", y_cumulate[time_init]
    print "predicted", state_predicted
    print "true", y_cumulate[time_target]