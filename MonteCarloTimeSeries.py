"""
multivariate time series extropolate via Monte Carlo simulation
default Dashun KDD'14 HEARD model [1]
"""
import numpy as np

class MonteCarloTimeSeries(object):
    def __init__(self, state_init, mu, theta, f, Nsamp = 26610):
        ## model dependent parameter ##
        self.state_init = state_init
        self.para_mu = mu
        self.para_theta = theta
        self.para_f = f
        self.para_K = state_init.shape[0]

        ## MC dependent parameter ##
        self.time_init = 0 # the time stamp of initial state
        self.time_target = 0
        self.SEED = 2017
        self.Nsamp = Nsamp # default as in [1]

    def transition(self, state_cur, time_cur):
        ## model dependent ##
        phi = self.para_mu + self.para_f[time_cur]*np.inner(self.para_theta, state_cur)
        phi_exp = np.exp(phi)
        beta = phi_exp / np.sum(phi_exp)
        return beta