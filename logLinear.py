"""
Implementation based on Dekel, O., Manning, C.D. and Singer, Y., 2003, December. Log-Linear Models for Label Ranking. In NIPS (Vol. 16).
"""
import numpy as np

class logLinear(object):
    def __init__(self, K):
        # Model parameters #
        self.K = K  # # base functions
        self.lamda = None

        # intermediate parameters #
        self.p = None   # base function difference for each pair of each instance
        self.ro = 0     # max of all self.p
        self.wp = None
        self.wm = None
        self.loss = 0
        self.lamda_old = None   # keep track of lamda in previous iteration

    def fit(self, x, y):
        """
        input y: ranking vectors np.ndarray([Nsamp, Nclass])
        """
        