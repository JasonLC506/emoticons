import numpy as np

class poisson(object):
    def __init__(self):
        self.probs = []

    def fit(self, state_init, time_init):
        state_init_abs = state_init * time_init
        K = state_init.shape[0]
        for k in range(K):
            state_init_abs[k] += 1.0
        total = np.sum(state_init_abs)
        self.probs = state_init_abs / total
        return self

    def predict(self):
        return self.probs