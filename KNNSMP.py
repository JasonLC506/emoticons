from KNNPlackettLuce import KNN
import numpy as np
from SMPrank import SmpRank

class KNNSMP(KNN):
    def aggregate(self, neighbors):
        y_train = self.y[neighbors,:]
        x_train = np.ones([neighbors.shape[0],2], dtype=np.float64)
        return SmpRank(K=1).fit(x_train, y_train).predict(x_train[0])

    def probcal(self, x_test, y_test):
        neighbors = self.neighborhood(x_test)
        y_train = self.y[neighbors,:]
        x_train = np.ones([neighbors.shape[0],2], dtype=np.float64)
        smp = SmpRank(K=1).fit(x_train, y_train)
        smp.probcal(x_train[0], smp.rank2pair(y_test))
        prob = smp.probsum
        return prob