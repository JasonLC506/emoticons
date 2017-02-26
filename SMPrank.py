"""
Jason Zhang 02/25/2017 jpz5181@psu.edu
follow Grbovic et al. 2013 IJCAI [1]
"""
import numpy as np
import warnings


class SMPrank(object):
    def __int__(self, K):
        ## intrinsic parameters ##
        self.mfeature = [[] for c in range(K)]
        self.mpairlabel = [[] for c in range(K)]
        self.varfeature = np.nan
        self.varpairlabel = np.nan
        self.K = K
        self.d = np.nan
        self.L = np.nan

        ## intermediate parameters ##
        self.probfeature = [[] for c in range(K)]
        self.probpairlabel = [[] for c in range(K)]
        self.probsum = []
        self.losssmp = np.nan

        ## hyperparameter for SGD ##
        self.decayvarf = np.nan
        self.initvarf = np.nan
        self.decaylearningrate = np.nan
        self.initlearningrate = 0.04 # from [1] empirical #

    def fit(self, x_train, y_train, x_valid = None, y_valid = None, Max_epoch = 10):
        """

        :param x_train: N*d np.array
        :param y_train: pairwise comparison matrix n*L*L np.array
        :return: self
        """
        if x_valid is None:
            N0 = x_train.shape[0]
            samps = np.arange(N0)
            np.random.shuffle(samps)
            N_val = N0/10
            x_valid = x_train[samps[:N_val],:]
            y_valid = y_train[samps[:N_val],:,:]
            x_train = x_train[samps[N_val:],:]
            y_train = y_train[samps[N_val:],:,:]

        N = x_train.shape[0]
        d = x_train.shape[1]
        L = y_train.shape[1]

        self.initialize(N,d,L, x_train, y_train)

        ## SGD ##
        loss_valid = np.nan
        for epoch in range(Max_epoch):
            samps = np.arange(N)
            np.random.shuffle(samps)
            for t in range(N):
                # complete training set #
                samp = samps[t]
                self.update(x_train[samp], y_train[samp], t = (epoch * N + t))

            loss_valid_now = self.losscomp(x_valid, y_valid)
            loss_train_now = self.losscomp(x_train, y_train)
            print "loss_valid", epoch, loss_valid_now
            print "loss_train", epoch, loss_train_now
            # for SGD correctness test #
            if not np.isnan(self.losssmp) and self.losssmp < loss_train_now:
                warnings.warn("training set loss increase")
                print "last epoch loss: ", self.losssmp
                print "current epoch loss: ", loss_train_now
            self.losssmp = loss_train_now
            # stop criterion #
            if not np.isnan(loss_valid) and loss_valid < loss_valid_now:
                print "early stop at the end of epoch: ", epoch
                return self
        return self

    def predict(self, x_test):
        pass

    def initialize(self, N, d, L, x_train, y_train):
        ## initialize parameters according [1] ##
        self.decayvarf = N*5.0
        self.initvarf = np.mean(np.linalg.norm(
            x_train - np.mean(x_train, axis = 0),
            axis = 1
        ))
        self.varfeature = self.initvarf
        self.varpairlabel = np.mean(np.linalg.norm(
            y_train - np.mean(y_train, axis = 0),
            axis = (1,2)
        ))
        self.decaylearningrate = N*5.0
        self.d = d
        self.L = L

        cf = 0
        cp = 0
        for samp in range(N):
            x_samp = x_train[samp]
            y_samp = y_train[samp]
            if x_samp not in self.mfeature and cf < self.K:
                self.mfeature[cf] = x_samp
                cf += 1
            if y_samp not in self.mpairlabel and cp < self.K:
                self.mpairlabel[cp] = y_samp
                cp += 1
            if cf >= self.K and cp >= self.K:
                break
        if cf < self.K or cp < self.K:
            print "distinct features: ", cf, "distinct pairlabel: ", cp
            raise("too many prototypes")
