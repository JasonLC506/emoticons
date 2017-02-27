"""
Jason Zhang 02/25/2017 jpz5181@psu.edu
follow Grbovic et al. 2013 IJCAI [1]
"""
import numpy as np
import warnings


class SMPrank(object):
    def __int__(self, K):
        ## intrinsic parameters ##
        self.mfeature = [np.nan for c in range(K)]
        self.mpairlabel = [np.nan for c in range(K)]
        self.varfeature = None
        self.varpairlabel = None
        self.K = K
        self.d = None
        self.L = None

        ## intermediate parameters ##
        self.probfeature = [0.0 for c in range(K)]
        self.probpairlabel = [0.0 for c in range(K)]
        self.probsum = 0.0

        ## loss_SMP for previous training set ##
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

            # real time result test #
            loss_valid_now = self.losscal(x_valid, y_valid)
            loss_train_now = self.losscal(x_train, y_train)
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
        """
        calculate predicted RANKING for input feature values
        :param x_test:
        :return: y_pred n*L ranking vectors
        """
        ## batch predict ##
        if x_test.ndim != 1:
            N = x_test.shape[0]
            y_rank_pred = [[] for i in range(N)]
            for samp in range(N):
                y_rank_pred[samp] = self.predict(x_test[samp])
            return np.array(y_rank_pred, dtype = np.int8)

        ## weighted average of preference matrices ##
        self.probcal(x_test)
        y_pred = np.zeros([self.L, self.L], dtype = np.float16)
        for k in range(self.K):
            y_pred = y_pred + self.probfeature[k] * self.mpairlabel[k]

        ## aggregate preference to rank ##
        y_rank_pred = self.aggregate(y_pred)

        return y_rank_pred

    def aggregate(self, y):
        """
        transfer preference matrix to ranking
        using TOUGH from [1]
        :param y: L*L np.ndarray
        :return: L ranking
        """
        y_rank = []
        labels = [label for label in range(self.L)]
        while len(labels)>0:
            ind_max = np.argmax(y)
            prior, latter = ind_max / self.L, ind_max % self.L
            # for test #
            if prior not in labels:
                print y
                raise("cannot find maximum in remaining labels")
            for pos in range(len(y_rank)+1):
                rank_temp = [l for l in y_rank]
                rank_temp.insert(pos, prior)
                pass

    def initialize(self, N, d, L, x_train, y_train):
        ## initialize parameters according [1] ##
        self.decayvarf = N*5.0
        self.initvarf = float(np.mean(np.linalg.norm(
            x_train - np.mean(x_train, axis = 0),
            axis = 1
        )))
        self.varfeature = self.initvarf
        self.varpairlabel = float(np.mean(np.linalg.norm(
            y_train - np.mean(y_train, axis = 0),
            axis = (1,2)
        )))
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
                self.mpairlabel[cp] = self.complete(y_samp)
                cp += 1
            if cf >= self.K and cp >= self.K:
                break
        if cf < self.K or cp < self.K:
            print "distinct features: ", cf, "distinct pairlabel: ", cp
            raise("too many prototypes")

    def complete(self, y):
        for i in range(self.L):
            for j in range(i+1, self.L):
                if y[i,j] == 0 and y[j, i] == 0:
                    y[i,j] = 0.5
                    y[j,i] = 0.5
        return y

    def update(self, x, y, t):
        ## update decaying parameters ##
        self.varfeature = self.initvarf * (self.decayvarf * 1.0 / (self.decayvarf + t))

        # calculate intermediate parameters #
        self.probcal(x, y)

        ## update ##
        learningrate = self.initlearningrate * \
                       (self.decaylearningrate * 1.0 / (self.decaylearningrate +t))
        for k in range(self.K):
            self.mfeature[k] = self.mfeature[k] - ((learningrate * (self.probsum - self.probpairlabel[k]) *
                                              self.probfeature[k] / (self.probsum * pow(self.varfeature, 2))) *
                                             (x - self.mfeature[k]))
            self.mpairlabel[k] = self.mpairlabel[k] + ((learningrate * (self.probpairlabel[k] * self.probfeature[k]) /
                                                  (self.probsum * pow(self.varpairlabel, 2))) *
                                                 (y - self.mpairlabel[k]))

    def losscal(self, xs, ys):
        """
        calculate losssmp with current parameters value and given xs, ys set
        :param xs: n*d np.ndarray
        :param ys: n*L*L np.ndarray
        :return: losssmp
        """
        losssmp = 0.0
        N = xs.shape[0]
        for samp in range(N):
            self.probcal(xs[samp], ys[samp])
            losssmp += (-np.log(self.probsum)/N)
        return losssmp

    def probcal(self, x, y = None):
        sum_probf = 0.0
        for k in range(self.K):
            self.probfeature[k] = frobeniusGaussianCore(x, self.mfeature[k], self.varfeature)
            sum_probf += self.probfeature[k]
            if y is None:
                continue
            self.probpairlabel[k] = frobeniusGaussianCore(y, self.mpairlabel[k], self.varpairlabel)
        if y is None:
            return
        self.probsum = 0.0
        for k in range(self.K):
            self.probfeature[k] = self.probfeature[k] / sum_probf
            self.probsum += self.probfeature[k] * self.probpairlabel[k]


def frobeniusGaussianCore(x, m, sigma):
    return np.exp(-pow(np.linalg.norm(x - m)/sigma, 2))