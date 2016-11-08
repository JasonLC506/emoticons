import logRegFeatureEmotion as LogR
import numpy as np
from queryAlchemy import emotion_list
from logRegFeatureEmotion import emoticon_list

def score2pair(x,y, k = 2, Abstention = False):
    """
    k is the enhance factor for preference pairs over no-prefer pairs
    the bigger k is, the smaller the effect of no-prefer pairs,
    kind of inverse of laplace smooth
    """
    # consider not appearing emoticons as ranked lowest #
    Nsamp = y.shape[0]
    Nclass = y.shape[1]
    # no-prefer pairs are included #
    y_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
    x_enlarge = [[[] for j in range(Nclass)] for i in range(Nclass)]
    # for case no-prefer pairs are excluded#
    x_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
    y_list = [[[] for j in range(Nclass)] for i in range(Nclass)]
    # i,j value = 1 if emoticon i ranks higher than j, 0 otherwise #
    for samp in range(Nsamp):
        for i in range(Nclass):
            for j in range(i+1, Nclass):
                if y[samp,i] > y[samp,j]:
                    for _ in range(k):
                        y_enlarge[i][j].append(1)
                        x_enlarge[i][j].append(x[samp])
                    y_list[i][j].append(1)
                    x_list[i][j].append(x[samp])
                elif y[samp,i] < y[samp, j]:
                    for _ in range(k):
                        y_enlarge[i][j].append(0)
                        x_enlarge[i][j].append(x[samp])
                    y_list[i][j].append(0)
                    x_list[i][j].append(x[samp])
                else: # y[samp,i] == y[samp,j]
                    for value in [0,1]:
                        y_enlarge[i][j].append(value)
                        x_enlarge[i][j].append(x[samp])
    if Abstention:
        # include no-preference pair #
        return x_enlarge, y_enlarge
    else:
        # exclude no-preference pair #
        return x_list, y_list


if __name__ == "__main__":
    x = np.array([[1,2,3],[3,2,1]])
    y = np.array([[10,8,6,5,0,0],[7,9,0,0,0,0]])
