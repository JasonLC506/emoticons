import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools
import math


NCLASS = 6
NRANK = 6

def nConstraints():
    N_per = math.factorial(NCLASS)/math.factorial(NCLASS-NRANK)
    N_constr = [0 for i in range(N_per)]
    rank_per = rankPer()

def rankPer():
    N_per = math.factorial(NCLASS) / math.factorial(NCLASS - NRANK)
    rank_per = [[] for i in range(N_per)]
    


