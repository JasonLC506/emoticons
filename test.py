import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools

def adding(additor):
    a = additor[0]
    b = additor[1]
    z = a+b
    return z
a =[np.nan, 1, 2]
b =np.array([a,[2,1,2],[1,2,1]])
c = np.nanmean(b,axis=0)
print c