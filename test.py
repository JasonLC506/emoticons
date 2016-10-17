import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools

def adding(additor):
    a = additor[0]
    b = additor[1]
    z = a+b
    return z
a =[1 for i in range(3)]
u,v,w = a
print u,v,w,a