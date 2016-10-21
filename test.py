import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold
import itertools


a =[1,2,2,3,4,5,3,5]
a = np.array(a)
print np.argmax(a)