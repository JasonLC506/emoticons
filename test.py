import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold

a = np.array([[1,2],[2,3],[3,4]])
print np.mean(a,axis=0)
print np.std(a,axis=0)
