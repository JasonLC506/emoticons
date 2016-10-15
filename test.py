import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold

a = np.array([[1,2],[2,3],[3,4]])
c = np.sum(a,axis=1,keepdims=True)
print a*1.0/c

print a>3