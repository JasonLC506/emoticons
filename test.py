import numpy as np
from logRegFeatureEmotion import *
from sklearn.model_selection import KFold

a = np.array([[1,2],[2,3],[3,4]])
b = a[:,0]
print b.shape

print b[2,:]