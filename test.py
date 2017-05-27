import numpy as np
#from logRegFeatureEmotion import *
#from sklearn.model_selection import KFold
#import itertools
#import math
#from scipy.stats import kendalltau
from matplotlib import pyplot as plt
#from scipy.stats.mstats import gmean
#import DecisionTreeWeight_Bordar as dtb
#from readSushiData import readSushiData
import sys
import copy
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time

p = np.zeros([3,4,5,6])
x = np.arange(12).reshape([2,6])
p[1,1,1, :] = x[1, :]
print p[-2]