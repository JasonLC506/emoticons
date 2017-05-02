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

a = np.ones([2,4])
b = np.ones([10,4])
print np.inner(a,b).shape