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

ctime = time.strptime("2017-05-02T15:28:00+0000", "%Y-%m-%dT%H:%M:%S+0000")
ntime = time.strptime("2015-05-02T15:28:00+0000", "%Y-%m-%dT%H:%M:%S+0000")
print ntime > ctime
