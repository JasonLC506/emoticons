import numpy as np
import logRegFeatureEmotion

MINNODE = 1

def divideset(x,y,samples, feature, value):
    # divide samples (index) into two sets according to split value

    if type(x) != np.ndarray:
        x= np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples)!= np.ndarray:
        samples = np.array(samples)
    split_function = None
    if isinstance(value,int) or isinstance(value,float):
        # numerical feature
        split_function = lambda sample: x[sample,feature]>=value
    else:
        raise("nominal feature not supported")

    index = split_function(samples)
    set1 = samples[index]
    set2 = samples[index==False]
    return set1, set2

def giniRank(y,samples):
    # y should be ranking data here
    # calculate it for every split node, which is O(n^2), can do better in O(nlogn)
    if type(y) != np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    Nclass = Ranks
    gini_rank = 0.0
    for rank in range(Ranks):
        n_class = [0.0 for i in range(Nclass)]
        gini = 0.0
        for sample in samples:
            emoti = y[sample,rank]
            if emoti>=0:
                n_class[emoti]+=1
        n = sum(n_class)
        if n < 1:
            gini_rank += gini*n
        else:
            gini = sum([n_class[i]*(n-n_class[i]) for i in range(Nclass)])*1.0/n/n
            gini_rank += gini * n
    return gini_rank

def rankResult(y,samples):
    if type(y) != np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    N_class = Ranks
    result = []
    for rank in range(N_class):
        n_class = [0 for i in range(N_class)]
        for sample in samples:
            emoti = y[sample,rank]
            if emoti>=0:
                n_class[emoti]+=1
        max_value = max(n_class)
        for i in range(N_class):
            if n_class[i] == max_value and i not in result:
                result.append(i)
    return result
class decisionnode:
    def __init__(self,feature=-1,value=None, result = None, tb=None, fb=None):
        self.feature = feature
        self.value = value
        self.result = result
        self.tb = tb
        self.fb = fb

def buildtree(x,y, samples, criterion = giniRank, min_node=1):
    if type(x) != np.ndarray:
        y = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples) != np.ndarray:
        y = np.array(samples)
    if len(samples) == 0:
        return decisionnode()
    current_criterion = criterion(y,samples)

    if len(samples)<= min_node:
        return decisionnode(result=rankResult(y,samples))
    # find best split
    best_gain = 0.0
    best_split = []
    best_sets = []

    N_feature = x.shape[1]

    for feature in range(N_feature):
        values ={}
        for sample in samples:
            values[x[sample,feature]]=1
        for value in values.keys():
            samps1, samps2 = divideset(x,y,samples,feature,value)

            ## gain by split
            # giniRank already include size of node weight
            gain = criterion(y,samps1)+criterion(y,samps2)-current_criterion
            if gain>best_gain and len(samps1)*len(samps2)>0:
                best_gain = gain
                best_split = [feature,value]
                best_sets = [samps1, samps2]
    if best_gain>0:
        tb = buildtree(x,y, best_sets[0], min_node = min_node)
        fb = buildtree(x,y, best_sets[1], min_node = min_node)
        return decisionnode(feature=best_split[0], value = best_split[1],
                            tb = tb, fb = fb)
    else:
        return decisionnode(result = rankResult(y,samples))
