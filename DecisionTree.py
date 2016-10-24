import numpy as np
import logRegFeatureEmotion as LogR

MINNODE = 1

def divideset(x,samples, feature, value):
    # divide samples (index) into two sets according to split value

    if type(x) != np.ndarray:
        x= np.array(x)
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
            gini_rank += gini * n # adding weight of size
        # ### test ###
        # print "n_class: ", n_class
        # print "gini: ", gini
    return gini_rank


# def rankResult(y,samples):
#     if type(y) != np.ndarray:
#         y = np.array(y)
#     Ranks = y.shape[1]
#     N_class = Ranks
#     result = []
#     n_class =[[] for i in range(N_class)]
#     for rank in range(N_class):
#         n_class[rank] = [0 for i in range(N_class)]
#         for sample in samples:
#             emoti = y[sample,rank]
#             if emoti>=0:
#                 n_class[rank][emoti]+=1
#         # assign ranking #
#         if sum(n_class[rank])==0:
#             result.append(-1)
#             continue
#         flag = False
#         while flag == False:
#             max_value = max(n_class[rank])
#             for i in range(N_class):
#                 if n_class[rank][i] == max_value:
#                     if i not in result:
#                         result.append(i)
#                         flag = True
#                     else:
#                         n_class[rank][i] = 0
#     return result

def rankResult(y,samples):
    if type(y)!=np.ndarray:
        y = np.array(y)
    Ranks = y.shape[1]
    N_class = Ranks
    result = []
    n_class =[[] for i in range(N_class)]
    for rank in range(N_class):
        n_class[rank] = [0 for i in range(N_class)]
        for sample in samples:
            emoti = y[sample,rank]
            if emoti>=0:
                n_class[rank][emoti]+=1
    for rank in range(N_class):
        # assign ranking #
        # current rank with highest priority, when zero appearance, from highest rank to lowest #
        priority = [i for i in range(N_class) if i != rank]
        priority.insert(0,rank)
        flag = False
        for i in priority:
            n_class_cur = n_class[i]
            while not flag:
                max_value = max(n_class_cur)
                if max_value < 1:
                    break # all are 0 in current rank
                emoti = n_class_cur.index(max_value)
                if emoti not in result:
                    result.append(emoti)
                    flag = True # find best emoticon
                else:
                    n_class_cur[emoti]=0
            if flag:
                break
        if not flag:
            result.append(-1)
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
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.array(y)
    if type(samples) != np.ndarray:
        samples = np.array(samples)
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
        feature = N_feature - 1 - feature ### test

        values ={}
        for sample in samples:
            values[x[sample,feature]]=1
        for value in values.keys():
            samps1, samps2 = divideset(x,samples,feature,value)

            ## gain by split
            # giniRank already include size of node weight
            gain = current_criterion - (criterion(y,samps1)+criterion(y,samps2))
            if gain > best_gain and len(samps1)*len(samps2)>0:
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

def printtree(tree,indent=""):
    if tree.result != None:
        print(indent+str(tree.result))
    else:
        print(indent+str(tree.feature)+">="+str(tree.value)+"?")
        print(indent+"T->\n")
        printtree(tree.tb,indent+"  ")
        print(indent + "F->\n")
        printtree(tree.fb,indent+"  ")

def predict(observation,tree):
    # prediction of single observation
    if tree.result!=None:
        return tree.result
    value = observation[tree.feature]
    branch = None
    if isinstance(value,int) or isinstance(value,float):
        if value>=tree.value:
            branch = tree.tb
        else:
            branch = tree.fb
    else:
        raise("nominal feature not supported")
    return predict(observation,branch)

def dataSimulated(Nsamp, Nfeature, Nclass):
    np.random.seed(seed=10)
    x = np.arange(Nsamp*Nfeature,dtype="float").reshape([Nsamp,Nfeature])
    x += np.random.random(x.shape)*10
    y = np.random.random(Nsamp*Nclass).reshape([Nsamp, Nclass])
    y *= 2
    y = y.astype(int)
    y = map(LogR.rankOrder,y)
    return x,y


if __name__ == "__main__":
    ### test ###
    x,y = dataSimulated(Nsamp=6,Nfeature=5,Nclass=6)
    print x
    print y
    samples = [i for i in range(x.shape[0])]
    tree = buildtree(x,y,samples)
    printtree(tree)

    # print rankResult(y,samples)

    # set1,set2 = divideset(x,samples,2,24)
    # print set1, set2
