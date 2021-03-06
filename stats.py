from logRegFeatureEmotion import dataClean
from logRegFeatureEmotion import emoticon_list
from scipy.stats import pearsonr
import numpy as np
import math
from queryAlchemy import emotion_list
from logRegFeatureEmotion import rankOrder
from readSyntheticData import readSyntheticData
from SMPrank import SmpRank

def stats(y):
    total = y.shape[0]
    Nclass = len(emoticon_list)
    like_only = 0
    emoticons_total = [0 for i in range(Nclass)]
    emoticon_sig = [0 for i in range(Nclass)] ## second large when led by "like" or first large
    emoticon_exist = [0 for i in range(Nclass)]
    like_total = 0
    for post in y:
        if np.sum(post[1:])==0:
            like_only += 1
            like_total += post[0]
        else:
            for i in range(len(emoticon_list)):
                emoticons_total[i] += post[i]
        for i in range(Nclass):
            if post[i]>=1:
                emoticon_exist[i]+=1
        rank = rankOrder(post)
        if rank[0]==0:
            if rank[1]>=0:
                emoticon_sig[rank[1]]+=1
        else:
            if rank[0]>0:
                emoticon_sig[rank[0]]+=1
    print "total posts: ", total
    print "posts with only like: ", like_only
    print "                                  ", emoticon_list
    print "emoticons in multiemoticon posts: ", emoticons_total
    print "posts with certain emoticon:      ", emoticon_exist
    print "posts with significant emoticon:  ", emoticon_sig
    print "total likes in only like posts: ", like_total


def pairwise(y):
    Nclass = len(emoticon_list)
    # first and second dimension is the indices of emoticon pairs, the first value is # posts first emoticon rank higher
    # than the second, vise versa
    paircomp = [[[0,0] for i in range(Nclass)] for j in range(Nclass)] # take not appearing emoticons to rank the lowest
    paircomp_sub = [[[0,0] for i in range(Nclass)] for j in range(Nclass)] # excluding not appearing emoticons
    y = y.tolist()
    for post in y:
        for i in range(Nclass):
            for j in range(i+1, Nclass):
                if post[i] < 1:
                    if post[j] >= 1:
                        paircomp[i][j][1] += 1
                    # else both zero, no comparison
                else: # emoticon i exists
                    if post[j] < 1:
                        paircomp[i][j][0] += 1
                    else: # both exist
                        if post[i] > post[j]:
                            paircomp[i][j][0] += 1
                            paircomp_sub[i][j][0] += 1
                        elif post[i] < post[j]:
                            paircomp[i][j][1] += 1
                            paircomp_sub[i][j][1] += 1
    for i in range(1,Nclass):
        for j in range(i):
            paircomp[i][j][0] = paircomp[j][i][1]
            paircomp[i][j][1] = paircomp[j][i][0]
            paircomp_sub[i][j][0] = paircomp_sub[j][i][1]
            paircomp_sub[i][j][1] = paircomp_sub[j][i][0]
    n_pair = 0
    n_pair_sub = 0
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            n_pair += sum(paircomp[i][j])
            n_pair_sub += sum(paircomp_sub[i][j])
    print "complete paircompare:", "(total %d)" % n_pair
    print "emoticon ", "\t".join(emoticon_list)
    for i in range(Nclass):
        print emoticon_list[i], "\t".join(map(str, paircomp[i]))
    print "appearing paircompare:", "(total %d)" % n_pair_sub
    print "emoticon ", "\t".join(emoticon_list)
    for i in range(Nclass):
        print emoticon_list[i], "\t".join(map(str, paircomp_sub[i]))
    return paircomp, paircomp_sub


def imbalanceMeasure(paircomp):
    Nclass = len(paircomp)
    imba = 0
    cnt_pair = 0
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            imba += abs(math.log(paircomp[i][j][0]+1)-math.log(paircomp[i][j][1]+1))
            cnt_pair += 1
    print "imba: ", imba/cnt_pair
    return imba/cnt_pair


def imbalanceMeasureNolike(paircomp):
    Nclass = len(paircomp)
    imba = 0
    cnt_pair = 0
    for i in range(1,Nclass):
        for j in range(i+1, Nclass):
            imba += abs(math.log(paircomp[i][j][0]+1)-math.log(paircomp[i][j][1]+1))
            cnt_pair += 1
    print "imba without like: ", imba/cnt_pair
    return imba/cnt_pair


def statsAnal(x,y):
    N_feature = x.shape[1]
    N_class = y.shape[1]
    y_total = np.sum(y,axis=1,keepdims=True)
    y = y*1.0/y_total
    # correlation in whole set #
    ps_whole = [[0 for i in range(N_feature)] for j in range(N_class)]
    for f in range(N_feature):
        for c in range(N_class):
            ps_whole[c][f] = pearsonr(x[:,f],y[:,c])
    # correlation in multiemoticon set #
    ps_multi = [[0 for i in range(N_feature)] for j in range(N_class)]
    emoticons = np.sum(y[:,1:],axis=1)
    multi_e = emoticons>0
    for f in range(N_feature):
        for c in range(N_class):
            ps_multi[c][f] = pearsonr(x[multi_e,f],y[multi_e,c])

    for c in range(N_class):
        print "emoticon: %s" % emoticon_list[c]
        print "emotion: ", emotion_list
        print "pearson in whole set: ", ps_whole[c]
        print "pearson in multi set: ", ps_multi[c]

    return ps_whole, ps_multi


def statsRank(x, y):
    """
    stats of ranking input
    """
    results = {}
    results["Dfeature"] = x.shape[1]
    results["Nsamp"] = x.shape[0]
    results["Nclass"] = y.shape[1]
    smp = SmpRank(K=1)
    y_pref = np.array(map(lambda z: smp.rank2pair(np.array(z)), y.tolist()))
    paircomp = np.sum(y_pref, axis = 0)
    results["IMBA"] = imbalanceMeasureRank(paircomp)
    return results


def imbalanceMeasureRank(paircomp):
    Nclass = len(paircomp)
    imba = 0
    cnt_pair = 0
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            imba += abs(math.log(paircomp[i][j]+1)-math.log(paircomp[j][i]+1))
            cnt_pair += 1
    print "imba: ", imba/cnt_pair
    return imba/cnt_pair



if __name__ == "__main__":
    # x, y = dataClean("data/foxnews_Feature_linkemotion_deduplic.txt")
    # stats(y)
    # statsAnal(x,y)
    # paircomp, paircomp_sub = pairwise(y)
    # imbalanceMeasure(paircomp)
    # imbalanceMeasureNolike(paircomp)
    dataset_list = ["authorship", "bodyfat", "calhousing", "cpu", "elevators", "fried", "glass", "housing", "iris", "pendigits", "segment", "stock", "vehicle", "vowel", "wine", "wisconsin"]
    #dataset = "bodyfat"
    for dataset in dataset_list:
        x, y = readSyntheticData("data/synthetic/" + dataset )
        result = statsRank(x, y)
        with open("results/stats_synthetic", "a") as f:
            f.write(dataset + "\n")
            f.write(str(result) + "\n")
