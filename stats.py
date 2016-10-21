from logRegFeatureEmotion import dataClean
from logRegFeatureEmotion import emoticon_list
from scipy.stats import pearsonr
import numpy as np
from queryAlchemy import emotion_list
from logRegFeatureEmotion import rankOrder

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


if __name__ == "__main__":
    x, y = dataClean("data/posts_Feature_Emotion.txt")
    stats(y)
    statsAnal(x,y)