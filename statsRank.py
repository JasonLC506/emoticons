from stats import imbalanceMeasure
import math
from readSushiData import readSushiData

def pairwise(y):
    Nclass = y.shape[1]
    pair = [[[0,0] for i in range(Nclass)] for j in range(Nclass)]
    y = y.tolist()
    for post in y:
        for pos_i in range(Nclass-1):
            for pos_j in range(pos_i+1, Nclass):
                pair[post[pos_i]][post[pos_j]][0] += 1
                pair[post[pos_j]][post[pos_i]][1] += 1
    n_pair = 0
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            n_pair += sum(pair[i][j])
    print "number of ordered pairs: ", n_pair
    for i in range(Nclass):
        print "\t".join(map(str,pair[i]))
    return pair

if __name__ == "__main__":
    x,y= readSushiData()
    print "# instance: ", y.shape[0]
    pair = pairwise(y)
    imbalanceMeasure(pair)
