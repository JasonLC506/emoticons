import itertools
import math
from matplotlib import pyplot as plt
# from matplotlib import rc
import numpy as np

# rc("text", usetex=True)
# plt.rc("text", usetex=True)
# plt.rc('font', family='serif')

def inversions(rank, baserank):
    Nclass = len(rank)
    invs = 0
    for i in range(Nclass):
        for j in range(i+1, Nclass):
            lp = baserank[i]
            ll = baserank[j]
            if not (lp in rank and ll in rank):
                raise("not supporting incomplete ranking")
            p = rank.index(lp)
            l = rank.index(ll)
            if l>p:
                continue
            else:
                invs += 1
    return invs


def mallowsInnormal(invs, theta):
    return math.exp(-invs*theta)


def EGini(rankprob):
    Nclass = len(rankprob[0][0])
    Gini = [0.0 for i in range(Nclass)]
    for i in range(Nclass):
        ## position i ##
        prob = [0.0 for _ in range(Nclass)] # complete ranking
        for j in range(len(rankprob)):
            prob[rankprob[j][0][i]] += rankprob[j][1]
        Gini[i] = 1-sum([pow(prob[m],2) for m in range(Nclass)])
    return sum(Gini), Gini


plt.figure(1)
theta_list = np.arange(0.0, 10.0, 0.1)
theta_list = theta_list.tolist()
for N in range(3,7):
    EGini_list = [0.0 for i in range(len(theta_list))]
    EGini_pos_list = [[0.0 for i in range(len(theta_list))] for pos in range(N)]
    for i in range(len(theta_list)):
        theta = theta_list[i]
        baserank = [j for j in range(N)]
        rankprob = []
        probsum = 0.0
        for rank in itertools.permutations(baserank):
            prob = mallowsInnormal(inversions(rank, baserank), theta)
            rankprob.append([rank, prob])
            probsum += prob
        for k in range(len(rankprob)):
            rankprob[k][1]/= probsum
            # print rankprob[i]
        eGini,eGini_pos = EGini(rankprob)
        print eGini,eGini_pos
        EGini_list[i] = eGini
        # for pos in range(N):
            # EGini_pos_list[pos][i]=eGini_pos[pos]
    ### test ###
    # EGini_list=map(lambda x: x/(N-1), EGini_list)

    # for pos in range(N):
        # plt.subplot(N+1,1,pos+1)
        # plt.plot(theta_list, EGini_pos_list[pos])
    plt.plot(theta_list, EGini_list, label="d="+str(N))
plt.ylabel("E(Gini)")
plt.xlabel(r"$\theta$")
plt.legend()
plt.show()
plt.savefig("Mallows2Gini.svg", format="svg")
