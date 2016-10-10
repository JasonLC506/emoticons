import sklearn
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
from queryAlchemy import emotion_list as EL
import numpy as np
import ast
import math

emoticon_list = ["Like","Love","Sad","Wow","Haha","Angry"]
NOISE = 0.001

def dataClean(datafile):
    file = open(datafile,"r")
    x=[]
    y=[]
    for item in file.readlines():
        try:
            sample = ast.literal_eval(item.rstrip())
        except SyntaxError, e:
            print e.message
            continue
        if sample["feature_emotion"][0]<0:
            continue
        x.append(sample["feature_emotion"])
        emoticons = [0 for i in range(len(emoticon_list))]
        for j in range(len(emoticon_list)):
            if emoticon_list[j] in sample["emoticons"].keys():
                emoticons[j] = sample["emoticons"][emoticon_list[j]]
            else:
                emoticons[j] = 0
        y.append(emoticons)
    file.close()
    x = np.asarray(x,dtype="float")
    y = np.asarray(y,dtype="float")
    return x,y

def trainTest(x,y,cv=10):
    ###  error measure
    performance =[]
    coefi=[]
    intercept=[]
    # cross validation #
    kf = KFold(n_splits = cv, shuffle = True, random_state = 0) ## for testing fixing random_state
    for train,test in kf.split(x):
        x_train = x[train,:]
        y_train = y[train,:]
        x_test = x[test,:]
        y_test = y[test,:]

        # from multilabel to multiclass based on independencec assumption
        x_train, y_train = multiClass(x_train,y_train)

        # feature standardization #
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # training and predict
        y_pred, coef, interc = logRegFeatureEmotion(x_train, y_train, x_test)

        # performance measure
        performance.append(perfMeasure(y_pred,y_test))
        coefi.append(coef)
        intercept.append(interc)

    coefi = np.array(coefi)
    coef = np.mean(coefi,axis=0)
    coef_std = np.std(coefi,axis=0)
    performance = np.array(performance)
    perf = np.mean(performance,axis=0)
    perf_std = np.std(performance,axis=0)
    intercept = np.array(intercept)
    interc = np.mean(intercept,axis=0)
    interc_std = np.std(intercept,axis=0)

    return coef, coef_std, perf, perf_std, interc, interc_std

def perfMeasure(y_pred, y_test):
    # performance measurement #
    # with background noise probability distribution
    Nperf=3
    perf_list={"acc3":0, "dcg":1, "llh":2}
    perf=[0 for i in range(Nperf)]

    rank_test = map(rankOrder,y_test)
    rank_pred = map(rankOrder,y_pred)
    Nsamp = y_test.shape[0]
    Nclass = y_test.shape[1]


    # acc3 #
    if Nclass>=3:
        for i in range(Nsamp):
            acc3 = 1
            for j in range(3):
                if rank_test[i][j]>=0 and rank_pred[i][j]>=0 and rank_test[i][j]!=rank_pred[i][j]:
                    acc3 = 0
            perf[perf_list["acc3"]] += acc3
        perf[perf_list["acc3"]]=perf[perf_list["acc3"]]/(1.0*Nsamp)
    else:
        print "cannot calculate acc@3 for less than 3 classes"

    # dcg #
    for i in range(Nsamp):
        dcg = 0
        maxdcg = 0
        for j in range(Nclass):
            if rank_pred[i][j]>=0:
                dcg += y_test[i][rank_pred[i][j]]/math.log(j+2)
            if rank_test[i][j]>=0:
                maxdcg += y_test[i][rank_test[i][j]]/math.log(j+2)
        perf[perf_list["dcg"]] += dcg*1.0/maxdcg
    perf[perf_list["dcg"]]= perf[perf_list["dcg"]] /(1.0*Nsamp)

    # llh (log-likelihood)#
    y_pred_noise=addNoise(y_pred)# add noise prior
    # y_pred_noise = y_pred
    for i in range(Nsamp):
        llh = 0
        for j in range(Nclass):
            if y_test[i][j]==0:
                maxllh=0
            else:
                maxllh=y_test[i][j]*math.log(y_test[i][j])
            llh += (y_test[i][j]*math.log(y_pred_noise[i][j])-maxllh)
        llh = llh + sum(y_test[i])*math.log(sum(y_test[i]))
        perf[perf_list["llh"]] += llh

    return perf

def addNoise(dist_list):
    noise = np.array([NOISE for i in range(dist_list.shape[1])])
    return map(lambda x: (x+noise)/sum(x+noise), dist_list)
def rankOrder(dist):
    rank=[i for i in range(len(dist))]
    for i in range(1,len(dist)):
        for j in range(1,len(dist)-i+1):
            if dist[rank[j]]>dist[rank[j-1]]:
                temp=rank[j]
                rank[j]=rank[j-1]
                rank[j-1]=temp
    for i in range(len(dist)):
        if dist[rank[i]]==0:
            rank[i]=-1
    return rank


def multiClass(x,y):
    # reducing multiclass samples with cumulative # labels to samples each with one label
    y_shape = y.shape
    n_sample = y_shape[0]
    n_class = y_shape[1]

    rep = np.sum(y, axis = 1)
    x_rep = np.repeat(x,rep.astype(int),axis=0)

    base_class = np.array([i for i in range(n_class)])
    y_rep = np.repeat(base_class.reshape([1,n_class]),n_sample,axis = 0)
    y_rep = y_rep.reshape([n_class*n_sample])
    rep = y.reshape([n_class*n_sample])
    y_rep = np.repeat(y_rep,rep.astype(int))

    return x_rep, y_rep



def logRegFeatureEmotion(x_training,y_training, x_test):
    logReg = linear_model.LogisticRegression(C=1e9, fit_intercept=True, multi_class="ovr")### test
    fitResult = logReg.fit(x_training,y_training)
    y= fitResult.predict_proba(x_test)
    coef = logReg.coef_
    intercept = logReg.intercept_
    return y, coef, intercept

def DataSimulated(Nsamp, Nfeature, Nclass, Beta, Robs, Lrandom=0.5):
    x=np.random.random((Nsamp,Nfeature))
    beta = Beta.reshape([Nclass,Nfeature])
    y_innorm= np.exp(np.inner(x,beta)) + np.random.random((Nsamp,Nclass))*Lrandom
    y_normalizer = np.sum(y_innorm,axis=1,keepdims=True)
    y_norm = y_innorm/y_normalizer
    obs = np.random.random([Nsamp,1])*Robs
    y = np.multiply(y_norm,np.repeat(obs,Nclass,axis=1)).astype(int)+1
    # print "beta", beta
    # print "y_innorm", y_innorm
    # print "y_normalizer", y_normalizer
    # print "y_norm", y_norm
    # print "sum_y_norm", np.sum(y_norm, axis=1)
    # print "obs", obs
    # print "y", y

    return x,y

if __name__ == "__main__":
    x,y= dataClean("data/posts_Feature_Emotion.txt")
    print "number of samples: ", x.shape[0]

    ### test ####
    # feature = 0 # chose single feature to fit
    # feature_name = EL[feature]
    # result = trainTest(x[:,feature].reshape([x.shape[0],1]),y)
    # feature_name = "No feature"
    # X_non =np.ones([y.shape[0],1]).astype("float")
    # result = trainTest(X_non,y)
    result = trainTest(x,y)
    feature_name = "all"
    print "------%s feature -----" % feature_name
    for item in result:
        print item
    # write2result #
    file = open("result.txt","a")
    file.write("------%s feature -----" % feature_name)
    file.write(str(result))
    file.close()

    # ## test ###
    # Nfeature = 4
    # Nclass = 6
    # Beta = np.arange(Nclass)/5.0-1.0
    # Beta = np.repeat(Beta,Nfeature)
    # print Beta
    # x,y=DataSimulated(3,Nfeature,Nclass,Beta,100,Lrandom=0.0)
    # # result = trainTest(x,y)
    # # print result
    # print x
    # print y
    # print multiClass(x,y)
