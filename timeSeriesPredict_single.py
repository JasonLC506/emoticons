import cPickle
from Heard import *
from Poisson import poisson
import numpy as np
from datetime import datetime
from datetime import timedelta
import sys
# from matplotlib import pyplot as plt

MCSAMPLES = 100 # number of MC samples
ZEROSUBSTITUTE = 0.001
AlARMTHRESHOLD = 10.0

def readGrainedData(filename, Nclass, Nsamp=None):
    with open(filename, "r") as f:
        series_set = cPickle.load(f)
    time_series_list = []
    min_length = 0
    max_length = 0
    isamp = 0
    for key in series_set.keys():
        series = series_set[key]
        length = len(series)
        if min_length == 0 or length < min_length:
            min_length = length
        if max_length == 0 or length > max_length:
            max_length = length
        series = series[::-1] # the origin order is reversed in crawling
        label_sequence = value2oneHot(series, Nclass)
        time_series_list.append(label_sequence)
        isamp += 1
        if Nsamp is not None and isamp >= Nsamp:
            break
    print "in total ", len(time_series_list)
    print "shortest ", min_length
    print "longest ",  max_length
    return time_series_list


def value2oneHot(series, Nclass):
    L = len(series)
    label_sequence = np.zeros([L, Nclass], dtype = np.float64)
    for time in range(L):
        label = series[time]
        label_sequence[time][label] = 1.0
    assert np.sum(label_sequence) == L
    return label_sequence


def traintest(time_series_list, time_init_prop, time_target_absdiff):
    result = {"perf":[], "mu":[], "theta":[]}

    ## calculate for each series ##
    for series in time_series_list:
        series_cumulate = cumulate(series)
        ### test ###
        # for d in range(series_cumulate.shape[1]):
        #     plt.plot(series_cumulate[:,d], label="%d" % d)
        # plt.legend()
        # plt.show()
        #############

        ## fit & predict ##
        L = series.shape[0]
        time_init = int(time_init_prop * L) # for propotion input
        time_target = time_init + time_target_absdiff
        assert time_target < L
        # fit #
        #### using independent Poisson model  ###
        heard = Heard().fit(series[:time_init,:], lamda = 1.0, f_constant=True)
        # write fit parameter #
        result["mu"].append(heard.mu)
        result["theta"].append(heard.theta)
        # predict #
        state_init = series_cumulate[time_init]
        state_target = series_cumulate[time_target]
        #### using independent Poisson model no need of fitting ###
        state_predicted = heard.predict(time_target, Nsamp = MCSAMPLES) # MC samples
        # state_predicted = poisson().fit(state_init, time_init).predict() # prediction of independent Poisson model
        # print "init state: ", state_init
        # print "target state: ", state_target
        # print "predicted state: ", state_predicted
        #### using independent Poisson model no need of fitting ###
        diff_predict = stateDiff(state_init, state_predicted, time_init, time_target)
        # diff_predict = state_predicted # prediction of independent Poisson model
        diff_true = stateDiff(state_init, state_target, time_init, time_target)
        perf = performance(diff_predict, diff_true)
        if perf >= AlARMTHRESHOLD:
            print "huge ppl ", perf
            print "state_init ", state_init
            print "state_predicted ", state_predicted
            print "state_target ", state_target
            print "diff_predict ", diff_predict
            print "diff_true ", diff_true
        result["perf"].append(perf)

    ## summarize ##
    for key in result.keys():
        item = np.array(result[key])
        mean = np.nanmean(item, axis = 0)
        std = np.nanstd(item, axis = 0)
        result[key] = [mean, std]

    return result


def stateDiff(state_init, state_target, time_init, time_target, distribution = True):
    """
    input: cumulative label distribution at init and target time stamp,
    output: cumulative label distribution in between or absolute # labels
    """
    diff_abs = state_target * time_target - state_init * time_init
    diff_total = np.sum(diff_abs)
    # try:
    #     assert diff_total == (time_target-time_init)
    # except AssertionError, e:
    #     print diff_total
    #     print time_target-time_init
    #     raise e
    if not distribution:
        return diff_total
    diff_distribution = diff_abs / diff_total
    return diff_distribution


def performance(diff_pred, diff_true):
    """
    perplexity
    """
    ppl = - np.inner(diff_true, np.log(diff_pred))
    return ppl


if __name__ == "__main__":
    ## input: python timeSeriesPredict_single.py news time_init_proportion ##
    news = str(sys.argv[1])
    time_init_proportion = float(sys.argv[2])
    # news = "nytimes"
    # time_init_proportion = 0.7

    filename = "data/" + news + "_grained_reaction"
    result_filename = "results/Heard_predict.txt"

    time_target_absdiff = 100
    time_series_list = readGrainedData(filename, Nclass=6)
    start = datetime.now()
    result = traintest(time_series_list, time_init_proportion, time_target_absdiff)
    duration = (datetime.now() - start).total_seconds()
    print result
    with open(result_filename, "a") as f:
        f.write(news+"\n")
        f.write("total time series: %d\n" % len(time_series_list))
        f.write("time_init_proportion: %f\n" % time_init_proportion)
        f.write("time_target_absdiff: %d\n" % time_target_absdiff)
        f.write("MCSAMPLES: %d\n" % MCSAMPLES)
        f.write("fconstant\n")
        f.write("takes %f seconds\n" % duration)
        f.write(str(result)+"\n")
