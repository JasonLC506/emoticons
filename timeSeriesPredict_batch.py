from timeSeriesPredict_single import readGrainedData
from timeSeriesPredict_single import stateDiff
from timeSeriesPredict_single import performance
from timeSeriesPredict_single import AlARMTHRESHOLD
from Heard_batch import HeardBatch
from Heard import cumulate
import numpy as np
import sys
from datetime import datetime
from datetime import timedelta

MCSAMPLES = 1000


def traintest(time_series_list, time_init_prop, time_target_absdiff):
    result = {"perf":[], "mu":[], "theta":[], "f":[]}

    ## preprocess ##
    M = len(time_series_list)
    time_series_cumulate_list = map(cumulate, time_series_list)
    L_list = map(lambda item: item.shape[0], time_series_list)
    time_init_list = map(lambda item: int(item*time_init_prop), L_list)
    time_target_list = map(lambda item: item + time_target_absdiff, time_init_list)

    for m in range(M):
        assert time_target_list[m] <= L_list[m]

    state_init_list = []
    state_target_list = []
    train = []
    for m in range(M):
        train.append(time_series_list[m][:time_init_list[m],:])
        state_init_list.append(time_series_cumulate_list[m][time_init_list[m]])
        state_target_list.append(time_series_cumulate_list[m][time_target_list[m]])

    ## fit ##
    print "start fitting" ### test
    heard = HeardBatch().fit(train, lamda = 1.0, f_constant=True)
    # write fit parameters #
    result["mu"] = heard.mu
    result["theta"] = heard.theta
    result["f"] = heard.f

    ## predict ##
    print "start predicting" ### test
    state_predicted_list = heard.predict(time_target_list, Nsamp=MCSAMPLES)
    for m in range(M):
        diff_predict = stateDiff(state_init_list[m], state_predicted_list[m], time_init_list[m], time_target_list[m])
        diff_true = stateDiff(state_init_list[m], state_target_list[m], time_init_list[m], time_target_list[m])
        perf = performance(diff_predict, diff_true)
        if perf >= AlARMTHRESHOLD:
            print "huge ppl ", perf
            print "state_init ", state_init_list[m]
            print "state_predicted ", state_predicted_list[m]
            print "state_target ", state_target_list[m]
            print "diff_predict ", diff_predict
            print "diff_true ", diff_true
        result["perf"].append(perf)
    mean = np.mean(result["perf"])
    std = np.std(result["perf"])
    result["perf"] = [mean, std]

    return result


if __name__ == "__main__":
    ## input: python timeSeriesPredict_single.py news time_init_proportion ##
    news = str(sys.argv[1])
    time_init_proportion = float(sys.argv[2])
    # news = "nytimes"
    # time_init_proportion = 0.7

    filename = "data/" + news + "_grained_reaction"
    result_filename = "results/Heard_batch_predict.txt"

    time_target_absdiff = 100
    time_series_list = readGrainedData(filename, Nclass=6)
    # for i in range(1790):
    #     del time_series_list[0] ### test
    print "in total", len(time_series_list)
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
        f.write("fConstant\n")
        f.write("takes %f seconds\n" % duration)
        f.write(str(result)+"\n")
