"""
cross data set experiment default DecisionTree without pruning
"""

from DecisionTreeWeight import DecisionTree
from DecisionTreeWeight import label2Rank
import logRegFeatureEmotion as LogR

def crossData(data_list, alpha = 0.0, rank_weight = False, stop_criterion_mis_rate = None, stop_criterion_min_node = 1,
                  stop_criterion_gain = 0.0, prune_criteria = 0):
    results = {}
    for data_train in data_list:
        results[data_train] = {}
        for data_test in data_list:
            if data_test == data_train:
                continue
            x_train, y_tr = LogR.dataClean(data_train)
            y_train = label2Rank(y_tr.tolist())
            x_test, y_te = LogR.dataClean(data_test)
            y_test = label2Rank(y_te.tolist())
            tree = DecisionTree().buildtree(x_train, y_train, weights= None,
                                            stop_criterion_mis_rate=stop_criterion_mis_rate,
                                            stop_criterion_min_node=stop_criterion_min_node,
                                            stop_criterion_gain=stop_criterion_gain
                                            )
            y_pred = tree.predict(x_test, alpha)
            results[data_train][data_test] = LogR.perfMeasure(y_pred, y_test, rankopt=True)
    return results

if __name__ == "__main__":
    news_list = ["nytimes", "foxnews", "wsj", "washington"]
    data_list = map(lambda x: "data/" + x + "_Feature_linkemotion.txt", news_list)
    results = crossData(data_list)
    with open("results/result_crossdata.txt", "a") as f:
        f.write("decisiontree without pruning\n")
        f.write(str(results)+"\n")