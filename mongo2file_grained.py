import pymongo
from logRegFeatureEmotion import emoticon_list
from CrawlerEmotiComment_Grained import postsFilter
import cPickle

news = "washington"
posts = postsFilter("data/"+news+"_Feature_linkemotion.txt")
filename = "data/"+news+"_grained_reaction"

for i in range(len(emoticon_list)):
    emoticon_list[i] = emoticon_list[i].upper()

database = pymongo.MongoClient().dataSet
col_r = database.user_reaction

series_set = {}
for post in posts:
    react_series = []
    query_result = col_r.find({"POSTID":post})
    for react in query_result:
        react_series.append(emoticon_list.index(react["EMOTICON"]))
    series_set[post] = react_series
    print "done ", post

with open(filename,"w") as f:
    cPickle.dump(series_set, f)
