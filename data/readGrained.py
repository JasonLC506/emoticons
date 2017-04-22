import cPickle

news = "nytimes"
filename = news + "_grained_reaction"

with open(filename, "r") as f:
    series_set = cPickle.load(f)
keys = series_set.keys()
print "# of posts ", len(keys)

total_length = 0
for key in keys:
   total_length += len(series_set[key])
print "# of reactions total ", total_length

print "------ first post ---------"
print series_set[keys[0]]
