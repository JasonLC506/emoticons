import ast

filename = "data/foxnews_Feature_linkemotion.txt"
resultname = "data/foxnews_Feature_linkemotion_deduplic.txt"

def deduplicate(filename):
    file = open(filename, "r")
    ids = {}
    resultfile = open(resultname, "a")
    for item in file.readlines():
        try:
            sample = ast.literal_eval(item.rstrip())
        except SyntaxError, e:
            print e.message
            continue
        if sample["id"] not in ids:
            ids[sample["id"]]=True
            resultfile.write(str(item))
    file.close()
    resultfile.close()
    print "done"

if __name__ == "__main__":
    deduplicate(filename)
