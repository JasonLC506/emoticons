import numpy as np

def readSyntheticData(filename):
    x = []
    y = []
    with open(filename, "r") as f:
        for line in f.readlines():
            atts = line.rstrip("\n").split(",")
            y_single = atts[-1]
            del atts[-1]
            x_single = map(float, atts)
            x.append(x_single)
            y.append(rankParse(y_single))
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.int16) - 1 # minus 1 for 0-index
    return x, y

def rankParse(rankstr):
    labels = rankstr.split(">")
    labels = map(lambda x: int(x.lstrip("L")), labels)
    return labels

if __name__ == "__main__":
    x, y = readSyntheticData("data/synthetic/vowel")
    print x[0]
    print y[0]