import numpy as np
filename_feature = "data/sushi3.udata"
filename_target = "data/sushi3a.5000.10.order"
def readSushiData(filename_feature=filename_feature , filename_target=filename_target):
    """
    return x: ndarray int32
           y: ndarray int32 ranking vector with item id shown in ranking position
    """
    x = []
    y = []
    # read feature file #
    file_feature = open(filename_feature,"r")
    for line in file_feature.readlines():
        record = line.rstrip().split("\t")
        record = map(int, record)
        del record[0]
        x.append(record)
    file_feature.close()
    x = np.array(x)
    # print x.shape
    # print x[0]
    # print type(x), x.dtype

    # read target file #
    file_target = open(filename_target,"r")
    file_target.readline()
    for line in file_target.readlines():
        record = line.rstrip().split(" ")
        record = map(int, record)
        del record[0]
        del record[0]
        y.append(record)
    file_target.close()
    y = np.array(y)

    return x,y

if __name__ == "__main__":
    x, y = readSushiData()
    print y.shape[1]