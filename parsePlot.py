import numpy as np
from logRegFeatureEmotion import dataClean
from stats import pairwise
from matplotlib import pyplot as plt
import math

# post = "washington"
# filenames = [post+"_recall_rpp.txt", post+"_recall_dt.txt"]
# nums_list = [[] for i in range(len(filenames))]
# for i in range(len(filenames)):
#     filename = filenames[i]
#     file = open(filename,"r")
#     for line in file.readlines():
#         set = line.rstrip().split(",")
#         for num in set:
#             if "nan" in num:
#                 nums_list[i].append(np.NaN)
#             elif not num:
#                 continue
#             else:
#                 print num
#                 nums_list[i].append(float(num))
#     print nums_list[i]
#     file.close()
#
# x,y= dataClean("data/"+post+"_Feature_linkemotion.txt")
# paircomp, paircomp_sub = pairwise(y)
# Nclass = len(paircomp)
# imbalist = []
# for i in range(Nclass):
#     for j in range(Nclass):
#         if i==j:
#             imbalist.append(np.NaN)
#         else:
#             if paircomp[i][j][0]*paircomp[i][j][1]!=0:
#                 imbalist.append(math.log(paircomp[i][j][0]*1.0/paircomp[i][j][1]))
#             else:
#                 imbalist.append(np.NaN)
#
#
# for i in range(len(nums_list)):
#     nums = nums_list[i]
#     plt.plot(imbalist,nums, "o", label=filenames[i])
# plt.legend()
# plt.show()

posts =["nytimes", "wsj", "washington","rou"]
nums_list = [[] for i in range(2)]
imbalist = []
for post in posts:
    filenames = [post+"_recall_rpp.txt", post+"_recall_dt.txt"]
    for i in range(len(filenames)):
        filename = filenames[i]
        file = open(filename,"r")
        for line in file.readlines():
            set = line.rstrip().split(",")
            for num in set:
                if "nan" in num:
                    nums_list[i].append(np.NaN)
                elif not num:
                    continue
                else:
                    print num
                    nums_list[i].append(float(num))
        print nums_list[i]
        file.close()
    if post != "rou":
        x,y= dataClean("data/"+post+"_Feature_linkemotion.txt")
    else:
        x,y= dataClean("data/posts_Feature_Emotion.txt")
    paircomp, paircomp_sub = pairwise(y)
    Nclass = len(paircomp)
    for i in range(Nclass):
        for j in range(Nclass):
            if i==j:
                imbalist.append(np.NaN)
            else:
                if paircomp[i][j][0]*paircomp[i][j][1]!=0:
                    imbalist.append(math.log(paircomp[i][j][0]*1.0/paircomp[i][j][1]))
                else:
                    imbalist.append(np.NaN)

labels = ["rpp", "dt"]
for i in range(len(nums_list)):
    nums = nums_list[i]
    plt.plot(imbalist,nums, "o", label=filenames[i])
plt.legend()
plt.show()