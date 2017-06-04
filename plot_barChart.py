import numpy as np
import matplotlib.pyplot as plt
import matplotlib

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
matplotlib.rcParams.update({'font.size': 12})

group_names = ["(\"haha\",\"like\")", "(\"sad\",\"like\")", "(\"angry\",\"like\")"]
model_names = ["LogR","RPC","NAIVE","LWR","SMP","KNN-PL","KNN-M", "LogLinear","DTPG"]

N_group = len(group_names)
N_model = len(model_names)

# model_group_value = np.zeros([len(model_names), len(group_names)])
model_group_value = np.array([[0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.157, 0.038, 0.024],
                              [0.213, 0.118, 0.085]])

ind = np.arange(N_group)
width = 0.1

fig, ax = plt.subplots()
rects = []

for imodel in range(N_model):
    if imodel != 8:
        rects.append(ax.bar(ind + imodel*width, model_group_value[imodel], width, color=colors[imodel]))
    else:
        print "last one"
        rects.append(ax.bar(ind + imodel * width, model_group_value[imodel], width, color="y"))
ax.set_ylabel("recall")
ax.set_xticks(ind + width * N_model / 2.0)
ax.set_xticklabels(group_names)
ax.legend(rects, model_names)

plt.show()
