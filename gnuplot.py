import matplotlib.pyplot as plt

filenames = ["time_dt", "time_Mallows"]
labels = ["DTPG", "DTMallows"]
x_list = []
y_list = []
for filename in filenames:
    x = []
    y = []
    with open(filename,"r") as f:
        for line in f.readlines():
            nums = line.rstrip().split(" ")
            x.append(float(nums[0]))
            y.append(float(nums[1]))
    x_list.append(x)
    y_list.append(y)
marks =["+","*"]
for i in range(2):
    x=x_list[i]
    y=y_list[i]
    plt.plot(x,y, marks[i], label=labels[i])
    # plt.yscale("log")
    # plt.xscale("log")
plt.xlabel("N (size of dataset)")
plt.ylabel("running time (secs)")
plt.legend()
plt.show()