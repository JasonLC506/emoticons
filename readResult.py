from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np


def readResult(filename):
    results = [] # [Nu, Nv, acc3, tau, GMR]
    with open(filename, "r") as f:
        result = []
        interval = None
        for line in f.readlines():
            words = line.lstrip(" ").split(" ")
            if interval is not None:
                interval += 1
                if interval == 7:
                    result.append(float(words[-1].rstrip(",\n")))
                if interval == 32:
                    result.append(float(words[0].rstrip("]),")))
                    ## complete one record ##
                    results.append(result)
                    result = []
                    interval = None
            if "Nu" in words[0]:
                result.append(int(words[1].rstrip(",")))
                result.append(int(words[3].rstrip("\n")))
                continue
            if "perf" in words[0]:
                result.append(float(words[3].rstrip(","))) # acc @3
                interval = 0
                continue
    return results


def aggregatePlot(results, agg_axis, x_axis, y_axes, fieldnames):
    ## aggregate ##
    aggregated_data = {}
    for record in results:
        if record[agg_axis] not in aggregated_data.keys():
            aggregated_data[record[agg_axis]] = [record]
        else:
            aggregated_data[record[agg_axis]].append(record)
    for key in aggregated_data.keys():
        aggregated_data[key] = np.array(aggregated_data[key])
    ## plot ##
    for y_axis in y_axes:
        # plt.title(fieldnames[y_axis])
        for key in aggregated_data.keys():
            plt.plot(aggregated_data[key][:,x_axis], aggregated_data[key][:,y_axis], label=fieldnames[agg_axis]+"="+str(key))
        plt.xlabel(fieldnames[x_axis])
        plt.ylabel(fieldnames[y_axis])
        plt.legend()
        plt.show()


def tdplot(results, x_axis, y_axis, z_axes, fieldnames):
    data = np.array(results)
    x_values = []
    y_values = []
    for isamp in range(data.shape[0]):
        x_value = data[isamp][x_axis]
        y_value = data[isamp][y_axis]
        if x_value not in x_values:
            x_values.append(x_value)
        if y_value not in y_values:
            y_values.append(y_value)
    x_values.sort()
    y_values.sort()
    xs, ys = np.meshgrid(np.array(x_values), np.array(y_values))
    for z_axis in z_axes:
        zs = np.zeros(xs.shape, dtype=np.float64)
        for isamp in range(data.shape[0]):
            zs[x_values.index(data[isamp, x_axis]), y_values.index(data[isamp, y_axis])] = data[isamp, z_axis]
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm, linewidth=0)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel(fieldnames[x_axis])
        ax.set_ylabel(fieldnames[y_axis])
        ax.set_zlabel(fieldnames[z_axis])
        fig.colorbar(surf)
        plt.show()


if __name__ == "__main__":
    filename = "results/result_CAD.txt"
    fieldnames = ["Nu", "Nv", "acc@3", "tau", "GMR"]
    results = readResult(filename)
    # aggregatePlot(results, 0, 1, [2,3,4], fieldnames)
    tdplot(results, 0, 1, [2,3,4], fieldnames)