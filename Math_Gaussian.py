import numpy as np

def Gaussian(x, mean, var, scalar_variance = False, diagonal_variance = False):
    y = x.flatten()
    u = mean.flatten()
    if scalar_variance:
        prob = np.exp(-0.5 * np.inner((y-u),(y-u)) / var) \
               / np.sqrt(np.power(2 * np.pi * var, u.shape[0]))
    else:
        if diagonal_variance:
            prob = np.exp(-0.5 * np.sum(np.divide(np.power((y-u),2), var.flatten()))) \
                / np.sqrt(np.power(2 * np.pi, u.shape[0]) * np.prod(var))
        else:
            prob = np.exp(-0.5 * np.inner((y- u), np.inner(np.linalg.inv(var), (y - u)))) \
                / np.sqrt(np.power(2 * np.pi, u.shape[0]) * abs(np.linalg.det(var)))
    return prob