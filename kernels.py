import numpy as np


def linear_kernel(x, y):
    return np.dot(x, y)


def polynomial_kernel(x, y, d=5):
    return (np.dot(x, y)) ** d


def RBF_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm((x - y)) ** 2 / (2 * (sigma ** 2)))
