import numpy as np


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist3(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
