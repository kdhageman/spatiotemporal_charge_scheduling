from typing import List

import numpy as np


def dist(a: List[float], b: List[float]) -> bool:
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist3(a: List[float], b: List[float]) -> bool:
    """
    Returns the Euclidean distance between points 'a' and 'b'
    :param a: the (x,y,z) coordinates of the first point
    :param b: the (x,y,z) coordinates of the second point
    """
    x1, y1, z1 = a
    x2, y2, z2 = b
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
