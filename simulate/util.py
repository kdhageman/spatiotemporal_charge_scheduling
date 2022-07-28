import numpy as np

from util.scenario import Scenario


def maximum_schedule_delta(sc: Scenario, v: np.array, W: int, sigma: int):
    n_waypoints = min(1 + sigma * (W - 1), sc.N_w)
    move_times = []

    for d in range(sc.N_d):
        for offset in range(sc.N_w - n_waypoints + 1):
            dist = sum(sc.D_N[d, -1, offset:offset + n_waypoints - 1])
            move_times.append(dist / v[d])
    return min(move_times)


def gen_colors(n: int):
    np.random.seed(0)
    res = []
    for d in range(n):
        c = np.random.rand(3).tolist()
        res.append(c)
    return res
