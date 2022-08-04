import numpy as np


def maximum_schedule_delta(sc, v: np.array, W: int, sigma: int):
    n_waypoints = min(1 + sigma * (W - 1), sc.N_w)
    move_times = []

    for d in range(sc.N_d):
        for offset in range(sc.N_w - n_waypoints + 1):
            dist = sum(sc.D_N[d, -1, offset:offset + n_waypoints - 1])
            move_times.append(dist / v[d])
    return min(move_times)


def gen_colors(n: int):
    base_colors = np.array([
        [204, 51, 51],
        [51, 204, 51],
        [51, 51, 204],
    ]) / 255

    np.random.seed(3)
    res = []
    i = 0
    while len(res) < 3:
        c = base_colors[i]
        res.append(c)
        i += 1
        if i == 3:
            break

    while len(res) < n:
        c = np.random.rand(3).tolist()
        res.append(c)
    return res


def gen_linestyles(n: int):
    styles = ['solid', 'dotted', 'dashed', 'dashdot']
    res = []
    for d in range(n):
        style = styles[d % len(styles)]
        res.append(style)
    return res
