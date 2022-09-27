import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


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


def is_feasible(sc, params, anchors) -> bool:
    """
    Returns whether the given scenario and parameters *might* be feasible.
    This can be used as a sanity check for the given problem.
    Note that it is not guaranteed to be correct.
    :param sc: scenario
    :param params: parameters
    """
    for d in range(sc.N_d):
        if len(anchors[d]) == 0:
            b_before = params.B_start[d]
            dist = sc.D_N[d, -1, :].sum()
            l = ["W"] * (sc.N_w + 1)
            depletion = dist / params.v[d] * params.r_deplete[d]
            b_end = b_before - depletion
            surplus = b_end - params.B_min[d]
            logger.debug(f"for UAV [{d}] without anchors, traversing {dist:.2f}m, depleting {depletion * 100:.2f}% battery (or a {surplus * 100:.2f}% surplus) ({' - '.join(l)})")
            if surplus < 0:
                return False
            continue
        for i, a in enumerate(anchors[d]):
            dist = 0
            if i == 0:
                # first anchor
                b_before = params.B_start[d]
                l = []

                # distances between non-anchor waypoints up to the first one
                for w in range(a):
                    dist += sc.D_N[d, -1, w]
                    l.append("W")
                l.append("A")  # anchor itself

                # distance to nearest station after anchor
                dist += sc.D_N[d, :-1, a].min()
                l.append("S")
            else:
                # subsequent anchors
                b_before = params.B_max[d]  # we assume full charge after the previous anchor
                a_prev = anchors[d][i - 1]
                l = []

                # dist from nearest charging station after anchor
                dist += sc.D_W[d, :-1, a_prev].min()
                l.append("S")

                # distances between non-anchor waypoints in between
                for w in range(a_prev + 1, a):
                    dist += sc.D_N[d, -1, w]
                    l.append("W")
                l.append("A")  # anchor itself

                # distance from anchor to the nearest station
                dist += sc.D_N[d, :-1, a].min()
                l.append("S")
            depletion = dist / params.v[d] * params.r_deplete[d]
            b_end = b_before - depletion
            surplus = b_end - params.B_min[d]
            logger.debug(f"for UAV [{d}] at anchor '{a}', traversing {dist:.2f}m, depleting {depletion * 100:.2f}% battery (or a {surplus * 100:.2f}% surplus) ({' - '.join(l)})")
            if surplus < 0:
                return False
        # after last anchor
        b_before = params.B_max[d]

        # distance from closest charging station
        last_anchor = anchors[d][-1]
        dist = sc.D_W[d, :-1, last_anchor].min()
        l = ["S"]

        # distance across remaining waypoints
        for w in range(last_anchor + 1, sc.N_w):
            dist += sc.D_N[d, -1, w]
            l.append("W")
        l.append("W")  # for last waypoint

        depletion = dist / params.v[d] * params.r_deplete[d]
        b_end = b_before - depletion
        surplus = b_end - params.B_end[d]
        logger.debug(f"for UAV [{d}] after anchor '{a}', traversing {dist:.2f}m, depleting {depletion * 100:.2f}% battery (or a {surplus * 100:.2f}% surplus) ({' - '.join(l)})")
        if surplus < 0:
            return False
    return True


X_OFFSET = 1
Y_DIST = 0.25

def as_graph(sc, params, anchors, d):
    g = nx.DiGraph()
    new_node = f"w0"
    g.add_node(new_node)
    positions = {new_node: (0, 0)}

    x = 1
    for w_d in range(1, sc.N_w + 1):
        w_s = w_d - 1

        # new waypoint node
        prev_node = f"w{w_s}"
        new_node = f"w{w_d}"
        positions[new_node] = (x, 0)
        g.add_node(new_node)

        # path layer
        path_layer = []
        path_idx = 0
        for s in range(sc.N_s):
            if w_s in anchors[d]:
                path_layer.append((path_idx, f"w{w_s}_s{s}"))
            path_idx += 1
        path_layer.append((path_idx, f"w'{w_d}"))

        for i, (idx, n) in enumerate(path_layer):
            g.add_node(n)
            dist_to = sc.D_N[d, idx, w_s]
            dist_from = sc.D_W[d, idx, w_s]

            g.add_edge(prev_node, n, dist=dist_to)
            g.add_edge(n, new_node, dist=dist_from)
            y_total = (len(path_layer) - 1) * Y_DIST
            if not y_total:
                y = 0
            else:
                y = i * y_total / (len(path_layer) - 1) - (y_total / 2)
            positions[n] = (x - (X_OFFSET / 2), y)

        x += X_OFFSET
    # TODO: add weights
    # TODO: consider anchors

    return g, positions
