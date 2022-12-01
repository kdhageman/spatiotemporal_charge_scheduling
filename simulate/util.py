import logging

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

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


def is_feasible(sc, params) -> bool:
    """
    Returns whether the given scenario and parameters *might* be feasible.
    This can be used as a sanity check for the given problem.
    Note that it is not guaranteed to be correct.
    :param sc: scenario
    :param params: parameters
    """
    for d in range(sc.N_d):
        if len(sc.anchors[d]) == 0:
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
        for i, a in enumerate(sc.anchors[d]):
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
                a_prev = sc.anchors[d][i - 1]
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
            surplus = b_end - params.B_min[d, a]
            logger.debug(f"for UAV [{d}] at anchor '{a}', traversing {dist:.2f}m, depleting {depletion * 100:.2f}% battery (or a {surplus * 100:.2f}% surplus) ({' - '.join(l)})")
            if surplus < 0:
                return False
        # after last anchor
        b_before = params.B_max[d]

        # distance from closest charging station
        last_anchor = sc.anchors[d][-1]
        dist = sc.D_W[d, :-1, last_anchor].min()
        l = ["S"]

        # distance across remaining waypoints
        for w in range(last_anchor + 1, sc.N_w):
            dist += sc.D_N[d, -1, w]
            l.append("W")
        l.append("W")  # for last waypoint

        depletion = dist / params.v[d] * params.r_deplete[d]
        b_end = b_before - depletion
        surplus = b_end - params.B_min[d, -1]
        logger.debug(f"for UAV [{d}] after anchor '{a}', traversing {dist:.2f}m, depleting {depletion * 100:.2f}% battery (or a {surplus * 100:.2f}% surplus) ({' - '.join(l)})")
        if surplus < 0:
            return False
    return True


X_OFFSET = 1
Y_DIST = 0.25


def as_graph(sc, d):
    g = nx.DiGraph()
    w = 0
    wp_counter = 0
    node_idx = f'w{wp_counter}'
    label = r'$w_s$'
    nodetype = "waypoint"
    g.add_node(node_idx, w=w, label=label, nodetype=nodetype)
    positions = {node_idx: (0, 0)}
    prev_node = node_idx
    wp_counter += 1

    x = 1
    for w_d in range(1, sc.N_w + 1):
        w_s = w_d - 1

        # new waypoint node
        node_idx = f'w{wp_counter}'
        if w_s < len(sc.waypoint_ids[d]):
            waypoint_idx = sc.waypoint_ids[d][w_s]
        else:
            waypoint_idx = sc.waypoint_ids[d][-1]
        label = f'$w_{{{waypoint_idx}}}$'
        nodetype = 'waypoint'
        positions[node_idx] = (x, 0)
        g.add_node(node_idx, w=w_d, label=label, nodetype=nodetype)
        new_node = node_idx

        # path layer
        path_layer = []
        path_idx = 0
        for s in range(sc.N_s):
            node_idx = f'w{wp_counter}p{s}'
            if w_s in sc.anchors[d]:
                path_layer.append((path_idx, node_idx))
            path_idx += 1
        node_idx = f'w{wp_counter}p{sc.N_s}'
        path_layer.append((path_idx, node_idx))

        for i, (n, nid) in enumerate(path_layer):
            nodetype = "path_node"
            if n == sc.N_s:
                # directly to next waypoint
                label = f"$w'$"
            else:
                # via station
                label = f"$s_{{{n}}}$"
            g.add_node(nid, w=w_s, n=n, nodetype=nodetype, label=label)
            dist_to = sc.D_N[d, n, w_s]
            dist_from = sc.D_W[d, n, w_s]

            g.add_edge(prev_node, nid, dist=dist_to)
            g.add_edge(nid, new_node, dist=dist_from)
            y_total = (len(path_layer) - 1) * Y_DIST
            if not y_total:
                y = 0
            else:
                y = i * y_total / (len(path_layer) - 1) - (y_total / 2)
            positions[nid] = (x - (X_OFFSET / 2), y)

        wp_counter += 1
        x += X_OFFSET
        prev_node = new_node

    return g, positions


def draw_graph(sc, params, fname):
    height = (sc.N_s + 1) * sc.N_d
    width = sc.N_w * 2
    _, axes = plt.subplots(nrows=sc.N_d, ncols=1, figsize=(width, height))
    if sc.N_d == 1:
        axes = [axes]

    for d in range(sc.N_d):
        g, pos = as_graph(sc, d)

        ax = axes[d]
        nx.draw(g, pos, ax=ax)
        node_labels = {node: dat['label'] for node, dat in g.nodes(data=True)}
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=6, font_color='white', ax=ax)
        edge_labels = {(n1, n2): f"{dat['dist']:.1f}" for n1, n2, dat in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        # battery levels
        y = -0.1
        xs = [0, sc.N_w * X_OFFSET]
        vals = [params.B_start[d], params.B_min[d, -1]]
        for x, val in zip(xs, vals):
            ax.text(x, y, f"{val * 100:.1f}%", color='red', fontdict={"size": 6}, ha='center', backgroundcolor='white')

    plt.savefig(fname, bbox_inches='tight')


def draw_schedule(sc, model, fname):
    height = (sc.N_s + 1) * sc.N_d
    width = sc.N_w * 2
    _, axes = plt.subplots(nrows=sc.N_d, ncols=1, figsize=(width, height), sharey=True, sharex=True)
    if sc.N_d == 1:
        axes = [axes]

    for d in range(sc.N_d):
        g, pos = as_graph(sc, d)
        path = [p.tolist().index(1) for p in model.P_np[d].transpose()]
        g_path = graph_from_path(g, path)

        ax = axes[d]
        drawn_nodes = nx.draw_networkx_nodes(g, pos, ax=ax)
        drawn_nodes.set_linestyle(":")
        drawn_nodes.set_linewidth(0.5)
        drawn_nodes.set_alpha(0.3)
        drawn_nodes.set_color("white")
        drawn_nodes.set_edgecolor("black")
        nx.draw_networkx_edges(g, pos, width=0.5, style=":", alpha=0.3, edge_color='black', ax=ax)

        nx.draw_networkx_nodes(g_path, pos, ax=ax)  # solid
        node_labels = {node: dat['label'] for node, dat in g.nodes(data=True)}
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=6, font_color='white', ax=ax)
        nx.draw_networkx_edges(g_path, pos, edge_color='red', ax=ax)

        # b_star
        for w in model.w:
            node_idx = f'w{w}'
            b_star = model.b_star(d, w)
            if type(b_star) not in [np.int64, np.float64]:
                b_star = b_star()
            x_node, y_node = pos[node_idx]
            x = x_node
            y = y_node - 0.06
            ax.text(x, y, f"{b_star * 100:.1f}%", color='red', fontdict={"size": 6}, ha='center', backgroundcolor='white')

        # b_min
        for w_s, p in zip(model.w_s, path):
            node_idx = f"w{w_s + 1}p{p}"
            b_min = model.b_min(d, w_s)
            if type(b_min) not in [np.int64, np.float64]:
                b_min = b_min()
            x_node, y_node = pos[node_idx]
            x = x_node
            y = y_node - 0.04
            ax.text(x, y, f"{b_min * 100:.1f}%", color='red', fontdict={"size": 4}, ha='center')

        # b_plus
        for w_s, p in zip(model.w_s, path):
            node_idx = f"w{w_s + 1}p{p}"
            b_plus = model.b_plus(d, w_s)
            if type(b_plus) not in [np.int64, np.float64]:
                b_plus = b_plus()
            x_node, y_node = pos[node_idx]
            x = x_node
            y = y_node + 0.04
            ax.text(x, y, f"{b_plus * 100:.1f}%", color='green', fontdict={"size": 4}, ha='center')

        ax.axis('off')

    plt.savefig(fname, bbox_inches='tight')


def graph_from_path(g, path):
    nodes = ['w0']
    for wp_idx, p in zip(range(1, len(path) + 1), path):
        path_node = f'w{wp_idx}p{p}'
        nodes.append(path_node)

        wp_node = f"w{wp_idx}"
        nodes.append(wp_node)

    gsub = nx.subgraph(g, nodes)

    return gsub
