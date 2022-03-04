import numpy as np

from util.distance import dist


def get_T_N(N_d, N_s, N_w_s, positions_S, positions_w):
    T_n = []
    for d in range(N_d):
        matr = []
        waypoints = positions_w[d]
        for w_s in range(N_w_s):
            row = []
            cur_waypoint = waypoints[w_s]

            # distance to charging points
            for s in range(N_s):
                pos_S = positions_S[s]
                d = dist(cur_waypoint, pos_S)
                row.append(d)

            # distance to next waypoint
            next_waypoint = waypoints[w_s + 1]
            d = dist(cur_waypoint, next_waypoint)
            row.append(d)
            matr.append(row)
        T_n.append(matr)
    T_n = np.array(T_n).transpose(0, 2, 1)
    return T_n


def get_T_W(N_d, N_s, N_w_s, positions_S, positions_w):
    T_w = []
    for d in range(N_d):
        matr = []
        waypoints = positions_w[d]
        for w_s in range(N_w_s):
            row = []
            next_waypoint = waypoints[w_s + 1]

            # distance to charging points
            for s in range(N_s):
                pos_S = positions_S[s]
                d = dist(next_waypoint, pos_S)
                row.append(d)

            row.append(0)
            matr.append(row)
        T_w.append(matr)
    T_w = np.array(T_w).transpose(0, 2, 1)
    return T_w
