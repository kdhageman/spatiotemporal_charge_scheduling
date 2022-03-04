import numpy as np

from util.distance import dist


def get_T_W(N_d, N_w, positions_w):
    T_W = []  # d x w_s
    for d in range(N_d):
        row = []
        for w_s in range(N_w - 1):
            a = positions_w[d][w_s]
            b = positions_w[d][w_s + 1]
            row.append(dist(a, b))
        T_W.append(row)
    T_W = np.array(T_W)

    return T_W


def get_T_S(N_d, N_s, N_w, positions_S, positions_w):
    T_S = []  # d x s x w
    for d in range(N_d):
        matr = []
        for s in range(N_s):
            a = positions_S[s]

            row = []
            for w in range(N_w):
                b = positions_w[d][w]
                row.append(dist(a, b))
            matr.append(row)
        T_S.append(matr)
    T_S = np.array(T_S)

    return T_S
