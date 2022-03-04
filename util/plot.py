from itertools import product

from matplotlib import pyplot as plt

from util import distance

_W_COLORS = ['red', 'blue']

def plot_model(positions_S, positions_w, ax=None):
    N_d = len(positions_w)
    N_w = len(positions_w[0])
    N_s = len(positions_S)

    if not ax:
        _, ax = plt.subplots()

    # draw lines between waypoints and S
    for d, s, w in product(range(N_d), range(N_s), range(N_w)):
        pos_s = positions_S[s]
        pos_w = positions_w[d][w]
        dist = distance.dist(pos_s, pos_w)
        x = [pos_s[0], pos_w[0]]
        y = [pos_s[1], pos_w[1]]
        ax.plot(x, y, color='k', alpha=0.2)

        alpha = 0.7  # SET ALPHA

        x_text = pos_s[0] + alpha * (pos_w[0] - pos_s[0])
        y_text = pos_s[1] + alpha * (pos_w[1] - pos_s[1])
        ax.text(x_text, y_text, f"{dist:.2f}", color='k', alpha=0.4)

    for s in range(N_s):
        x_s = positions_S[s][0]
        y_s = positions_S[s][1]
        ax.scatter(x_s, y_s, marker='s', color='k', s=100)

        # label station
        x_text = x_s + 0.1
        y_text = y_s
        ax.text(x_text, y_text, f"$s_{s + 1}$", fontsize=15)

    # plot waypoints
    for d in range(N_d):
        waypoints = positions_w[d]
        x = [i[0] for i in waypoints]
        y = [i[1] for i in waypoints]
        ax.plot(x, y, marker='o', color=_W_COLORS[d], markersize=10, markerfacecolor='white')

        # label waypoints
        for w in range(N_w):
            x_text = waypoints[w][0]
            y_text = waypoints[w][1] + 0.05
            ax.text(x_text, y_text, f"$w^{d + 1}_{w + 1}$", color=_W_COLORS[d], fontsize=15)

        # label time between waypoints
        for w_s in range(N_w - 1):
            pos_w_s = waypoints[w_s]
            pos_w_d = waypoints[w_s + 1]
            dist = distance.dist(pos_w_s, pos_w_d)

            alpha = 0.5  # SET ALPHA

            x_text = pos_w_s[0] + alpha * (pos_w_d[0] - pos_w_s[0])
            y_text = pos_w_s[1] + alpha * (pos_w_d[1] - pos_w_s[1]) + 0.05
            ax.text(x_text, y_text, f"{dist:.2f}", color=_W_COLORS[d])

    # add margins to plot
    ax.margins(0.2)