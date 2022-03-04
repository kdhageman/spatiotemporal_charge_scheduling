from matplotlib import pyplot as plt

from util.plot import _W_COLORS


# print model WITH solution
def plot_model_solved(model, positions_S, positions_w, ax=None, battery_params={}):
    N_d = len(positions_w)
    N_w = len(positions_w[0])
    N_s = len(positions_S)

    if not ax:
        _, ax = plt.subplots()

    # draw charge station(s)
    for s in range(N_s):
        x_s = positions_S[s][0]
        y_s = positions_S[s][1]
        ax.scatter(x_s, y_s, marker='s', color='k', s=100)

        # label station
        x_text = x_s + 0.1
        y_text = y_s
        ax.text(x_text, y_text, f"$s_{s + 1}$", fontsize=15)

    # draw waypoints and lines for each drone
    for d in range(N_d):
        waypoints = positions_w[d]
        x = [w[0] for w in waypoints[:N_w]]
        y = [w[1] for w in waypoints[:N_w]]
        # ax.scatter(x,y, marker='o', color=_w_colors[d], markersize=10, markerfacecolor='white')
        ax.scatter(x, y, marker='o', color=_W_COLORS[d], facecolor='white', s=70)

        # label waypoints
        for w in range(N_w):
            x_text = waypoints[w][0]
            y_text = waypoints[w][1] + 0.05
            ax.text(x_text, y_text, f"$w^{d + 1}_{w + 1}$", color=_W_COLORS[d], fontsize=15)

    # draw lines
    for d in model.d:
        for w_s in model.w_s:
            cur_waypoint = positions_w[d][w_s]
            next_waypoint = positions_w[d][w_s + 1]
            for n in model.n:
                if model.P[d, n, w_s]():
                    if n == N_s:
                        # directly to next waypoint
                        x = [cur_waypoint[0], next_waypoint[0]]
                        y = [cur_waypoint[1], next_waypoint[1]]
                    else:
                        # via charging station S
                        pos_S = positions_S[n]
                        x = [cur_waypoint[0], pos_S[0], next_waypoint[0]]
                        y = [cur_waypoint[1], pos_S[1], next_waypoint[1]]
                    ax.plot(x, y, _W_COLORS[d], linewidth=2, zorder=-1)

    # draw batteries
    # battery_width = battery_params.get("battery_width", 0.15)
    # battery_height = battery_params.get("battery_height", 0.3)
    # y_margin = battery_params.get("y_margin", 0.1)
    # for d in model.d:
    #     for w in model.w:
    #         pos_w = positions_w[d][w]
    #         b = model.b_arr[d, w]()
    #         outer, inner = battery_rectangles(pos_w, b, battery_width=battery_width, battery_height=battery_height,
    #                                           y_margin=y_margin)
    #         ax.add_patch(inner)
    #         ax.add_patch(outer)

    # add margins to plot
    ax.margins(0.2)