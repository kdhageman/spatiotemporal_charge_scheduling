from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from util.scenario import _W_COLORS

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


def plot_battery_over_time(model, N_s, d, ax=None):
    if not ax:
        _, ax = plt.subplots()

    x = [0]  # covered distances
    y = [model.b_arr[d, 0]()]  # battery charges
    x_ticks = [0]
    x_labels = [f'$w_{{{1}}}$']
    rectangles = []

    total_dist = 0
    for w_s in model.w_s:
        p = model.P[d, :, w_s]()
        for n in model.n:
            if p[n] == 1:
                if n == N_s:
                    # next waypoint
                    dist_to_next_node = model.T_N[d, n, w_s]
                    total_dist += dist_to_next_node
                    x.append(total_dist)
                    x_ticks.append(total_dist)
                    x_labels.append(f'$w_{{{w_s + 2}}}$')
                    y.append(model.b_arr[d, w_s + 1]())
                else:
                    # charging
                    # x-value
                    dist_to_next_node = model.T_N[d, n, w_s]
                    total_dist += dist_to_next_node
                    x.append(total_dist)
                    x_ticks.append(total_dist)
                    x_rect = total_dist
                    total_dist += model.D[d, w_s]()
                    x.append(total_dist)
                    dist_from_station = model.T_W[d, n, w_s]
                    total_dist += dist_from_station
                    x.append(total_dist)
                    x_ticks.append(total_dist)

                    # x-label
                    x_labels.append(f'$s_{{{n + 1}}}$')
                    x_labels.append(f'$w_{{{w_s + 2}}}$')

                    # y-value
                    y.append(model.b_min[d, w_s]())
                    y.append(model.b_plus[d, w_s]())
                    y.append(model.b_arr[d, w_s + 1]())

                    # rectangle
                    width_rect = model.D[d, w_s]()
                    height_rect = 1
                    y_rect = 0
                    rectangles.append(
                        Rectangle(
                            (x_rect, y_rect),
                            width_rect,
                            height_rect,
                            color='green',
                            linewidth=None,
                            alpha=0.2,
                            zorder=-1
                        )
                    )

    ax.plot(x, y)

    for rect in rectangles:
        ax.add_patch(rect)
    ax.set_ylim([0, 1])
    ax.set_xticks(x_ticks, x_labels)
    ax.set_xlabel("Arrival at node")
    ax.set_ylabel("Battery")