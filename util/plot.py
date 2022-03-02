from itertools import product

from matplotlib import pyplot as plt, patches

from util.parameters import dist

w_colors = ['red', 'blue']

def battery_rectangles(pos_w, b, battery_width=0.15, battery_height=0.3, y_margin=0.1):
    x = pos_w[0] - (battery_width / 2)
    y = pos_w[1] - battery_height - y_margin
    outer_patch = patches.Rectangle((x, y), battery_width, battery_height, linewidth=1, edgecolor='k', facecolor='none')

    if b > .7:
        color = 'g'
    elif b > .3:
        color = 'y'
    else:
        color = 'r'

    inner_patch = patches.Rectangle((x, y), battery_width, battery_height * b, linewidth=1, edgecolor=color,
                                    facecolor=color)

    return outer_patch, inner_patch


# plot model WITHOUT solution
def plot_model(model, N_d, N_s, N_w, positions_S, positions_w, ax=None):
    if not ax:
        _, ax = plt.subplots()

    # draw lines between waypoints and S
    for d, s, w in product(range(N_d), range(N_s), range(N_w)):
        pos_s = positions_S[s]
        pos_w = positions_w[d][w]
        distance = dist(pos_s, pos_w)
        x = [pos_s[0], pos_w[0]]
        y = [pos_s[1], pos_w[1]]
        ax.plot(x, y, color='k', alpha=0.2)

        alpha = 0.7  # SET ALPHA

        x_text = pos_s[0] + alpha * (pos_w[0] - pos_s[0])
        y_text = pos_s[1] + alpha * (pos_w[1] - pos_s[1])
        ax.text(x_text, y_text, f"{distance:.1f}", color='k', alpha=0.4)

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
        ax.plot(x, y, marker='o', color=w_colors[d], markersize=10, markerfacecolor='white')

        # label waypoints
        for w in range(N_w):
            x_text = waypoints[w][0]
            y_text = waypoints[w][1] + 0.05
            ax.text(x_text, y_text, f"$w^{d + 1}_{w + 1}$", color=w_colors[d], fontsize=15)

        # label time between waypoints
        for w_s in range(N_w - 1):
            pos_w_s = waypoints[w_s]
            pos_w_d = waypoints[w_s + 1]
            distance = dist(pos_w_s, pos_w_d)

            alpha = 0.5  # SET ALPHA

            x_text = pos_w_s[0] + alpha * (pos_w_d[0] - pos_w_s[0])
            y_text = pos_w_s[1] + alpha * (pos_w_d[1] - pos_w_s[1]) + 0.05
            ax.text(x_text, y_text, f"{distance:.1f}", color=w_colors[d])

    # add margins to plot
    ax.margins(0.2)


# print model WITH solution
def plot_model_solved(model, N_d, N_s, N_w, positions_S, positions_w, ax=None, battery_params={}):
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
        # ax.scatter(x,y, marker='o', color=w_colors[d], markersize=10, markerfacecolor='white')
        ax.scatter(x, y, marker='o', color=w_colors[d], facecolor='white', s=70)

        # label waypoints
        for w in range(N_w):
            x_text = waypoints[w][0]
            y_text = waypoints[w][1] + 0.05
            ax.text(x_text, y_text, f"$w^{d + 1}_{w + 1}$", color=w_colors[d], fontsize=15)

    # draw lines
    for d in model.d:
        for w_s in model.w_s:
            do_charge = sum(model.x[d, :, w_s]())
            if not do_charge:
                # draw line between this and next point
                x, y = [(x, y) for x, y in zip(positions_w[d][w_s], positions_w[d][w_s + 1])]
                ax.plot(x, y, color=w_colors[d], zorder=-1)
            else:
                # figure out where to charge
                for s in model.s:
                    if model.x[d, s, w_s]():
                        x_s, y_s = positions_S[s]
                        x_src, y_src = positions_w[d][w_s]
                        x_dst, y_dst = positions_w[d][w_s + 1]

                        x = [x_src, x_s, x_dst]
                        y = [y_src, y_s, y_dst]

                        ax.plot(x, y, color=w_colors[d], zorder=-1)

                        # draw batteries
    battery_width = battery_params.get("battery_width", 0.15)
    battery_height = battery_params.get("battery_height", 0.3)
    y_margin = battery_params.get("y_margin", 0.1)
    for d in model.d:
        for w in model.w:
            pos_w = positions_w[d][w]
            b = model.b_arr[d, w]()
            outer, inner = battery_rectangles(pos_w, b, battery_width=battery_width, battery_height=battery_height,
                                              y_margin=y_margin)
            ax.add_patch(inner)
            ax.add_patch(outer)

    # add margins to plot
    ax.margins(0.2)
