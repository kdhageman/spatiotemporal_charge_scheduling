import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def battery_over_time_from_base_model(base_model, d, ax=None, charge_bars=True):
    N_s = len(base_model.s)
    N_w_s = len(base_model.w_s)

    decisions = np.reshape(base_model.P[d, :, :](), (N_s + 1, N_w_s))
    wait_times = base_model.D[d, :]()

    return battery_over_time(
        decisions,
        wait_times,
        base_model.b_arr[d, :](),
        base_model.b_min[d, :](),
        base_model.b_plus[d, :](),
        base_model.T_N[d, :, :],
        base_model.T_W[d, :, :],
        ax=ax,
        charge_bars=charge_bars
    )


def battery_over_time(decisions, wait_times, b_arr, b_min, b_plus, T_N, T_W, ax=None, charge_bars=True):
    if not ax:
        _, ax = plt.subplots()

    next_waypoint_idx = decisions.shape[0] - 1

    x = [0]  # covered distances
    y = [b_arr[0]]  # battery charges
    x_ticks = [0]
    x_labels = [f'$w_{{{1}}}$']
    rectangles = []

    total_dist = 0
    for w_s in range(T_W.shape[1]):
        p = decisions[:, w_s]
        for n in range(len(p)):
            if p[n] == 1:
                if n == next_waypoint_idx:
                    # next waypoint
                    dist_to_next_node = T_N[n, w_s]
                    total_dist += dist_to_next_node
                    x.append(total_dist)
                    x_ticks.append(total_dist)
                    x_labels.append(f'$w_{{{w_s + 2}}}$')
                    y.append(b_arr[w_s + 1])
                else:
                    # charging
                    # x-value
                    dist_to_next_node = T_N[n, w_s]
                    total_dist += dist_to_next_node
                    x.append(total_dist)
                    x_ticks.append(total_dist)
                    x_rect = total_dist
                    total_dist += wait_times[w_s]
                    x.append(total_dist)
                    dist_from_station = T_W[n, w_s]
                    total_dist += dist_from_station
                    x.append(total_dist)
                    x_ticks.append(total_dist)

                    # x-label
                    x_labels.append(f'$s_{{{n + 1}}}$')
                    x_labels.append(f'$w_{{{w_s + 2}}}$')

                    # y-value
                    y.append(b_min[w_s])
                    y.append(b_plus[w_s])
                    y.append(b_arr[w_s + 1])

                    # rectangle
                    width_rect = wait_times[w_s]
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

    ax.plot(x, y, marker='o')

    if charge_bars:
        for rect in rectangles:
            ax.add_patch(rect)

    ax.set_ylim([0, 1])
    ax.set_xticks(x_ticks, x_labels, rotation=45)
    ax.set_xlabel("Arrival time at node")
    ax.set_ylabel("Battery")
