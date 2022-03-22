from itertools import product

import yaml
from matplotlib import pyplot as plt
from yaml import Loader

from util import distance, constants


class Scenario:
    def __init__(self, fname):
        with open(fname, 'r') as f:
            doc = yaml.load(f, Loader=Loader)

        self.positions_S = []
        for charging_station in doc.get('charging_stations', []):
            x, y = charging_station['x'], charging_station['y']
            self.positions_S.append((x, y))

        self.positions_w = []
        for drone in doc.get('drones', []):
            waypoints = []
            for waypoint in drone.get('waypoints', []):
                x, y = waypoint['x'], waypoint['y']
                waypoints.append((x, y))
            self.positions_w.append(waypoints)

        self.N_d = len(self.positions_w)
        self.N_s = len(self.positions_S)
        self.N_w = len(self.positions_w[0])

    def plot(self, ax=None):
        if not ax:
            _, ax = plt.subplots()

        # draw lines between waypoints and S
        for d, s, w in product(range(self.N_d), range(self.N_s), range(self.N_w)):
            pos_s = self.positions_S[s]
            pos_w = self.positions_w[d][w]
            dist = distance.dist(pos_s, pos_w)
            x = [pos_s[0], pos_w[0]]
            y = [pos_s[1], pos_w[1]]
            ax.plot(x, y, color='k', alpha=0.2)

            alpha = 0.7  # SET ALPHA

            x_text = pos_s[0] + alpha * (pos_w[0] - pos_s[0])
            y_text = pos_s[1] + alpha * (pos_w[1] - pos_s[1])
            ax.text(x_text, y_text, f"{dist:.2f}", color='k', alpha=0.4)

        for s in range(self.N_s):
            x_s = self.positions_S[s][0]
            y_s = self.positions_S[s][1]
            ax.scatter(x_s, y_s, marker='s', color='k', s=100)

            # label station
            x_text = x_s + 0.1
            y_text = y_s
            ax.text(x_text, y_text, f"$s_{{{s + 1}}}$", fontsize=15)

        # plot waypoints
        for d in range(self.N_d):
            waypoints = self.positions_w[d]
            x = [i[0] for i in waypoints]
            y = [i[1] for i in waypoints]
            ax.plot(x, y, marker='o', color=constants.W_COLORS[d], markersize=10, markerfacecolor='white')

            # label waypoints
            for w in range(self.N_w):
                x_text = waypoints[w][0]
                y_text = waypoints[w][1] + 0.05
                ax.text(x_text, y_text, f"$w^{{{d + 1}}}_{{{w + 1}}}$", color=constants.W_COLORS[d], fontsize=15)

            # label time between waypoints
            for w_s in range(self.N_w - 1):
                pos_w_s = waypoints[w_s]
                pos_w_d = waypoints[w_s + 1]
                dist = distance.dist(pos_w_s, pos_w_d)

                alpha = 0.5  # SET ALPHA

                x_text = pos_w_s[0] + alpha * (pos_w_d[0] - pos_w_s[0])
                y_text = pos_w_s[1] + alpha * (pos_w_d[1] - pos_w_s[1]) + 0.05
                ax.text(x_text, y_text, f"{dist:.2f}", color=constants.W_COLORS[d])
