from itertools import product

import numpy as np
import yaml
from matplotlib import pyplot as plt
from yaml import Loader

from util import distance, constants
from util.distance import dist3


class Scenario:
    def __init__(self, positions_S: list, positions_w: list):
        """
        :param positions_S: list of charging point positions (x,y,z coordinates)
        :param positions_w: list of list of waypoint positions (x,y,z coordinates)
        """
        self.positions_S = []
        for pos in positions_S:
            if len(pos) == 2:
                pos = (pos[0], pos[1], 0)
            self.positions_S.append(pos)
        self.N_w = max([len(l) for l in positions_w])
        self.positions_w = []
        for l in positions_w:
            waypoints = []
            for wp in l:
                if len(wp) == 2:
                    wp = (wp[0], wp[1], 0)
                waypoints.append(wp)
            padding_val = waypoints[-1]
            padcount = self.N_w - len(l)
            self.positions_w.append(waypoints + [padding_val] * padcount)

        self.N_d = len(self.positions_w)
        self.N_s = len(self.positions_S)
        self.N_w_s = self.N_w - 1
        self.n_original_waypoints = [len(l) for l in positions_w]

        # calculate distance matrices
        self.D_N = self._get_D_N()
        self.D_W = self._get_D_W()

    def _get_D_N(self):
        T_n = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.positions_w[d]
            for w_s in range(self.N_w_s):
                row = []
                cur_waypoint = waypoints[w_s]

                # distance to charging points
                for s in range(self.N_s):
                    pos_S = self.positions_S[s]
                    d = dist3(cur_waypoint, pos_S)
                    row.append(d)

                # distance to next waypoint
                next_waypoint = waypoints[w_s + 1]
                d = dist3(cur_waypoint, next_waypoint)
                row.append(d)
                matr.append(row)
            T_n.append(matr)
        T_n = np.array(T_n).transpose(0, 2, 1)
        return T_n

    def _get_D_W(self):
        T_w = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.positions_w[d]
            for w_s in range(self.N_w_s):
                row = []
                next_waypoint = waypoints[w_s + 1]

                # distance to charging points
                for s in range(self.N_s):
                    pos_S = self.positions_S[s]
                    d = dist3(next_waypoint, pos_S)
                    row.append(d)

                row.append(0)
                matr.append(row)
            T_w.append(matr)
        T_w = np.array(T_w).transpose(0, 2, 1)
        return T_w

    @classmethod
    def from_file(cls, fname):
        with open(fname, 'r') as f:
            doc = yaml.load(f, Loader=Loader)

        positions_S = []
        for cs in doc.get('charging_stations', []):
            x, y, z = cs['x'], cs['y'], cs.get('z', 0)
            positions_S.append((x, y, z))

        positions_w = []
        drones = doc.get('drones', [])

        for drone in drones:
            waypoints = []
            for wp in drone.get('waypoints', []):
                x, y, z = wp['x'], wp['y'], wp.get('z', 0)
                waypoints.append((x, y, z))
            positions_w.append(waypoints)

        return Scenario(positions_S, positions_w)

    def nearest_station(self, pos):
        """
        Returns the distance and index of the nearest charging station for the given position
        :param pos:
        :return:
        """
        distances = []
        for pos_s in self.positions_S:
            distances += [dist3(pos, pos_s)]
        idx = np.argmin(distances)
        return idx, distances[idx]

    def bounding_box(self):
        """
        Returns a (xmin, xmax, ymin, ymin) tuple that represents the bounding box around the positions (= waypoints and charging stations) in the scenario
        """
        X = [pos[0] for pos in self.positions_S]
        Y = [pos[1] for pos in self.positions_S]

        for l in self.positions_w:
            X += [pos[0] for pos in l]
            Y += [pos[1] for pos in l]

        return min(X), max(X), min(Y), max(Y)

    def plot(self, ax=None, draw_distances=True):
        if not ax:
            _, ax = plt.subplots()

        if draw_distances:
            # draw lines between waypoints and S
            for d, s, w in product(range(self.N_d), range(self.N_s), range(self.N_w)):
                pos_s = self.positions_S[s]
                pos_w = self.positions_w[d][w]
                dist = distance.dist3(pos_s, pos_w)
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
            ax.plot(x, y, color=constants.W_COLORS[d], zorder=-1)
            ax.scatter(x, y, marker='o', color=constants.W_COLORS[d], facecolor='white', s=70)

            # label waypoints
            for w in range(self.N_w):
                x_text = waypoints[w][0]
                y_text = waypoints[w][1] + 0.05
                ax.text(x_text, y_text, f"$w^{{{d + 1}}}_{{{w + 1}}}$", color=constants.W_COLORS[d], fontsize=15)

            # label time between waypoints
            if draw_distances:
                for w_s in range(self.N_w - 1):
                    pos_w_s = waypoints[w_s]
                    pos_w_d = waypoints[w_s + 1]
                    dist = distance.dist3(pos_w_s, pos_w_d)

                    alpha = 0.5  # SET ALPHA

                    x_text = pos_w_s[0] + alpha * (pos_w_d[0] - pos_w_s[0])
                    y_text = pos_w_s[1] + alpha * (pos_w_d[1] - pos_w_s[1]) + 0.05
                    ax.text(x_text, y_text, f"{dist:.2f}", color=constants.W_COLORS[d])
