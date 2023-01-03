from itertools import product
from typing import Dict, List, Tuple

import jsons
import numpy as np
import yaml
from matplotlib import pyplot as plt
from yaml import Loader

from simulate.util import gen_colors, gen_linestyles
from util import distance
from util.distance import dist3


class Scenario:
    def __init__(self, start_positions: list, positions_S: list, positions_w: list, wp_max: int=None, anchors: list = None, offsets: list = None, waypoint_ids: list = None, n_parent_anchors: list = None, source_file=None):
        """
        :param positions_S: list of charging point positions (x,y,z coordinates)
        :param positions_w: list of list of waypoint positions (x,y,z coordinates)
        :param waypoint_ids: only for plotting purposes
        """
        self.start_positions = []
        for pos in start_positions:
            if len(pos) == 2:
                pos = (pos[0], pos[1], 0)
            self.start_positions.append(pos)
        self.positions_S = []
        for pos in positions_S:
            if len(pos) == 2:
                pos = (pos[0], pos[1], 0)
            self.positions_S.append(pos)
        self.N_w = max([len(l) for l in positions_w])
        self.positions_w = []
        for d, l in enumerate(positions_w):
            waypoints = []
            for wp in l:
                if len(wp) == 2:
                    wp = (wp[0], wp[1], 0)
                waypoints.append(wp)
            if waypoints:
                padding_val = waypoints[-1]
            else:
                padding_val = start_positions[d]
            padcount = self.N_w - len(l)
            self.positions_w.append(waypoints + [padding_val] * padcount)

        self.N_d = len(self.positions_w)
        self.N_s = len(self.positions_S)
        self.n_original_waypoints = [len(l) for l in positions_w]

        # anchors
        if not anchors:
            anchors = []
            for d in range(self.N_d):
                anchors.append(list(range(self.N_w)))
        self.anchors = anchors
        self.N_w_anchors = max([len(a) for a in anchors])

        # offsets
        if not offsets:
            offsets = [0] * self.N_d
        self.offsets = offsets

        if not wp_max:
            wp_max = self.N_w
        self.wp_max = wp_max

        # waypoint ids
        if not waypoint_ids:
            waypoint_ids = []
            for d in range(self.N_d):
                waypoint_ids.append(
                    np.minimum(self.offsets[d] + np.arange(1, self.N_w + 1), self.wp_max).tolist()
                )
        self.waypoint_ids = waypoint_ids

        # n parent anchors
        if not n_parent_anchors:
            n_parent_anchors = self.anchors
        self.n_parent_anchors = n_parent_anchors

        # calculate distance matrices
        self.D_N = self._get_D_N()
        self.D_W = self._get_D_W()

        self.source_file = source_file

    def waypoints(self, d):
        return [self.start_positions[d]] + self.positions_w[d]

    def _get_D_N(self):
        T_n = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.waypoints(d)
            for w_s in range(self.N_w):
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
        if self.N_w == 0:
            # no waypoints to visit
            T_n = None
        else:
            T_n = np.array(T_n).transpose(0, 2, 1)
        return T_n

    def _get_D_W(self):
        T_w = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.waypoints(d)
            for w_s in range(self.N_w):
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
        if self.N_w == 0:
            # no waypoints to visit
            T_w = None
        else:
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

        start_positions = []
        positions_w = []
        drones = doc.get('drones', [])

        for drone in drones:
            waypoints = []
            for i, wp in enumerate(drone.get('waypoints', [])):
                x, y, z = wp['x'], wp['y'], wp.get('z', 0)
                if i == 0:
                    # start position
                    start_positions.append((x, y, z))
                else:
                    waypoints.append((x, y, z))
            positions_w.append(waypoints)

        return Scenario(start_positions, positions_S, positions_w)

    def nearest_station(self, pos) -> Tuple[int, float]:
        """
        Returns the distance and index of the nearest charging station for the given position
        :param pos: three-dimensional position
        :return: index, distance
        """
        distances = []
        for pos_s in self.positions_S:
            distances += [dist3(pos, pos_s)]
        idx = np.argmin(distances)
        return idx, distances[idx]

    def nearest_station_to_start(self, d) -> Tuple[int, float]:
        """
        Returns the distance and index of the nearest charging stations for the start position of the given drone 'd'
        :param d:
        :return: index, distance
        """
        idx = self.D_N[d, :-1, 0].argmin()
        dist = self.D_N[d, idx, 0]
        return idx, dist

    def bounding_box(self):
        """
        Returns a (xmin, xmax, ymin, ymin) tuple that represents the bounding box around the positions (= waypoints and charging stations) in the scenario
        """
        X = [pos[0] for pos in self.positions_S]
        Y = [pos[1] for pos in self.positions_S]

        for l in self.positions_w:
            X += [pos[0] for pos in l]
            Y += [pos[1] for pos in l]

        X += [pos[0] for pos in self.start_positions]
        Y += [pos[1] for pos in self.start_positions]

        return min(X), max(X), min(Y), max(Y)

    def is_at_charging_station(self, pos):
        """
        Returns whether the given position is a charging station
        """
        if len(pos) == 2:
            pos = pos[0], pos[1], 0
        pos = tuple(pos)
        return pos in self.positions_S

    def collapse(self):
        D_N = []
        D_W = []
        # calculate distance matrices
        for d in range(self.N_d):
            if len(self.anchors[d]) == 0:
                D_N.append(self.D_N[d])
                D_W.append(self.D_W[d])
                continue
            D_N_d = []
            D_W_d = []
            last_anchor = -1
            for a in self.anchors[d]:
                val = self.D_N[d, :, a].copy()
                if last_anchor != a:
                    val += self.D_N[d, -1, last_anchor + 1:a].sum()
                D_N_d.append(val)
                D_W_d.append(self.D_W[d, :, a].copy())
                last_anchor = a
            # adjust last D_w for remaining distance after last anchor
            D_W_d[-1] += self.D_N[d, -1, last_anchor + 1:].sum()

            # pad for cases with fewer anchors
            if len(D_N_d) != self.N_w_anchors:
                D_W_d.append(np.array([1] * self.N_s + [0]))
                D_N_d.append(np.array([1] * self.N_s + [0]))

            D_N.append(D_N_d)
            D_W.append(D_W_d)
        D_N = np.array(D_N).transpose(0, 2, 1)
        D_W = np.array(D_W).transpose(0, 2, 1)

        # calculate waypoint positions
        positions_w = []
        for d in range(self.N_d):
            l = []
            for a in self.anchors[d][:-1]:
                l.append(self.positions_w[d][a])
            while len(l) != self.N_w_anchors:
                l.append(self.positions_w[d][-1])
            positions_w.append(l)

        # calculate the waypoint ids
        waypoint_ids = []
        for d in range(self.N_d):
            l = []
            for a in self.offsets[d] + np.array(self.anchors[d][:-1]):
                l.append(a + 1)
            l.append(self.offsets[d] + self.N_w + 1)
            waypoint_ids.append(np.minimum(l, self.wp_max))

        sc = Scenario(start_positions=self.start_positions, positions_S=self.positions_S, positions_w=positions_w, waypoint_ids=waypoint_ids, n_parent_anchors=[len(l) for l in self.anchors])
        sc.D_N = D_N
        sc.D_W = D_W
        return sc

    def plot(self, ax=None, draw_distances=True, greyscale=False):
        if not ax:
            _, ax = plt.subplots()

        colors = gen_colors(self.N_d)
        if greyscale:
            # TODO: implement grey scale plotting
            pass
        linestyles = gen_linestyles(self.N_d)

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
            ax.plot(x, y, color=colors[d], linestyle=linestyles[d], zorder=-1)
            ax.scatter(x, y, marker='o', color=colors[d], facecolor='white', s=70)
            ax.scatter(x[:1], y[:1], marker='o', color=colors[d], facecolor=colors[d], s=100)

            # label waypoints
            for w in range(self.N_w):
                x_text = waypoints[w][0]
                y_text = waypoints[w][1] + 0.05
                ax.text(x_text, y_text, f"$w^{{{d + 1}}}_{{{w + 1}}}$", color=colors[d], fontsize=15)

            # label time between waypoints
            if draw_distances:
                for w_s in range(self.N_w - 1):
                    pos_w_s = waypoints[w_s]
                    pos_w_d = waypoints[w_s + 1]
                    dist = distance.dist3(pos_w_s, pos_w_d)

                    alpha = 0.5  # SET ALPHA

                    x_text = pos_w_s[0] + alpha * (pos_w_d[0] - pos_w_s[0])
                    y_text = pos_w_s[1] + alpha * (pos_w_d[1] - pos_w_s[1]) + 0.05
                    ax.text(x_text, y_text, f"{dist:.2f}", color=colors[d])


class ScenarioFactory:
    """
    Generates scenarios on the fly based on the current progress of UAVs (self.offsets)
    and the given strategy for sampling waypoints (W and sigma)
    """

    def __init__(self, scenario: Scenario, W_hat: int, sigma: float):
        self.sc = scenario
        self.positions_S = scenario.positions_S
        self.positions_w = [wps for wps in scenario.positions_w]
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w

        self.W_hat = W_hat
        self.sigma = sigma

    def anchors(self):
        res = [0]

        n = self.sigma
        while n < self.N_w:
            res.append(n)
            n += self.sigma
        return res

    def next(self, start_positions: Dict[int, tuple], offsets: List[int]) -> Tuple[Scenario, List[float]]:
        """
        Returns the next scenario
        """
        positions_w = []
        for d in range(self.N_d):
            start = offsets[d]
            end = offsets[d] + self.W_hat
            wps = self.positions_w[d][start:end]
            if not wps:
                wps = [start_positions[d]]
            while len(wps) < self.W_hat:
                wps.append(wps[-1])
            positions_w.append(wps)

        # calculate remaining distances
        rhos = []
        for d in range(self.N_d):
            rho = self.sc.D_N[d, -1, offsets[d] + self.W_hat:].sum()
            rhos.append(rho)

        anchors = []
        for d in range(self.sc.N_d):
            anchors_d = np.array(self.anchors()) - (offsets[d] % self.sigma)
            anchors_trimmed_d = [a for a in anchors_d if 0 <= a < self.W_hat]
            # compensate for drones that are currently charging
            if self.sc.is_at_charging_station(start_positions[d]) and 0 not in anchors_trimmed_d:
                anchors_trimmed_d = [0] + anchors_trimmed_d
            anchors.append(anchors_trimmed_d)

        return Scenario(start_positions, self.positions_S, positions_w, wp_max=self.N_w, anchors=anchors, offsets=offsets), rhos


def scenario_serializer(obj: Scenario, *args, **kwargs):
    return dict(
        waypoints=obj.positions_w,
        start_positions=obj.start_positions,
        charging_stations=obj.positions_S,
        nr_drones=obj.N_d,
        nr_charging_stations=obj.N_s,
        nr_waypoints=obj.N_w,
        source_file=obj.source_file,
        D_N=obj.D_N.tolist(),
        D_W=obj.D_W.tolist(),
    )


jsons.set_serializer(scenario_serializer, Scenario)
