import numpy as np

from util.distance import dist3


def _next_drone(t_curs, finished):
    idx_sorted = np.argsort(t_curs)
    for idx in idx_sorted:
        if not finished[idx]:
            return idx


def _is_finished(finished):
    return sum(1 - finished) == 0


class NaiveStrategy:
    def __init__(self, scenario, parameters):
        self.scenario = scenario
        self.parameters = parameters

    def simulate(self):
        decisions = [[] for _ in range(self.scenario.N_d)]
        waiting_times = [[] for _ in range(self.scenario.N_d)]
        charging_times = [[] for _ in range(self.scenario.N_d)]

        t_curs = np.array(self.scenario.N_d * [0])
        waypoint_cur = np.array(self.scenario.N_d * [0])
        b_curs = self.parameters['B_start']
        finished = np.array(self.scenario.N_d * [False])

        while not _is_finished(finished):
            d = _next_drone(t_curs, finished)
            w_s = waypoint_cur[d]
            w_next = w_s + 1

            # find charge at next waypoint
            pos_cur = self.scenario.positions_w[d][w_s]
            print(d)
            pos_next = self.scenario.positions_w[d][w_next]
            dist_to_next = dist3(pos_cur, pos_next)
            t_to_next_waypoint = dist_to_next / self.parameters['v'][d]
            depletion = t_to_next_waypoint * self.parameters['r_deplete'][d]

            b_at_next = b_curs[d] - depletion

            # check if any charging station is reachable
            distances_to_stations = []
            for pos_station in self.scenario.positions_S:
                dist = dist3(pos_next, pos_station)
                distances_to_stations.append(dist)
            min_dist = min(distances_to_stations)
            depletion = min_dist / self.parameters['v'][d] * self.parameters['r_deplete'][d]
            b_at_station = b_at_next - depletion

            if b_at_station > 0:
                # move to next waypoint
                decision = [0] * self.scenario.N_s + [1]
                decisions[d].append(decision)
                waiting_times[d].append(0)
                charging_times[d].append(0)
                t_curs[d] = t_curs[d] + t_to_next_waypoint
                b_curs[d] = b_at_next
            else:
                # move to closest charging station
                distances_to_stations = []
                for pos_station in self.scenario.positions_S:
                    dist = dist3(pos_cur, pos_station)
                    distances_to_stations.append(dist)
                idx_min = np.argmin(distances_to_stations)
                dist_min = distances_to_stations[idx_min]
                t_min = dist_min / self.parameters['v'][d]
                depletion = t_min * self.parameters['r_deplete'][d]
                b_at_station = b_curs[d] - depletion
                time_to_full = (self.parameters['B_max'] - b_at_station) / self.parameters['r_charge']

                # todo: check if in use

                decision = [0] * (self.scenario.N_s + 1)
                decision[idx_min] = 1
                waiting_times.append(0)
                charging_times.append(0)

                pass

            if waypoint_cur[d] == self.scenario.N_w - 1:
                # reached last waypoint; no more decisions to make
                finished[d] = True

            waypoint_cur[d] = waypoint_cur[d] + 1
