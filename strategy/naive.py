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
        status = ['at_waypoint'] * self.scenario.N_d
        station_in_use = [0] * self.scenario.N_s

        t_curs = np.array(self.scenario.N_d * [0])
        waypoint_cur = np.array(self.scenario.N_d * [0])
        b_curs = self.parameters['B_start']
        finished = np.array(self.scenario.N_d * [False])

        while not _is_finished(finished):
            d = _next_drone(t_curs, finished)
            w_s = waypoint_cur[d]
            w_next = w_s + 1

            if status[d] == 'at_waypoint':
                # determine where to move next

                # find charge at next waypoint
                pos_cur = self.scenario.positions_w[d][w_s]
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

                    waypoint_cur[d] = waypoint_cur[d] + 1

                    print(f"drone {d} is moving from waypoint {w_s} to waypoint {w_next}")
                else:
                    # move to closest charging station
                    distances_to_stations = []
                    for pos_station in self.scenario.positions_S:
                        dist = dist3(pos_cur, pos_station)
                        distances_to_stations.append(dist)
                    station = np.argmin(distances_to_stations)
                    dist_min = distances_to_stations[station]
                    t_min = dist_min / self.parameters['v'][d]
                    depletion = t_min * self.parameters['r_deplete'][d]

                    decision = [0] * (self.scenario.N_s + 1)
                    decision[station] = 1
                    decisions[d].append(decision)
                    b_curs[d] = b_curs[d] - depletion
                    t_curs[d] = t_curs[d] + t_min
                    status[d] = 'arrived_at_station'

                    print(f"drone {d} is moving from waypoint {w_s} to station {station}")

            elif status[d] in ['arrived_at_station', 'finished_waiting']:
                station = decisions[d][-1].index(1)
                if station_in_use[station]:
                    # must wait first
                    t_wait = station_in_use[station] - t_curs[d]
                    waiting_times[d].append(t_wait)
                    t_curs[d] += t_wait
                    status[d] = 'finished_waiting'
                    print(f"drone {d} is waiting at station {station} after {t_wait:.1f}s")
                else:
                    # start charging without waiting
                    t_charge = (self.parameters['B_max'] - b_curs[d]) / self.parameters['r_charge'][d]
                    b_curs[d] = self.parameters['B_max']
                    t_curs[d] = t_curs[d] + t_charge
                    station_in_use[station] = t_curs[d]
                    status[d] = 'finished_charging'
                    waiting_times[d].append(0)
                    charging_times[d].append(t_charge)
                    print(f"drone {d} is charging at station {station}")
            elif status[d] == 'finished_charging':
                station = decisions[d][-1].index(1)
                next_waypoint = waypoint_cur[d] + 1
                pos_cur = self.scenario.positions_S[station]
                pos_next = self.scenario.positions_w[d][next_waypoint]
                dist = dist3(pos_cur, pos_next)
                t = dist / self.parameters['v'][d]
                depletion = t_to_next_waypoint * self.parameters['r_deplete'][d]
                b_curs[d] -= depletion
                t_curs[d] += t

                station_in_use[station] = False
                waypoint_cur[d] = next_waypoint
                status[d] = 'at_waypoint'
                print(f"drone {d} is moving from station {station} to waypoint {next_waypoint}")

            if waypoint_cur[d] == self.scenario.N_w - 1:
                finished[d] = True

        pass
