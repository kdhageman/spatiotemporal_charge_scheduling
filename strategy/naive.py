import copy
import logging

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
        self.logger = logging.getLogger(__name__)

    def simulate(self):
        decisions = [[] for _ in range(self.scenario.N_d)]  # P
        waiting_times = [[] for _ in range(self.scenario.N_d)]  # W
        charging_times = [[] for _ in range(self.scenario.N_d)]  # C
        status = ['at_waypoint'] * self.scenario.N_d  # status of each drone
        station_in_use = [0] * self.scenario.N_s  #

        t_curs = np.array(self.scenario.N_d * [0]).astype(np.float64)  # timestamp for each drone
        b_curs = copy.copy(self.parameters['B_start'])  # current battery charges for each drone
        waypoint_cur = np.array(self.scenario.N_d * [0])  # current waypoint for each drone
        finished = np.array(self.scenario.N_d * [False])  # finished state for each drone
        waited = np.array(self.scenario.N_d * [0]).astype(np.float64)

        while not _is_finished(finished):
            d = _next_drone(t_curs, finished)  # select next drone
            w_cur = waypoint_cur[d]  # current waypoint
            ts = t_curs[d] # current timestamp

            if status[d] == 'at_waypoint':
                # if station reachable from next waypoint, go directly to next waypoint
                # otherwise, move to closest charging station
                pos_cur = self.scenario.positions_w[d][w_cur]
                pos_next = self.scenario.positions_w[d][w_cur + 1]
                dist_to_next_wp = dist3(pos_cur, pos_next)

                dist_stations = []
                for pos_st in self.scenario.positions_S:
                    dist_stations.append(
                        dist3(pos_next, pos_st)
                    )
                dist_closest_st = min(dist_stations)

                # calculate time passing
                t_wp = dist_to_next_wp / self.parameters['v'][d]
                t_st = dist_closest_st / self.parameters['v'][d]

                # calculate depletion
                depletion_wp = t_wp * self.parameters['r_deplete'][d]
                depletion_st = t_st * self.parameters['r_deplete'][d]
                depletion_total = depletion_wp + depletion_st

                if b_curs[d] - depletion_total > 0:
                    # move directly to next waypoint
                    t_curs[d] += t_wp
                    b_curs[d] -= depletion_wp
                    waypoint_cur[d] = w_cur + 1
                    decision = [0] * self.scenario.N_s + [1]
                    decisions[d].append(decision)
                    waiting_times[d].append(0)
                    charging_times[d].append(0)
                    self.logger.info(f"[{ts:.2f}] drone [{d}] is moving from waypoint {w_cur} to waypoint {w_cur + 1}")
                else:
                    # move via closest charging station
                    dist_stations = []
                    for pos_st in self.scenario.positions_S:
                        dist_stations.append(
                            dist3(pos_cur, pos_st)
                        )
                    station = np.argmin(dist_stations)
                    dist_closest_st = dist_stations[station]
                    t_st = dist_closest_st / self.parameters['v'][d]
                    depletion_st = t_st * self.parameters['r_deplete'][d]

                    t_curs[d] += t_st
                    b_curs[d] -= depletion_st
                    decision = [0] * (self.scenario.N_s + 1)
                    decision[station] = 1
                    decisions[d].append(decision)

                    status[d] = 'at_station'
                    self.logger.info(f"[{ts:.2f}] drone [{d}] is moving from waypoint {w_cur} to station {station} to charge")
            elif status[d] == 'at_station':
                # wait if necessary, charge otherwise
                station = decisions[d][-1].index(1)
                if station_in_use[station]:
                    # wait
                    t_wait = station_in_use[station] + (np.random.rand() / 1000)
                    t_curs[d] += t_wait
                    waited[d] += t_wait
                    self.logger.info(f"[{ts:.2f}] drone [{d}] is waiting at station {station} for {t_wait:.1f}s (for a total of {waited[d]:.1f}s)")
                else:
                    # charge
                    perc_charge = self.parameters['B_max'] - b_curs[d]
                    t_charge = perc_charge / self.parameters['r_charge'][d]
                    t_curs[d] += t_charge
                    b_curs[d] = self.parameters['B_max']
                    charging_times[d].append(t_charge)
                    waiting_times[d].append(waited[d])
                    status[d] = 'finished_charging'
                    waited[d] = 0
                    station_in_use[station] = t_charge
                    self.logger.info(f"[{ts:.2f}] drone [{d}] is charging at station {station} for {t_charge:.1f}s")
            elif status[d] == 'finished_charging':
                # go to next waypoint
                station = decisions[d][-1].index(1)

                pos_st = self.scenario.positions_S[station]
                pos_wp = self.scenario.positions_w[d][w_cur]
                dist_to_next_wp = dist3(pos_st, pos_wp)
                t_wp = dist_to_next_wp / self.parameters['v'][d]
                depletion_wp = t_wp * self.parameters['r_deplete'][d]

                t_curs[d] += t_wp
                b_curs[d] -= depletion_wp
                waypoint_cur[d] += 1

                station_in_use[station] = 0
                status[d] = 'at_waypoint'
                self.logger.info(f"[{ts:.2f}] drone [{d}] is moving from station {station} to waypoint {waypoint_cur[d]}")

            if waypoint_cur[d] == self.scenario.N_w - 1 and status[d] == 'at_waypoint':
                # terminate
                self.logger.info(f"[{ts:.2f}] drone [{d}] is finished")
                finished[d] = True
        return np.array(decisions).transpose(0, 2, 1), np.array(waiting_times), np.array(charging_times)
