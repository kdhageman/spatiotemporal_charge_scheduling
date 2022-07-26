import logging

import numpy as np
from pyomo.opt import SolverFactory
from pyomo.util.infeasible import log_infeasible_constraints, log_close_to_bounds

from pyomo_models.multi_uavs import MultiUavModel
from simulate.node import ChargingStation, Waypoint, NodeType, AuxWaypoint
from simulate.parameters import Parameters
from simulate.uav import UavStateType
from util.decorators import timed
from util.distance import dist3
from util.exceptions import NotSolvableException
from util.scenario import Scenario


class Scheduler:
    def __init__(self, params: Parameters, sc: Scenario):
        """
        :param cb: the callback function that should be called whenever the scheduler generates a new schedule
        """
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.sc = sc
        self.offsets = [0] * sc.N_d

    def handle_event(self, event):
        """
        Allows the scheduler to handle simpy.Events
        :param event: simpy.Event
        """
        uav_id = event.value.uav.uav_id
        if event.value.node.node_type == NodeType.Waypoint:
            self.offsets[uav_id] += 1
        self._handle_event(event)

    def _handle_event(self, event):
        raise NotImplementedError

    @timed
    def schedule(self, start_positions: dict, batteries: dict, state_types: dict, uavs_to_schedule: list):
        """
        Creates a new schedule for the drones
        :return: optimal: True if the schedule is optimal, False otherwise
        :return: schedules: list of nodes for each schedules drone to follow
        """
        t_solve, schedules = self._schedule(start_positions, batteries, state_types, uavs_to_schedule)
        for d, schedule in schedules.items():
            offset = self.offsets[d]
            for node in schedule:
                if node.node_type == NodeType.Waypoint:
                    node.identifier = offset
                    offset += 1
        return t_solve, schedules

    def _schedule(self, start_positions: dict, batteries: dict, state_types: dict, uavs_to_schedule: list):
        raise NotImplementedError

    def n_remaining_waypoints(self, d):
        """
        Returns the remaining number of waypoints for the given UAV
        """
        return self.sc.N_w - self.offsets[d] - 1

    def remaining_waypoints(self, d):
        """
        Returns the list of remaining waypoints for the given UAV that need to be visited
        """
        return self.sc.positions_w[d][1 + self.offsets[d]:]


class ScenarioFactory:
    """
    Generates scenarios on the fly based on the current progress of UAVs (self.offsets)
    and the given strategy for sampling waypoints (W and sigma)
    """

    def __init__(self, scenario: Scenario, W: int, sigma: float):
        self.positions_S = scenario.positions_S
        self.positions_w = [wps[1:] for wps in scenario.positions_w]
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w

        self.W = W
        self.sigma = sigma

    def next(self, start_positions, offsets):
        """
        Returns the next scenario
        """
        positions_w = []
        D_N = []
        D_W = []

        for d, wps_src in enumerate(self.positions_w):
            wps_src_full = [start_positions[d]] + wps_src[offsets[d]:]
            while len(wps_src_full) < self.sigma * (self.W - 1) + 1:
                wps_src_full.append(wps_src_full[-1])

            wps = []
            D_N_matr = []
            D_W_matr = []

            n = 0
            while len(wps) < self.W:
                wp_hat = wps_src_full[n]
                wps.append(wp_hat)

                if len(wps) < self.W:
                    # calculate D_N
                    D_N_col = []
                    for pos_S in self.positions_S:
                        # distance to charging stations
                        distance = dist3(wp_hat, pos_S)
                        D_N_col.append(distance)

                    distance = 0
                    for i in range(n, n + self.sigma):
                        pos_a = wps_src_full[i]
                        pos_b = wps_src_full[i + 1]
                        distance += dist3(pos_a, pos_b)
                    D_N_col.append(distance)
                    D_N_matr.append(D_N_col)

                    # calculate D_W

                    D_W_col = []
                    for pos_S in self.positions_S:
                        # distance from charging station to next node
                        pos_wp = wps_src_full[n + 1]
                        distance = dist3(pos_S, pos_wp)

                        for i in range(n + 1, n + self.sigma):
                            pos_a = wps_src_full[i]
                            pos_b = wps_src_full[i + 1]
                            distance += dist3(pos_a, pos_b)
                        D_W_col.append(distance)
                    D_W_col.append(0)
                    D_W_matr.append(D_W_col)

                n += self.sigma
            D_N.append(D_N_matr)
            D_W.append(D_W_matr)
            positions_w.append(wps)
        sc = Scenario(positions_S=self.positions_S, positions_w=positions_w)

        D_N = np.array(D_N).transpose((0, 2, 1))
        sc.D_N = D_N

        D_W = np.array(D_W).transpose((0, 2, 1))
        sc.D_W = D_W

        return sc


class MilpScheduler(Scheduler):
    def __init__(self, params: Parameters, scenario: Scenario, solver=SolverFactory("gurobi")):
        super().__init__(params, scenario)
        self.sf = ScenarioFactory(self.sc, params.W, params.sigma)
        self.solver = solver

    def _handle_event(self, event):
        pass

    def _schedule(self, start_positions: dict, batteries: dict, state_types: dict, uavs_to_schedule: list):
        sc = self.sf.next(start_positions, self.offsets)

        # correct original parameters
        params = self.params.copy()
        params.B_start = np.array(list(batteries.values()))

        B_end = []
        for d, wps in enumerate(self.sc.positions_w):
            overall_pos_end_wp = self.sc.positions_w[d][-1]
            pos_end_wp = wps[-1]

            if overall_pos_end_wp == pos_end_wp:
                # last scheduled waypoint is the last of the mission, so no charging station needs to be
                # visited afterwards
                B_end.append(self.params.B_min[d])
            else:
                # it should be possible to reach the closest charging after this schedule,
                # because the mission is not finished yet afterwards
                dists_to_css = []
                for pos_cs in sc.positions_S:
                    dists_to_css.append(dist3(pos_end_wp, pos_cs))
                min_dist_to_cs = min(dists_to_css)
                B_end.append(
                    self.params.B_min[d] + min_dist_to_cs * self.params.r_deplete[d] / self.params.v[d]
                )
        params.B_end = np.array(B_end)

        W_zero_min = []
        for state_type in state_types:
            W_zero_min.append(self.params.epsilon if state_type == UavStateType.Charging else 0)
        params.W_zero_min = np.array(W_zero_min)

        model = MultiUavModel(scenario=sc, parameters=params.as_dict())
        solution = self.solver.solve(model)
        if solution['Solver'][0]['Status'] not in ['ok', 'aborted']:
            raise NotSolvableException(f"failed to solve model: {str(solution['Solver'][0])}")

        optimal = True if solution['Solver'][0]['Termination condition'] == 'optimal' else False

        # For debugging purposes
        # extract charging windows
        charging_windows = {}

        for d in range(sc.N_d):
            for w_s in range(sc.N_w_s):
                try:
                    station_idx = model.P_np[d, :-1, w_s].tolist().index(1)
                    t_s = model.T_s(d, w_s)()
                    t_e = model.T_e(d, w_s)()
                    if station_idx not in charging_windows:
                        charging_windows[station_idx] = {}
                    if d not in charging_windows[station_idx]:
                        charging_windows[station_idx][d] = []
                    charging_windows[station_idx][d].append((t_s, t_e))
                except ValueError:
                    # drone is NOT charging now
                    pass

        res = {}
        for d in uavs_to_schedule:
            start_node = AuxWaypoint(*start_positions[d])
            wps_full = [start_positions[d]] + self.remaining_waypoints(d)
            while len(wps_full) < self.params.sigma * (len(sc.positions_w[d]) - 1) + 1:
                wps_full.append(wps_full[-1])

            nodes = []
            for w_s_hat in model.w_s:
                n = model.P_np[d, :, w_s_hat].tolist().index(1)
                if n < model.N_s:
                    # visit charging station first
                    wt = max(model.W_np[d][w_s_hat], 0)  # for cases where the waiting time is negligibly negative
                    ct = max(model.C_np[d][w_s_hat], 0)  # for cases where the charging time is negligibly negative
                    nodes.append(
                        ChargingStation(*self.sc.positions_S[n], n, wt, ct)
                    )

                # add stride waypoints
                for i in range(w_s_hat * self.params.sigma + 1, w_s_hat * self.params.sigma + self.params.sigma):
                    pos = wps_full[i]
                    wp = Waypoint(*pos)
                    wp.strided = True
                    if not start_node.same_pos(wp):
                        nodes.append(wp)

                # add next waypoint
                pos = wps_full[self.params.sigma * (w_s_hat + 1)]
                wp = Waypoint(*pos)
                if not start_node.same_pos(wp):
                    nodes.append(wp)

            res[d] = nodes
        return optimal, res


class NaiveScheduler(Scheduler):
    _EPSILON = 0.0001

    def _handle_event(self, event):
        pass

    def _schedule(self, start_positions: dict, batteries: dict, state_types: dict, uavs_to_schedule: list):
        res = {}
        for d in uavs_to_schedule:
            if len(self.remaining_waypoints(d)) == 0:
                res[d] = []
                continue

            nodes = []

            dist_to_end = 0
            node_prev = start_positions[d]
            for pos in self.remaining_waypoints(d):
                dist = dist3(node_prev, pos)
                dist_to_end += dist
                node_prev = pos
            depletion_to_end = dist_to_end / self.params.v[d] * self.params.r_deplete[d]

            if batteries[d] - depletion_to_end > self.params.B_min[d]:
                # no need to charge
                pass
            else:
                # need to charge at some point, check if it should be now
                if len(self.remaining_waypoints(d)) == 1:
                    # only one waypoint remaining, so MUST charge
                    idx_station, dist_to_station = self.sc.nearest_station(start_positions[d])
                    dist_to_wp_from_station = self.sc.D_W[d, idx_station, self.offsets[d]]
                    depletion_to_wp_via_station = (dist_to_station + dist_to_wp_from_station) / self.params.v[d] * self.params.r_deplete[d]
                    ct = (depletion_to_wp_via_station + self.params.B_min[d] - batteries[d]) / self.params.r_charge[d] + NaiveScheduler._EPSILON
                    nodes.append(
                        ChargingStation(*self.sc.positions_S[idx_station], identifier=idx_station, wt=0, ct=ct)
                    )
                else:
                    # check if a station is reachable from next waypoint
                    dist_to_wp = self.sc.D_N[d, -1, self.offsets[d]]
                    idx_station, dist_to_station = self.sc.nearest_station(self.remaining_waypoints(d)[0])
                    dist_to_station_full = dist_to_wp + dist_to_station
                    depletion_to_station_full = dist_to_station_full / self.params.v[d] * self.params.r_deplete[d]
                    if batteries[d] - depletion_to_station_full < self.params.B_min[d]:
                        # must visit charging station next
                        idx_station, _ = self.sc.nearest_station(start_positions[d])

                        remaining_dist = 0
                        pos_prev = self.sc.positions_S[idx_station]
                        for pos in self.remaining_waypoints(d):
                            dist = dist3(pos_prev, pos)
                            remaining_dist += dist
                            pos_prev = pos
                        remaining_depletion = remaining_dist / self.params.v[d] * self.params.r_deplete[d]
                        if remaining_depletion + self.params.B_min[d] > self.params.B_max[d]:
                            ct = 'full'
                        else:
                            ct = (remaining_depletion + self.params.B_min[d] - batteries[d]) / self.params.r_charge[d] + NaiveScheduler._EPSILON
                        nodes.append(
                            ChargingStation(*self.sc.positions_S[idx_station], identifier=idx_station, wt=0, ct=ct)
                        )
            for pos in self.remaining_waypoints(d):
                wp = Waypoint(*pos)
                nodes.append(wp)
            res[d] = nodes
        return False, res
