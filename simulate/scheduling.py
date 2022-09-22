import logging
import time
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import simpy
from pyomo.core import Suffix
from pyomo.opt import SolverFactory

from pyomo_models.multi_uavs import MultiUavModel
from simulate.node import ChargingStation, Waypoint, NodeType, AuxWaypoint, Node
from simulate.parameters import Parameters
from simulate.uav import UavStateType
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
        self.n_scheduled = 0

    def handle_event(self, event: simpy.Event):
        """
        Allows the scheduler to handle simpy.Events
        :param event: simpy.Event
        """
        uav_id = event.value.uav.uav_id
        if event.value.node.node_type == NodeType.Waypoint:
            self.offsets[uav_id] += 1

    def schedule(self, start_positions: Dict[int, List[float]], batteries: Dict[int, float], cs_locks: np.array, uavs_to_schedule: List[int]) -> Tuple[float, Tuple[bool, Dict[int, List[Node]]]]:
        """
        Creates a new schedule for the drones
        :return: optimal: True if the schedule is optimal, False otherwise
        :return: schedules: list of nodes for each schedules drone to follow
        """
        t_solve, (optimal, schedules) = self._schedule(start_positions, batteries, cs_locks, uavs_to_schedule)

        for d, schedule in schedules.items():
            # ignore schedule when no waypoints are remaining
            if self.n_remaining_waypoints(d) == 0:
                schedule = []

            # trim duplicate auxiliary waypoints at the end:
            # the first duplicate of the intended end position is considered to be a duplicate
            # so up to that node serves as the cut-off point (i.e. all nodes after it are discarded)
            reached_last_node = False
            last_node_sc = Waypoint(*self.sc.positions_w[d][-1])
            for idx, node in enumerate(schedule):
                if node == last_node_sc:
                    if not reached_last_node:
                        reached_last_node = True
                    else:
                        schedule = schedule[:idx]
                        break

            schedules[d] = schedule

        # tag the waypoints with their IDs
        for d, schedule in schedules.items():
            offset = self.offsets[d]
            try:
                for node in schedule:
                    if node.node_type == NodeType.Waypoint:
                        node._identifier = offset + 1
                        offset += 1
            except TypeError as e:
                pass

        self.n_scheduled += 1

        return t_solve, (optimal, schedules)

    def _schedule(self, start_positions: Dict[int, List[float]], batteries: Dict[int, float], cs_locks: np.array, uavs_to_schedule: List[int]) -> Tuple[bool, Dict[int, List[Node]]]:
        raise NotImplementedError

    def n_remaining_waypoints(self, d: int):
        """
        Returns the remaining number of waypoints for the given UAV
        """
        return self.sc.n_original_waypoints[d] - self.offsets[d] - 1

    def remaining_waypoints(self, d: int):
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
        self.sc = scenario
        self.positions_S = scenario.positions_S
        self.positions_w = [wps[1:] for wps in scenario.positions_w]
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w

        self.W = W
        self.sigma = sigma

    def next(self, start_positions: Dict[int, tuple], offsets: List[int]) -> Tuple[Scenario, List[float]]:
        """
        Returns the next scenario
        """
        positions_w = []
        for d, start_pos in start_positions.items():
            start = offsets[d]
            end = offsets[d] + self.W
            wps = self.positions_w[d][start:end]
            positions_w.append(wps)

        remaining_distances = []
        for d in range(self.N_d):
            remaining_distances.append(
                self.sc.D_N[d, -1, offsets[d] + self.W:].sum()
            )

        return Scenario(start_positions, self.positions_S, positions_w), remaining_distances


class MilpScheduler(Scheduler):
    def __init__(self, params: Parameters, scenario: Scenario, solver=SolverFactory("gurobi_ampl", solver_io='nl')):
        super().__init__(params, scenario)
        self.sf = ScenarioFactory(self.sc, params.W, params.sigma)
        self.solver = solver

    def _schedule(self, start_positions: Dict[int, List[float]], batteries: Dict[int, float], cs_locks: np.array, uavs_to_schedule: List[int]) -> Tuple[bool, Dict[int, List[Node]]]:
        sc, remaining_distances = self.sf.next(start_positions, self.offsets)

        for d, remaining_distance in enumerate(remaining_distances):
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] determined remaining distance for UAV [{d}] to be {remaining_distance:.1f}")

        # correct original parameters
        params = self.params.copy()
        params.remaining_distances = remaining_distances
        params.B_start = np.array(list(batteries.values()))

        B_end = []
        for d in range(self.sc.N_d):
            shortest_distance_to_station = 0

            # add distance to next anchor
            last_scheduled_idx = self.offsets[d] + params.W - 1  # last ID
            id_next_anchor = np.ceil(last_scheduled_idx / params.sigma).astype(int) * params.sigma
            if id_next_anchor < self.sf.sc.N_w:
                for n in range(last_scheduled_idx, id_next_anchor):
                    shortest_distance_to_station += self.sf.sc.D_N[d, -1, n]

                # add distance from anchor to station
                pos_N = self.sf.sc.positions_w[d][last_scheduled_idx]
                _, dist_to_station = sc.nearest_station(pos_N)
                shortest_distance_to_station += dist_to_station

            depletion_to_station = shortest_distance_to_station / params.v[d] * params.r_deplete[d]
            B_end.append(depletion_to_station + params.B_min[d])

        params.B_end = np.array(B_end)

        # TODO: let the current battery value on apply to arrival at the FIRST node
        B_min = []
        for d in range(self.sc.N_d):
            _, dist_to_nearest_station = sc.nearest_station(start_positions[d])
            depletion_to_nearest_station = dist_to_nearest_station / self.params.v[d] * self.params.r_deplete[d]
            B_min.append(min(batteries[d] - depletion_to_nearest_station, self.params.B_min[d]))
        params.B_min = np.array(B_min)

        params.W_zero_min = cs_locks

        t_start = time.perf_counter()
        model = MultiUavModel(sc=sc, params=params, anchor_offsets=self.offsets)
        # model.iis = Suffix(direction=Suffix.IMPORT)
        elapsed = time.perf_counter() - t_start
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] constructed MILP model in {elapsed:.2f}s")
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] model has M: {model.M:,.1f}s")
        for d in model.d:
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] has a maximum waiting time of:  {model.W_max[d]:,.1f}s")
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] has a maximum charging time of: {model.C_max[d]:,.1f}s")

        t_start = time.perf_counter()
        solution = self.solver.solve(model, tee=True)
        t_solve = time.perf_counter() - t_start

        if solution['Solver'][0]['Status'] not in ['ok', 'aborted']:
            # print("")
            # print("IIS Results")
            # for component, value in model.iis.items():
            #     print(f"{component.name} {component.ctype.__name__} {value}")
            raise NotSolvableException(f"failed to solve model: {str(solution['Solver'][0])}")

        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] solved model successfully in {t_solve:.2f}s!")
        # if self.n_scheduled == 1:
        #     raise Exception

        # for d in model.d:
        #     self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] scheduled path:\n{model.P_np[d]}")
        #     self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] scheduled waiting time:\n{model.W_np[d]}")
        #     self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] scheduled charging time:\n{model.C_np[d]}")

        for d in model.d:
            oc = model.oc(d)
            if type(oc) not in [np.int64, np.float64]:
                oc = oc()
            end_battery = model.b_star(d, model.N_w)
            if type(end_battery) not in [np.int64, np.float64]:
                end_battery = end_battery()
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] has a projected end battery of {end_battery * 100:.1f}% ({oc * 100:.1f}% more than necessary)")

        for d in model.d:
            lambda_charge = model.lambda_charge(d)
            if type(lambda_charge) not in [np.int64, np.float64]:
                lambda_charge = lambda_charge()
            lambda_move = model.lambda_move(d)
            if type(lambda_move) not in [np.int64, np.float64]:
                lambda_move = lambda_move()
            erd = model.erd(d)
            if type(erd) not in [np.int64, np.float64]:
                erd = erd()
            oc = model.oc(d)
            if type(oc) not in [np.int64, np.float64]:
                oc = oc()
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] is penalized by {lambda_move:.2f}s (moving) and {lambda_charge:.2f}s (charging) [=({erd:.1f} - {oc:.1f}) / {model.r_charge[d]}]")

        for d in model.d:
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] UAV [{d}] has a projected mission execution time of {model.E(d)():.2f}s")

        optimal = True if solution['Solver'][0]['Termination condition'] == 'optimal' else False

        # For debugging purposes
        # extract charging windows
        charging_windows = {}

        for d in range(sc.N_d):
            for w_s in range(sc.N_w):
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
            nodes = []
            for w_s in model.w_s:
                n = model.P_np[d, :, w_s].tolist().index(1)
                if n < model.N_s:
                    # charging
                    wt = max(model.W_np[d, w_s], 0)
                    ct = max(model.C_np[d, w_s], 0)
                    node = ChargingStation(*sc.positions_S[n], n, wt, ct)
                    nodes.append(node)
                node = Waypoint(*sc.waypoints(d)[w_s + 1])
                nodes.append(node)
            res[d] = nodes
        return t_solve, (optimal, res)


class NaiveScheduler(Scheduler):
    _EPSILON = 0.0001

    def _schedule(self, start_positions: Dict[int, List[float]], batteries: Dict[int, float], state_types: Dict[int, UavStateType], uavs_to_schedule: List[int]) -> Tuple[bool, Dict[int, List[Node]]]:
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
        return float(0), (False, res)
