import logging

import numpy as np
import simpy
from matplotlib import pyplot as plt
from pyomo.opt import SolverFactory

from pyomo_models.multi_uavs import MultiUavModel
from simulate.node import ChargingStation, Waypoint, NodeType
from simulate.parameters import Parameters
from simulate.uav import UAV
from util.decorators import timed
from util.scenario import Scenario


class Schedule:
    def __init__(self, decisions: np.ndarray, charging_times: np.ndarray, waiting_times: np.ndarray):
        assert (decisions.ndim == 2)
        assert (charging_times.ndim == 1)
        assert (waiting_times.ndim == 1)

        self.decisions = decisions
        self.charging_times = charging_times
        self.waiting_times = waiting_times


class Environment:
    def distance(self, x):
        raise NotImplementedError

    def velocity(self, x):
        raise NotImplementedError

    def depletion(self, x):
        raise NotImplementedError


class DeterministicEnvironment(Environment):

    def distance(self, x):
        return x

    def velocity(self, x):
        return x

    def depletion(self, x):
        return x


class TimeStepper:
    def __init__(self, interval):
        self.timestep = 0
        self.interval = interval

    def _inc(self, _):
        self.timestep += 1

    def sim(self, env, callbacks=[]):
        while True:
            event = env.timeout(self.interval)
            for cb in callbacks:
                event.callbacks.append(cb)
            event.callbacks.append(self._inc)
            try:
                yield event
            except simpy.exceptions.Interrupt:
                break


class NotSolvableException(Exception):
    pass


class ScenarioFactory:
    """
    Creates new scenarios on the fly based on
    """

    def __init__(self, scenario: Scenario, W: int):
        self.original_start_pos = [wps[0] for wps in scenario.positions_w]
        self.positions_S = scenario.positions_S
        self.positions_w = [wps[1:] for wps in scenario.positions_w]
        self.offsets = [0] * scenario.N_d
        self.sc_orig = scenario
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w
        self.W = W

    def incr(self, d):
        """
        Increments the
        :param d: identifier of the UAV
        """
        self.offsets[d] += 1

    def next(self, start_positions):
        """
        Returns the next scenario
        """
        positions_w = []
        for i, wps in enumerate(self.positions_w):
            wps_truncated = [start_positions[i]] + wps[self.offsets[i]:self.offsets[i] + self.W - 1]
            while len(wps_truncated) < self.W:
                wps_truncated = [wps_truncated[0]] + wps_truncated
            positions_w.append(wps_truncated)
        return Scenario(positions_S=self.positions_S, positions_w=positions_w)


class Scheduler:
    def __init__(self, params: Parameters, scenario: Scenario):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.scenario = scenario

    @timed
    def schedule(self, solver=SolverFactory("gurobi")):
        """
        Return the most recent schedule
        :return: path to follow, dim = N_d x (W - 1) x (N_s + 1)
        :return: charging times, dim = N_d x (W - 1)
        :return: waiting times, dim = N_d x (W - 1)
        """
        model = MultiUavModel(scenario=self.scenario, parameters=self.params.as_dict())
        solution = solver.solve(model)
        if solution['Solver'][0]['Termination condition'] != 'optimal':
            raise NotSolvableException("non-optimal solution")

        res = []
        for d in model.d:
            start_pos = self.scenario.positions_w[d][0]
            nodes = []
            for w_s in model.w_s:
                n = model.P_np[d, :, w_s].tolist().index(1)
                if n < model.N_s:
                    # visit charging station first
                    ct = model.C_np[d][w_s]
                    wt = model.W_np[d][w_s]
                    nodes.append(
                        ChargingStation(*self.scenario.positions_S[n], n, wt, ct)
                    )
                wp = Waypoint(*self.scenario.positions_w[d][w_s + 1])
                if np.array_equal(wp.pos, start_pos):
                    wp.aux = True
                nodes.append(wp)
            res.append((start_pos, nodes))
        return res


class NaiveScheduler(Scheduler):
    """
    Subsclass of the scheduler which follows a naive strategy
    """

    def __init__(self, params: Parameters, scenario: Scenario):
        self.params = params
        self.scenario = scenario

    def schedule(self):
        pass


class Simulator:
    def __init__(self, scheduler_cls, params: Parameters, sc: Scenario, delta: float, W: int):
        self.logger = logging.getLogger(__name__)
        self.scheduler_cls = scheduler_cls
        self.delta = delta
        self.params = params
        self.sf = ScenarioFactory(sc, W)

        self.timestepper = TimeStepper(delta)

        # prepare waypoint indices
        self.current_waypoint_idx = []
        self.reset_waypoint_indices()

        # used in callbacks
        self.remaining = []

        # maintain history of simulation
        self.events = []
        for _ in range(self.sf.N_d):
            self.events.append([])

    def reset_waypoint_indices(self):
        self.current_waypoint_idx = []
        for d in range(self.sf.N_d):
            self.current_waypoint_idx.append(0)

    def sim(self):
        env = simpy.Environment()

        # prepare waypoint indices
        self.reset_waypoint_indices()

        # prepare shared resources
        charging_stations = []
        for s in range(self.sf.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sf.N_d):
            uav = UAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sf.N_w} waypoints per UAV in total")

        # convert scenario
        sc = self.sf.next(self.sf.original_start_pos)

        # get initial schedule
        scheduler = self.scheduler_cls(self.params, sc)
        t_solve, schedules = scheduler.schedule()
        self.logger.debug(f"[{env.now:.1f}] scheduled in {t_solve:.1f}s")

        for i, (start_pos, nodes) in enumerate(schedules):
            uavs[i].set_schedule(env, start_pos, nodes)

        _, ax = plt.subplots()
        fname = f"out/simulation/scenarios/scenario_{self.timestepper.timestep:03}.pdf"
        self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname)
        self.timestepper._inc(_)

        def arrival_cb(event):
            if event.value.node.node_type == NodeType.Waypoint:
                self.sf.incr(event.value.uav.uav_id)
                reached_name = f'new waypoint ({event.value.uav.waypoint_id})'
            elif event.value.node.node_type == NodeType.AuxWaypoint:
                reached_name = 'aux waypoint'
            elif event.value.node.node_type == NodeType.ChargingStation:
                reached_name = f"charging station ({event.value.node.identifier})"
            self.logger.debug(f"[{env.now:.1f}] UAV {event.value.uav.uav_id} reached a {reached_name}")

        def waited_cb(event):
            self.logger.debug(
                f"[{env.now:.1f}] UAV {event.value.uav.uav_id} finished waiting at station {event.value.node.identifier} for {event.value.node.wt:.1f}s")

        def charged_cb(event):
            self.logger.debug(
                f"[{env.now:.1f}] UAV {event.value.uav.uav_id} finished charging at station {event.value.node.identifier} for {event.value.node.ct:.1f}s")

        def ts_cb(_):
            start_positions = []
            batteries = []
            for i, uav in enumerate(uavs):
                state = uav.get_state(env)
                start_positions.append(state.pos)
                batteries.append(state.battery)
            sc = self.sf.next(start_positions)

            params = self.params.copy()
            params.B_start = np.array(batteries)

            scheduler = Scheduler(params, sc)
            t_solve, schedules = scheduler.schedule()
            self.logger.debug(f"[{env.now}] scheduled in {t_solve:.1f}s")

            for i, (start_pos, nodes) in enumerate(schedules):
                uavs[i].set_schedule(env, start_pos, nodes)

            _, ax = plt.subplots()
            fname = f"out/simulation/scenarios/scenario_{self.timestepper.timestep:03}.pdf"
            self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname)

        ts = env.process(self.timestepper.sim(env, callbacks=[ts_cb]))

        self.remaining = self.sf.N_d

        def uav_finished_cb():
            self.logger.debug("uav finished")
            self.remaining -= 1
            if self.remaining == 0:
                ts.interrupt()

        # run simulation
        for uav in uavs:
            uav.add_arrival_cb(arrival_cb)
            uav.add_waited_cb(waited_cb)
            uav.add_charged_cb(charged_cb)
            uav.add_finish_cb(uav_finished_cb)
            env.process(uav.sim(env))
        env.run(until=ts)

        return env

    def plot(self, schedules, batteries, ax=None, fname=None):
        if not ax:
            _, ax = plt.subplots()

        colors = ['red', 'blue']
        for i, (start_pos, nodes) in enumerate(schedules):
            x_all = [start_pos[0]] + [n.pos[0] for n in nodes]
            y_all = [start_pos[1]] + [n.pos[1] for n in nodes]
            x_wp = [n.pos[0] for n in nodes if n.node_type == NodeType.Waypoint]
            y_wp = [n.pos[1] for n in nodes if n.node_type == NodeType.Waypoint]
            x_c = [n.pos[0] for n in nodes if n.node_type == NodeType.ChargingStation]
            y_c = [n.pos[1] for n in nodes if n.node_type == NodeType.ChargingStation]
            x_aux = [start_pos[0]] + [n.pos[0] for n in nodes if n.node_type == NodeType.AuxWaypoint]
            y_aux = [start_pos[1]] + [n.pos[1] for n in nodes if n.node_type == NodeType.AuxWaypoint]
            alphas = np.linspace(1, 0.2, len(x_all) - 1)
            for j in range(len(x_all) - 1):
                x = x_all[j:j + 2]
                y = y_all[j:j + 2]
                label = f"{batteries[i] * 100:.1f}%" if j == 0 else None
                ax.plot(x, y, color=colors[i], label=label, alpha=alphas[j])
            ax.scatter(x_wp, y_wp, c=colors[i])
            ax.scatter(x_c, y_c, marker='s', c=colors[i], facecolor='white')
            ax.scatter(x_aux, y_aux, marker='o', c=colors[i], facecolor='white', zorder=10)
        for i, positions in enumerate(self.sf.sc_orig.positions_w):
            x = [x for x, _, _ in positions]
            y = [y for _, y, _ in positions]
            ax.scatter(x, y, marker='x', s=10, c=colors[i], zorder=-1, alpha=0.2)
        ax.legend()

        if fname:
            plt.savefig(fname, bbox_inches='tight')
