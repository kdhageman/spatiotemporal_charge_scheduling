import logging
import os.path

import numpy as np
import simpy
from PyPDF2 import PdfMerger
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pyomo.opt import SolverFactory

from pyomo_models.multi_uavs import MultiUavModel
from simulate.node import ChargingStation, Waypoint, NodeType, AuxWaypoint
from simulate.parameters import Parameters
from simulate.uav import UAV
from util.decorators import timed
from util.scenario import Scenario


def gen_colors(n):
    np.random.seed(0)
    res = []
    for d in range(n):
        c = np.random.rand(3).tolist()
        res.append(c)
    return res


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

    def sim(self, env, callbacks=[], finish_callbacks=[]):
        while True:
            event = env.timeout(self.interval)
            for cb in callbacks:
                event.callbacks.append(cb)
            event.callbacks.append(self._inc)
            try:
                yield event
            except simpy.exceptions.Interrupt:
                break
        for cb in finish_callbacks:
            cb(event)
            self._inc()


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
        Increments the waypoint offset of the given UAV
        """
        self.offsets[d] += 1

    def remaining_waypoints(self, d):
        """
        Returns the remaining number of waypoints for the given UAV
        """
        return self.N_w - self.offsets[d] - 1

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
                pos = self.scenario.positions_w[d][w_s + 1]
                if start_pos == pos:
                    wp = AuxWaypoint(*pos)
                else:
                    wp = Waypoint(*pos)
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
    def __init__(self, scheduler_cls, params: Parameters, sc: Scenario, schedule_delta: float, W: int,
                 plot_delta: float = 0.1, directory=None):
        self.logger = logging.getLogger(__name__)
        self.scheduler_cls = scheduler_cls
        self.params = params
        self.directory = directory
        self.sf = ScenarioFactory(sc, W)

        self.schedule_timestepper = TimeStepper(schedule_delta)
        self.plot_timestepper = TimeStepper(plot_delta)

        self.schedules = None
        self.pdfs = []

        # used in callbacks
        self.remaining = []

    def sim(self):
        env = simpy.Environment()

        # prepare shared resources
        charging_stations = []
        for s in range(self.sf.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sf.N_d):
            uav = UAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sf.N_w-1} waypoints per UAV in total")

        # convert scenario
        sc = self.sf.next(self.sf.original_start_pos)

        # get initial schedule
        scheduler = self.scheduler_cls(self.params, sc)
        t_solve, self.schedules = scheduler.schedule()
        self.logger.debug(f"[{env.now:.2f}] scheduled in {t_solve:.1f}s")
        for i, (start_pos, nodes) in enumerate(self.schedules):
            node_list = " - ".join([str(n) for n in [Waypoint(*start_pos)] + nodes])
            n_wp = len([n for n in nodes if n.node_type == NodeType.Waypoint])
            n_charge = len([n for n in nodes if n.node_type == NodeType.ChargingStation])
            n_aux = len([n for n in nodes if n.node_type == NodeType.AuxWaypoint])
            self.logger.debug(f"[{env.now:.2f}] schedule for UAV [{i}]: {node_list}")
            self.logger.debug(
                f"[{env.now:.2f}] schedule for UAV [{i}] is composed of {n_wp} waypoints, {n_charge} charging stations and {n_aux} auxiliary waypoints")

        for i, (start_pos, nodes) in enumerate(self.schedules):
            uavs[i].set_schedule(env, start_pos, nodes)

        if self.directory:
            _, ax = plt.subplots()
            fname = f"{self.directory}/it_{self.plot_timestepper.timestep:03}.pdf"
            schedules = []
            for uav in uavs:
                start_pos = uav.get_state(env).pos
                schedules.append((start_pos, uav.eg.nodes))
            self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname,
                      title=f"$t={env.now:.2f}$s")
            self.plot_timestepper._inc(_)

        def arrival_cb(event):
            uav_id = event.value.uav.uav_id
            if event.value.node.node_type == NodeType.Waypoint:
                self.sf.incr(uav_id)
            self.logger.debug(
                f"[{env.now:.2f}] UAV [{uav_id}] reached {event.value.node} with {event.value.uav.battery * 100:.1f}% battery ({self.sf.remaining_waypoints(uav_id)}/{self.sf.N_w-1} waypoints remaining)")

        def waited_cb(event):
            self.logger.debug(
                f"[{env.now:.2f}] UAV {event.value.uav.uav_id} finished waiting at station {event.value.node.identifier} for {event.value.node.wt:.2f}s")

        def charged_cb(event):
            self.logger.debug(
                f"[{env.now:.2f}] UAV {event.value.uav.uav_id} finished charging at station {event.value.node.identifier} for {event.value.node.ct:.2f}s")

        def schedule_ts_cb(_):
            start_positions = []
            batteries = []
            for i, uav in enumerate(uavs):
                state = uav.get_state(env)
                logging.debug(f"[{env.now:.2f}] determined position of UAV [{i}] to be {state.pos_str}")
                logging.debug(f"[{env.now:.2f}] determined battery of UAV [{i}] to be {state.battery * 100:.1f}%")
                start_positions.append(state.pos.tolist())
                batteries.append(state.battery)
            sc = self.sf.next(start_positions)

            params = self.params.copy()
            params.B_start = np.array(batteries)

            scheduler = Scheduler(params, sc)
            t_solve, self.schedules = scheduler.schedule()
            self.logger.debug(f"[{env.now:.2f}] scheduled in {t_solve:.1f}s")
            for i, (start_pos, nodes) in enumerate(self.schedules):
                node_list = " - ".join([str(n) for n in [Waypoint(*start_pos)] + nodes])
                n_wp = len([n for n in nodes if n.node_type == NodeType.Waypoint])
                n_charge = len([n for n in nodes if n.node_type == NodeType.ChargingStation])
                n_aux = len([n for n in nodes if n.node_type == NodeType.AuxWaypoint])
                self.logger.debug(f"[{env.now:.2f}] schedule for UAV [{i}]: {node_list}")
                self.logger.debug(
                    f"[{env.now:.2f}] schedule for UAV [{i}] is composed of {n_wp} waypoints, {n_charge} charging stations and {n_aux} auxiliary waypoints")

            for i, (start_pos, nodes) in enumerate(self.schedules):
                uavs[i].set_schedule(env, start_pos, nodes)

        def plot_ts_cb(_):
            _, ax = plt.subplots()
            fname = f"{self.directory}/it_{self.plot_timestepper.timestep:03}.pdf"
            schedules = []
            for uav in uavs:
                start_pos = uav.get_state(env).pos
                schedules.append((start_pos, uav.eg.nodes))
            self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname,
                      title=f"$t={env.now:.2f}$s")

        schedule_ts = env.process(self.schedule_timestepper.sim(env, callbacks=[schedule_ts_cb]))
        if self.directory:
            plot_ts = env.process(self.plot_timestepper.sim(env, callbacks=[plot_ts_cb], finish_callbacks=[plot_ts_cb]))

        self.remaining = self.sf.N_d

        def uav_finished_cb(uav):
            self.logger.debug(f"[{env.now:.2f}] UAV [{uav.uav_id}] finished")
            self.remaining -= 1
            if self.remaining == 0:
                schedule_ts.interrupt()
                if self.directory:
                    plot_ts.interrupt()

        # run simulation
        for uav in uavs:
            uav.add_arrival_cb(arrival_cb)
            uav.add_waited_cb(waited_cb)
            uav.add_charged_cb(charged_cb)
            uav.add_finish_cb(uav_finished_cb)
            env.process(uav.sim(env))

        try:
            env.run(until=schedule_ts)
        finally:
            if self.directory:
                merger = PdfMerger()
                for pdf in self.pdfs:
                    merger.append(pdf)
                fname = os.path.join(self.directory, "combined.pdf")
                merger.write(fname)
                merger.close()

                # remove the intermediate files
                for pdf in self.pdfs:
                    os.remove(pdf)

        return env, [uav.events for uav in uavs]

    def plot(self, schedules, batteries, ax=None, fname=None, title=None):
        if not ax:
            _, ax = plt.subplots()

        colors = gen_colors(len(schedules))
        for i, (start_pos, nodes) in enumerate(schedules):
            x_all = [start_pos[0]] + [n.pos[0] for n in nodes]
            y_all = [start_pos[1]] + [n.pos[1] for n in nodes]
            x_wp = [n.pos[0] for n in nodes if n.node_type == NodeType.Waypoint]
            y_wp = [n.pos[1] for n in nodes if n.node_type == NodeType.Waypoint]
            x_c = [n.pos[0] for n in nodes if n.node_type == NodeType.ChargingStation]
            y_c = [n.pos[1] for n in nodes if n.node_type == NodeType.ChargingStation]
            alphas = np.linspace(1, 0.2, len(x_all) - 1)
            for j in range(len(x_all) - 1):
                x = x_all[j:j + 2]
                y = y_all[j:j + 2]
                label = f"{batteries[i] * 100:.1f}%" if j == 0 else None
                ax.plot(x, y, color=colors[i], label=label, alpha=alphas[j])
            ax.scatter(x_wp, y_wp, c='white', s=40, edgecolor=colors[i], zorder=2)  # waypoints
            ax.scatter(x_c, y_c, marker='s', s=70, c='white', edgecolor=colors[i], zorder=2)  # charging stations
            ax.scatter([start_pos[0]], [start_pos[1]], marker='o', s=60, color=colors[i], zorder=10)  # starting point

        for i, positions in enumerate(self.sf.sc_orig.positions_w):
            x = [x for x, _, _ in positions]
            y = [y for _, y, _ in positions]
            ax.scatter(x, y, marker='x', s=10, color=colors[i], zorder=-1, alpha=0.2)

        x = [x for x, _, _ in self.sf.sc_orig.positions_S]
        y = [y for _, y, _ in self.sf.sc_orig.positions_S]
        ax.scatter(x, y, marker='s', s=70, c='white', edgecolor='black', zorder=-1, alpha=0.2)

        ax.axis("equal")

        xmin, xmax = ax.get_xlim()

        width_outer = (xmax - xmin) * 0.07
        height_outer = width_outer * 0.5
        y_offset = height_outer

        lw_outer = 1.5

        padding = height_outer / 3
        width_inner_max = width_outer - padding
        height_inner = height_outer - padding

        for i, (start_pos, nodes) in enumerate(schedules):
            # draw battery under current battery position
            x_outer = start_pos[0] - (width_outer / 2)
            y_outer = start_pos[1] - (height_outer / 2) - y_offset
            outer = Rectangle((x_outer, y_outer), width_outer, height_outer, color=colors[i], linewidth=lw_outer,
                              fill=False)
            ax.add_patch(outer)

            width_inner = width_inner_max * batteries[i]
            x_inner = start_pos[0] - (width_inner_max / 2)
            y_inner = start_pos[1] - (height_inner / 2) - y_offset
            inner = Rectangle((x_inner, y_inner), width_inner, height_inner, color=colors[i], linewidth=0, fill=True)
            ax.add_patch(inner)

        if title:
            ax.set_title(title)

        ax.margins(0.1, 0.1)
        ax.axis('off')

        if fname:
            plt.savefig(fname, bbox_inches='tight')
            self.pdfs.append(fname)
        plt.close()
