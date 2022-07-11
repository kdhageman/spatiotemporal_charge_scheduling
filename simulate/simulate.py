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
from simulate.uav import MilpUAV, NaiveUAV
from util.decorators import timed
from util.distance import dist3
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

    def __init__(self, scenario: Scenario):
        self.original_start_pos = [wps[0] for wps in scenario.positions_w]
        self.positions_S = scenario.positions_S
        self.positions_w = [wps[1:] for wps in scenario.positions_w]
        self.offsets = [0] * scenario.N_d
        self.sc_orig = scenario
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w

    def incr(self, d):
        """
        Increments the waypoint offset of the given UAV
        """
        self.offsets[d] += 1

    def n_remaining_waypoints(self, d):
        """
        Returns the remaining number of waypoints for the given UAV
        """
        return self.N_w - self.offsets[d] - 1

    def remaining_waypoints(self, d):
        """
        Returns the list of remaining waypoints for the given UAV that need to be visited
        """
        return self.positions_w[d][self.offsets[d]:]

    def next(self, W, start_positions=None, sigma=1):
        """
        Returns the next scenario
        """
        if not start_positions:
            start_positions = self.original_start_pos

        positions_w = []
        D_N = []
        D_W = []

        for d, wps_src in enumerate(self.positions_w):
            wps_src_full = [start_positions[d]] + wps_src[self.offsets[d]:]
            while len(wps_src_full) < sigma * (W - 1) + 1:
                wps_src_full.append(wps_src_full[-1])

            wps = []
            D_N_matr = []
            D_W_matr = []

            n = 0
            while len(wps) < W:
                wp_hat = wps_src_full[n]
                wps.append(wp_hat)

                if len(wps) < W:
                    # calculate D_N
                    D_N_col = []
                    for pos_S in self.positions_S:
                        # distance to charging stations
                        distance = dist3(wp_hat, pos_S)
                        D_N_col.append(distance)

                    distance = 0
                    for i in range(n, n + sigma):
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

                        for i in range(n + 1, n + sigma):
                            pos_a = wps_src_full[i]
                            pos_b = wps_src_full[i + 1]
                            distance += dist3(pos_a, pos_b)
                        D_W_col.append(distance)
                    D_W_col.append(0)
                    D_W_matr.append(D_W_col)

                n += sigma
            D_N.append(D_N_matr)
            D_W.append(D_W_matr)
            positions_w.append(wps)
        sc = Scenario(positions_S=self.positions_S, positions_w=positions_w)

        D_N = np.array(D_N).transpose((0, 2, 1))
        sc.D_N = D_N

        D_W = np.array(D_W).transpose((0, 2, 1))
        sc.D_W = D_W

        return sc


class Scheduler:
    def __init__(self, params: Parameters, scenario: Scenario, sigma: int = 1):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.scenario_unstride = scenario
        self.sigma = sigma

    @timed
    def schedule(self, sc: Scenario, solver=SolverFactory("gurobi")):
        """
        Return the most recent schedule
        :return: path to follow, dim = N_d x (W - 1) x (N_s + 1)
        :return: charging times, dim = N_d x (W - 1)
        :return: waiting times, dim = N_d x (W - 1)
        """
        model = MultiUavModel(scenario=sc, parameters=self.params.as_dict())
        solution = solver.solve(model)
        if solution['Solver'][0]['Termination condition'] != 'optimal':
            raise NotSolvableException("non-optimal solution")

        res = []
        for d in model.d:
            start_pos = sc.positions_w[d][0]
            wps_full = self.scenario_unstride.positions_w[d]
            while len(wps_full) < self.sigma * (len(sc.positions_w[d]) - 1) + 1:
                wps_full.append(wps_full[-1])

            nodes = []
            for w_s_hat in model.w_s:
                n = model.P_np[d, :, w_s_hat].tolist().index(1)
                if n < model.N_s:
                    # visit charging station first
                    ct = model.C_np[d][w_s_hat]
                    wt = model.W_np[d][w_s_hat]
                    nodes.append(
                        ChargingStation(*sc.positions_S[n], n, wt, ct)
                    )

                # add stride waypoints
                for i in range(w_s_hat * self.sigma + 1, w_s_hat * self.sigma + self.sigma):
                    pos = wps_full[i]
                    wp = Waypoint(*pos)
                    wp.strided = True
                    nodes.append(wp)

                # add next waypoint
                pos = wps_full[self.sigma * (w_s_hat + 1)]
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


class NaiveSimulator:
    def __init__(self, params: Parameters, sc: Scenario, plot_delta: float = 0.1, directory=None):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.directory = directory
        self.plot_timestepper = TimeStepper(plot_delta)
        self.sc = sc

        self.pdfs = []
        self.plot_params = {}
        self.remaining = []

    def sim(self):
        env = simpy.Environment()

        # prepare shared resources
        charging_stations = []
        for s in range(self.sc.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sc.N_d):
            uav = NaiveUAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d],
                           self.params.B_min[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sf.N_w - 1} waypoints per UAV in total")

        def arrival_cb(event):
            uav_id = event.value.uav.uav_id
            self.logger.debug(
                f"[{env.now:.2f}] UAV [{uav_id}] reached {event.value.node} with {event.value.uav.battery * 100:.1f}% battery ({self.sf.n_remaining_waypoints(uav_id)}/{self.sf.N_w - 1} waypoints remaining)")

        def waited_cb(event):
            self.logger.debug(
                f"[{env.now:.2f}] UAV {event.value.uav.uav_id} finished waiting at station {event.value.node.identifier} for {event.value.node.wt:.2f}s")

        def charged_cb(event):
            self.logger.debug(
                f"[{env.now:.2f}] UAV {event.value.uav.uav_id} finished charging at station {event.value.node.identifier} for {event.value.node.ct:.2f}s")

        for uav in uavs:
            uav.add_arrival_cb(arrival_cb)
            uav.add_waited_cb(waited_cb)
            uav.add_charged_cb(charged_cb)
            env.process(uav.sim(env))

        env.run()

        return env, [uav.events for uav in uavs]

class MilpSimulator:
    def __init__(self, scheduler_cls, params: Parameters, sc: Scenario, schedule_delta: float, W: int,
                 plot_delta: float = 0.1, sigma=1, directory=None):
        self.logger = logging.getLogger(__name__)
        self.scheduler_cls = scheduler_cls
        self.params = params
        self.directory = directory
        self.sf = ScenarioFactory(sc)
        self.W = W
        self.sc_orig = sc
        self.sigma = sigma

        self.schedule_timestepper = TimeStepper(schedule_delta)
        self.plot_timestepper = TimeStepper(plot_delta)

        self.schedules = None
        self.pdfs = []
        self.plot_params = {}
        self.solve_times = []

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
            uav = MilpUAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sf.N_w - 1} waypoints per UAV in total")

        # convert scenario
        sc = self.sf.next(self.W, sigma=self.sigma)

        # get initial schedule
        params = self.params.copy()
        params.B_end = params.B_min
        sc_remaining = self.sf.next(self.sc_orig.N_w, sigma=1)
        scheduler = self.scheduler_cls(params, sc_remaining, sigma=self.sigma)
        t_solve, self.schedules = scheduler.schedule(sc)
        self.solve_times.append(t_solve)
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
                f"[{env.now:.2f}] UAV [{uav_id}] reached {event.value.node} with {event.value.uav.battery * 100:.1f}% battery ({self.sf.n_remaining_waypoints(uav_id)}/{self.sf.N_w - 1} waypoints remaining)")

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
            sc = self.sf.next(self.W, start_positions, sigma=self.sigma)

            params = self.params.copy()
            params.B_start = np.array(batteries)

            B_end = []
            for d, wps in enumerate(self.sc_orig.positions_w):
                overall_pos_end_wp = self.sc_orig.positions_w[d][-1]
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

            sc_remaining = self.sf.next(self.sc_orig.N_w, start_positions, sigma=1)
            scheduler = self.scheduler_cls(params, sc_remaining, sigma=self.sigma)
            t_solve, self.schedules = scheduler.schedule(sc)
            self.solve_times.append(t_solve)
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

        return self.solve_times, env, [uav.events for uav in uavs]

    def plot(self, schedules, batteries, ax=None, fname=None, title=None):
        if not ax:
            _, ax = plt.subplots()

        colors = gen_colors(len(schedules))
        for i, (start_pos, nodes) in enumerate(schedules):
            x_all = [start_pos[0]] + [n.pos[0] for n in nodes]
            y_all = [start_pos[1]] + [n.pos[1] for n in nodes]
            x_wp_nonstride = [n.pos[0] for n in nodes if n.node_type == NodeType.Waypoint and not n.strided]
            y_wp_nonstride = [n.pos[1] for n in nodes if n.node_type == NodeType.Waypoint and not n.strided]
            x_wp_stride = [n.pos[0] for n in nodes if n.node_type == NodeType.Waypoint and n.strided]
            y_wp_stride = [n.pos[1] for n in nodes if n.node_type == NodeType.Waypoint and n.strided]
            x_c = [n.pos[0] for n in nodes if n.node_type == NodeType.ChargingStation]
            y_c = [n.pos[1] for n in nodes if n.node_type == NodeType.ChargingStation]
            alphas = np.linspace(1, 0.2, len(x_all) - 1)
            for j in range(len(x_all) - 1):
                x = x_all[j:j + 2]
                y = y_all[j:j + 2]
                label = f"{batteries[i] * 100:.1f}%" if j == 0 else None
                ax.plot(x, y, color=colors[i], label=label, alpha=alphas[j])
            ax.scatter(x_wp_nonstride, y_wp_nonstride, c='white', s=40, edgecolor=colors[i], zorder=2)  # waypoints
            ax.scatter(x_wp_stride, y_wp_stride, c='white', s=40, edgecolor=colors[i], zorder=2)  # waypoints (strided)
            ax.scatter(x_c, y_c, marker='s', s=70, c='white', edgecolor=colors[i], zorder=2)  # charging stations
            ax.scatter([start_pos[0]], [start_pos[1]], marker='o', s=60, color=colors[i], zorder=10)  # starting point

        # for i, positions in enumerate(self.sf.sc_orig.positions_w):
        for d in range(self.sf.N_d):
            remaining_waypoints = self.sf.remaining_waypoints(d)
            x = [x for x, _, _ in remaining_waypoints]
            y = [y for _, y, _ in remaining_waypoints]
            ax.scatter(x, y, marker='x', s=10, color=colors[d], zorder=-1, alpha=0.2)

        x = [x for x, _, _ in self.sf.sc_orig.positions_S]
        y = [y for _, y, _ in self.sf.sc_orig.positions_S]
        ax.scatter(x, y, marker='s', s=70, c='white', edgecolor='black', zorder=-1, alpha=0.2)

        ax.axis("equal")
        # fix plot limits for battery calculation
        if len(self.plot_params) == 0:
            self.plot_params['xlim'] = ax.get_xlim()
            self.plot_params['ylim'] = ax.get_ylim()

            xmin, xmax = ax.get_xlim()

            self.plot_params['width_outer'] = (xmax - xmin) * 0.07
            self.plot_params['height_outer'] = self.plot_params['width_outer'] * 0.5
            self.plot_params['y_offset'] = self.plot_params['height_outer']

            self.plot_params['lw_outer'] = 1.5

            padding = self.plot_params['height_outer'] / 3
            self.plot_params['width_inner_max'] = self.plot_params['width_outer'] - padding
            self.plot_params['height_inner'] = self.plot_params['height_outer'] - padding

        ax.set_xlim(self.plot_params['xlim'])
        ax.set_ylim(self.plot_params['ylim'])

        for i, (start_pos, nodes) in enumerate(schedules):
            # draw battery under current battery position
            x_outer = start_pos[0] - (self.plot_params['width_outer'] / 2)
            y_outer = start_pos[1] - (self.plot_params['height_outer'] / 2) - self.plot_params['y_offset']
            outer = Rectangle((x_outer, y_outer), self.plot_params['width_outer'], self.plot_params['height_outer'],
                              color=colors[i], linewidth=self.plot_params['lw_outer'],
                              fill=False)
            ax.add_patch(outer)

            width_inner = self.plot_params['width_inner_max'] * batteries[i]
            x_inner = start_pos[0] - (self.plot_params['width_inner_max'] / 2)
            y_inner = start_pos[1] - (self.plot_params['height_inner'] / 2) - self.plot_params['y_offset']
            inner = Rectangle((x_inner, y_inner), width_inner, self.plot_params['height_inner'], color=colors[i],
                              linewidth=0, fill=True)
            ax.add_patch(inner)

        if title:
            ax.set_title(title)

        ax.margins(0.1, 0.1)
        ax.axis('off')

        if fname:
            plt.savefig(fname, bbox_inches='tight')
            self.pdfs.append(fname)
        plt.close()
