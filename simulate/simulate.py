import logging
import os.path

import numpy as np
import simpy
from PyPDF2 import PdfMerger
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect
from matplotlib.patches import Rectangle

from simulate.event import EventType
from simulate.node import ChargingStation, NodeType, AuxWaypoint
from simulate.parameters import Parameters
from simulate.scheduling import Scheduler
from simulate.uav import UAV, UavStateType
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
            self._inc(None)


class Simulator:
    def __init__(self, scheduler: Scheduler, strategy, params: Parameters, sc: Scenario, directory=None):
        self.logger = logging.getLogger(__name__)
        self.scheduler = scheduler
        self.strategy = strategy
        self.params = params
        self.sc = sc
        self.directory = directory
        self.remaining = sc.N_d
        self.charging_stations = []

        # for outputting simulation
        self.plot_timestepper = TimeStepper(params.plot_delta) if params.plot_delta else None
        self.plot_params = {}
        self.pdfs = []
        self.solve_times = []

    def sim(self):
        env = simpy.Environment()

        # prepare shared resources
        self.charging_stations = []
        for s in range(self.sc.N_s):
            self.charging_stations.append(simpy.PriorityResource(env, capacity=1))

        # prepare UAVs
        def release_lock_cb(env, resource_id):
            epsilon = self.params.epsilon
            resource = self.charging_stations[resource_id]

            def release_after_epsilon(env, epsilon, resource, resource_id):
                self.debug(env, f"locking charging station [{resource_id}] for {epsilon:.2f} after a UAV finished charging")
                req = resource.request(priority=0)
                yield req
                self.debug(env, f"aqcuired lock on charging station [{resource_id}]")

                yield env.timeout(epsilon)
                resource.release(req)
                self.debug(env, f"releasing lock on charging station [{resource_id}]")

            env.process(release_after_epsilon(env, epsilon, resource, resource_id))

        self.uavs = []
        for d in range(self.sc.N_d):
            uav = UAV(d, self.charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d],
                      self.sc.positions_w[d][0])
            uav.add_release_lock_cb(release_lock_cb)
            self.uavs.append(uav)
        self.debug(env, f"visiting {self.sc.N_w - 1} waypoints per UAV in total")

        # get initial schedule
        def reschedule_cb(uavs_to_schedule):
            self.debug(env, f"triggered rescheduling for UAVs: {uavs_to_schedule}")
            if uavs_to_schedule == 'all':
                uavs_to_schedule = list(range(self.sc.N_d))
            start_positions = {}
            batteries = {}
            state_types = {}
            n_waiting = 0
            for d in uavs_to_schedule:
                uav = self.uavs[d]
                state = uav.get_state(env)
                start_positions[d] = state.node.pos.tolist()
                batteries[d] = state.battery
                state_types[d] = state.state_type
                if state.state_type == UavStateType.Waiting:
                    n_waiting += 1

            for d in uavs_to_schedule:
                self.debug(env, f"determined position of UAV [{d}] to be {AuxWaypoint(*start_positions[d])}")
            for d in uavs_to_schedule:
                self.debug(env, f"determined battery of UAV [{d}] to be {batteries[d] * 100:.1f}%")
            for d in uavs_to_schedule:
                self.debug(env, f"determined state type UAV [{d}] to be {state_types[d]}")
            deadlock = n_waiting == self.sc.N_d

            t_solve, (optimal, schedules) = self.scheduler.schedule(start_positions, batteries, state_types, uavs_to_schedule)
            self.debug(env, f"rescheduled {'non-' if not optimal else ''}optimal drone paths in {t_solve:.2}s")
            self.solve_times.append((env.now, optimal, t_solve))
            for d, nodes in schedules.items():
                self.uavs[d].set_schedule(env, nodes)

            for i, cs in enumerate(self.charging_stations):
                if cs.count == cs.capacity:
                    self.debug(env, f"charging station {i} is locked")
                else:
                    self.debug(env, f"charging station {i} is NOT locked")

        reschedule_cb('all')
        self.strategy.set_cb(reschedule_cb)
        strat_proc = env.process(self.strategy.sim(env))

        def plot_ts_cb(_):
            _, ax = plt.subplots()
            fname = f"{self.directory}/it_{self.plot_timestepper.timestep:03}.pdf"
            schedules = []
            for d, uav in enumerate(self.uavs):
                start_pos = uav.get_state(env).node.pos
                schedules.append((start_pos, uav.nodes_to_visit()))
            self.plot(schedules, [uav.get_state(env).battery for uav in self.uavs], ax=ax, fname=fname, title=f"$t={env.now:.2f}$s")

        if self.directory and self.plot_timestepper:
            plot_ts_cb(None)
            self.plot_timestepper._inc(None)
            plot_proc = env.process(
                self.plot_timestepper.sim(env, callbacks=[plot_ts_cb], finish_callbacks=[plot_ts_cb]))

        self.remaining = self.sc.N_d

        def uav_finished_cb(uav):
            self.debug(env, f"UAV [{uav.uav_id}] finished")
            self.remaining -= 1
            if self.remaining == 0:
                if self.directory and self.plot_timestepper:
                    plot_proc.interrupt()
                strat_proc.interrupt()

        for uav in self.uavs:
            uav.add_arrival_cb(self.scheduler.handle_event)
            uav.add_arrival_cb(self.strategy.handle_event)
            uav.add_waited_cb(self.scheduler.handle_event)
            uav.add_waited_cb(self.strategy.handle_event)
            uav.add_charged_cb(self.scheduler.handle_event)
            uav.add_charged_cb(self.strategy.handle_event)
            uav.add_finish_cb(uav_finished_cb)
            env.process(uav.sim(env))

        try:
            env.run(until=strat_proc)
        finally:
            if self.directory:
                fname = os.path.join(self.directory, "combined.pdf")
                if self.pdfs:
                    merger = PdfMerger()
                    for pdf in self.pdfs:
                        merger.append(pdf)

                    merger.write(fname)
                    merger.close()

                    # remove the intermediate files
                    for pdf in self.pdfs:
                        os.remove(pdf)
                elif os.path.exists(fname):
                    os.remove(fname)

                # plot batteries
                fname = os.path.join(self.directory, "battery.pdf")
                plot_events_battery([u.events for u in self.uavs], fname, aspect=self.params.r_charge.min())

                # plot occupancy
                fname = os.path.join(self.directory, "occupancy.pdf")
                plot_station_occupancy([u.events for u in self.uavs], self.sc.N_s, env.now, fname)

                # output events
                event_dir = os.path.join(self.directory, "events")
                os.makedirs(event_dir, exist_ok=True)
                for d, uav in enumerate(self.uavs):
                    fname = os.path.join(event_dir, f"{d}.txt")
                    with open(fname, "w") as f:
                        for ev in uav.events:
                            f.write(f"{ev}\n")

        return self.solve_times, env, [u.events for u in self.uavs]

    def debug(self, env, msg):
        self.logger.debug(f"[{env.now:.2f}] {msg}")

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

        for d in range(self.sc.N_d):
            remaining_waypoints = self.scheduler.remaining_waypoints(d)
            x = [x for x, _, _ in remaining_waypoints]
            y = [y for _, y, _ in remaining_waypoints]
            ax.scatter(x, y, marker='x', s=10, color=colors[d], zorder=-1, alpha=0.2)

        x = [x for x, _, _ in self.sc.positions_S]
        y = [y for _, y, _ in self.sc.positions_S]
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
        # ax.axis('off')

        if fname:
            plt.savefig(fname, bbox_inches='tight')
            self.pdfs.append(fname)
        plt.close()


def plot_events_battery(events: list, fname: str, aspect=None):
    """
    Plots the battery over time for the given events
    """
    execution_times = []
    for d in range(len(events)):
        execution_times.append(events[d][-1].ts_end)
    max_execution_time = max(execution_times)

    if aspect:
        w, h = figaspect(aspect)
        h = h * len(events)
        figsize = (w, h)
    else:
        figsize = None
    _, axes = plt.subplots(len(events), 1, sharex=True, sharey=True, figsize=figsize)
    if len(events) == 1:
        axes = [axes]

    uav_colors = gen_colors(len(events))

    station_ids = []
    for d in range(len(events)):
        for e in events[d]:
            if type(e.node) == ChargingStation:
                if e.node.identifier not in station_ids:
                    station_ids.append(e.node.identifier)
    station_colors = gen_colors(len(station_ids))

    for d in range(len(events)):
        X = []
        Y = []
        for i, e in enumerate(events[d]):
            ts = e.ts_end
            X.append(ts)
            Y.append(e.battery)

            if e.name == EventType.charged:
                rect = Rectangle(
                    (e.ts_start, 0),
                    e.duration,
                    1,
                    color=station_colors[e.node.identifier],
                    ec=None,
                    alpha=0.3,
                    zorder=-1
                )
                axes[d].add_patch(rect)
            elif e.name == EventType.waited:
                rect = Rectangle(
                    (e.ts_start, 0),
                    e.duration,
                    1,
                    color=station_colors[e.node.identifier],
                    fill=None,
                    linewidth=0,
                    alpha=0.1,
                    hatch="/" * 6,
                    ec=None,
                    zorder=-1
                )
                axes[d].add_patch(rect)

        axes[d].plot(X, Y, c=uav_colors[d])
        axes[d].set_ylim([0, 1])

    # add vertical lines
    for d in range(len(events)):
        axes[d].axvline(max_execution_time, color='red', zorder=-10)
    axes[np.argmin(execution_times)].text(max_execution_time, 0.5, f'{max_execution_time:.1f}s', color='red',
                                          backgroundcolor='white', fontsize='xx-small', ha='center', zorder=-9)

    plt.savefig(fname, bbox_inches='tight')


def plot_station_occupancy(events: list, nstations: int, total_duration: float, fname: str):
    """
    Plot the number of UAVs that use a charging station over time
    """
    colors = gen_colors(nstations)

    _, axes = plt.subplots(nrows=nstations, ncols=1, sharex=True, sharey=True)
    if nstations == 1:
        axes = [axes]

    for station in range(nstations):
        X = [0]
        Y = [0]
        cur_charged = 0

        changes = {}
        for d in range(len(events)):
            for ev in events[d]:
                if ev.name == EventType.charged and ev.node.identifier == station:
                    changes[ev.ts_start] = changes.get(ev.ts_start, 0) + 1
                    changes[ev.ts_end] = changes.get(ev.ts_end, 0) - 1

        for ts, change in sorted(changes.items()):
            prev_charged = cur_charged
            cur_charged = cur_charged + change

            X += [ts, ts]
            Y += [prev_charged, cur_charged]
        X.append(total_duration)
        Y.append(cur_charged)
        axes[station].set_title(f"Station {station}")
        axes[station].plot(X, Y, colors[station])
        axes[station].fill_between(X, Y, facecolor=colors[station], alpha=0.2)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
