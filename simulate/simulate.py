import logging
import os.path
import pickle
from datetime import datetime
from typing import List, Dict

import numpy as np
import simpy
from PyPDF2 import PdfMerger
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid

from simulate.event import EventType, Event
from simulate.node import ChargingStation, NodeType, AuxWaypoint
from simulate.parameters import Parameters
from simulate.plot import SimulationAnimator
from simulate.result import SimResult
from simulate.scheduling import Scheduler
from simulate.uav import UAV, UavStateType
from simulate.util import gen_colors
from util.scenario import Scenario


class SolveTime:
    def __init__(self, timestamp, optimal, t_solve, n_remaining_waypoints):
        self.timestamp = timestamp
        self.optimal = optimal
        self.t_solve = t_solve
        self.n_remaining_waypoints = n_remaining_waypoints


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
    def __init__(self, interval: float):
        self.timestep = 0
        self.interval = interval

    def _inc(self, _):
        self.timestep += 1

    def sim(self, env, callbacks: list = [], finish_callbacks: list = []):
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


class SimulationResult:
    def __init__(self, sc: Scenario, events, schedules):
        self.sc = sc
        self.events = events
        self.schedules = schedules


class Simulator:
    def __init__(self, scheduler: Scheduler, strategy, params: Parameters, sc: Scenario, directory: str = None):
        self.logger = logging.getLogger(__name__)
        self.scheduler = scheduler
        self.strategy = strategy
        self.params = params
        self.sc = sc
        self.directory = directory
        self.remaining = sc.N_d
        self.charging_stations = []
        self.charging_stations_locks = {}

        # for outputting simulation
        self.plot_params = {}
        self.solve_times = []
        self.all_schedules = {}

        env = simpy.Environment()

        # prepare shared resources
        self.charging_stations = []
        for s in range(self.sc.N_s):
            self.charging_stations.append(simpy.PriorityResource(env, capacity=1))

        # prepare UAVs
        def release_lock_cb(env, uav_id, resource_id):
            epsilon = self.params.epsilon
            resource = self.charging_stations[resource_id]

            def release_after_epsilon(env, epsilon, resource, resource_id):
                self.charging_stations_locks[resource_id] = (env.now, uav_id)
                self.debug(env, f"simulator is locking charging station [{resource_id}] for {epsilon:.2f} after a UAV finished charging")
                t_before = env.now
                req = resource.request(priority=0)
                yield req
                elapsed = env.now - t_before
                self.debug(env, f"simulator acquired lock on charging station [{resource_id}] (after {elapsed:.2f}s)")

                yield env.timeout(epsilon)
                resource.release(req)
                self.debug(env, f"simulator released lock on charging station [{resource_id}]")
                if resource_id in self.charging_stations_locks:
                    del self.charging_stations_locks[resource_id]  # TODO: fix bug here!

            env.process(release_after_epsilon(env, epsilon, resource, resource_id))

        self.uavs = []
        for d in range(self.sc.N_d):
            uav = UAV(d, self.charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d], self.sc.start_positions[d])
            uav.add_release_lock_cb(release_lock_cb)
            self.uavs.append(uav)
        self.debug(env, f"visiting {self.sc.N_w - 1} waypoints per UAV in total")

        # get initial schedule
        def reschedule_cb(uavs_to_schedule):
            self.info(env, "--------- START RESCHEDULING ---------")
            if uavs_to_schedule == 'all':
                uavs_to_schedule = list(range(self.sc.N_d))
            start_positions = {}
            batteries = {}
            n_waiting = 0
            for d in uavs_to_schedule:
                uav = self.uavs[d]
                state = uav.get_state(env)
                start_positions[d] = state.node.pos.tolist()
                batteries[d] = state.battery
                if state.state_type == UavStateType.Waiting:
                    n_waiting += 1

            for d in uavs_to_schedule:
                self.debug(env, f"determined position of UAV [{d}] to be {AuxWaypoint(*start_positions[d])}")
            for d in uavs_to_schedule:
                self.debug(env, f"determined battery of UAV [{d}] to be {batteries[d] * 100:.1f}%")

            cs_locks = np.zeros((len(self.uavs), len(self.charging_stations)))
            for cs, (ts, uav_id) in self.charging_stations_locks.items():
                elapsed = env.now - ts
                remaining = self.params.epsilon - elapsed
                for d in range(len(self.uavs)):
                    cs_locks[d, cs] = remaining
                    # TODO: should the original UAV NOT be blocked? (see code below)
                    # if d != uav_id:
                    #     cs_locks[d, cs] = remaining
            for uav in self.uavs:
                if uav.resource_id is not None:
                    for d in range(len(self.uavs)):
                        if d != uav.uav_id:
                            cs_locks[d, uav.resource_id] = self.params.epsilon

            t_solve, (optimal, schedules) = self.scheduler.schedule(start_positions, batteries, cs_locks, uavs_to_schedule)
            self.debug(env, f"rescheduled {'non-' if not optimal else ''}optimal drone paths in {t_solve:.2f}s")
            n_remaining_waypoints = [self.scheduler.n_remaining_waypoints(d) for d in range(self.sc.N_d)]
            self.solve_times.append(SolveTime(env.now, optimal, t_solve, n_remaining_waypoints))

            for d, nodes in schedules.items():
                wps = [n for n in nodes if n.node_type == NodeType.Waypoint]
                if wps:
                    first_wp_id = [n for n in nodes if n.node_type == NodeType.Waypoint][0].identifier
                    last_wp_id = [n for n in nodes if n.node_type == NodeType.Waypoint][-1].identifier
                    self.debug(env, f"for UAV [{d}] scheduled from waypoint [{first_wp_id}] up to waypoint [{last_wp_id}]")
                else:
                    self.debug(env, f"for UAV [{d}] scheduled NO waypoints")

            for d, nodes in schedules.items():
                self.uavs[d].set_schedule(env, nodes)
                existing_schedules = self.all_schedules.get(d, [])
                self.all_schedules[d] = existing_schedules + [(env.now, nodes)]

            for i, cs in enumerate(self.charging_stations):
                if cs.count == cs.capacity:
                    self.debug(env, f"charging station {i} is locked")
                else:
                    self.debug(env, f"charging station {i} is NOT locked")
            self.info(env, "--------- END RESCHEDULING ---------")

        reschedule_cb('all')
        self.strategy.set_cb(reschedule_cb)
        strat_proc = env.process(self.strategy.sim(env))

        def uav_finished_cb(uav):
            self.debug(env, f"UAV [{uav.uav_id}] finished its mission")
            if self.scheduler.n_remaining_waypoints(uav.uav_id) != 0:
                raise Exception(f"UAV [{uav.uav_id}] is finished, but still has waypoints to visit")

            self.remaining -= 1
            if self.remaining == 0:
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

        self.env = env
        self.strat_proc = strat_proc

    def sim(self) -> SimResult:
        try:
            self.env.run(until=self.strat_proc)
        finally:
            self.info(self.env, f"finished simulation in {self.env.now:.2f}s")
            for d in range(self.sc.N_d):
                self.info(self.env, f"UAV [{d}] has {self.scheduler.n_remaining_waypoints(d)} remaining waypoints")

            if self.directory:
                events = [u.events(self.env) for u in self.uavs]
                time_spent = {d: uav.time_spent for d, uav in enumerate(self.uavs)}
                for d in range(len(self.uavs)):
                    time_spent[d]['moving_minimum'] = self.sc.D_N[d, -1, :].sum() / self.params.v[d]
                nr_visited_waypoints = [uav.waypoint_id for uav in self.uavs]
                occupancy = events_to_occupancy(events)
                result = SimResult(self.params, self.sc, events, self.solve_times, self.env.now, time_spent, self.all_schedules, nr_visited_waypoints, occupancy, self.scheduler)

                # plot batteries
                fname = os.path.join(self.directory, "battery.pdf")
                plot_events_battery(result, fname)

                # plot occupancy
                fname = os.path.join(self.directory, "occupancy.pdf")
                plot_station_occupancy(result, fname)

                fname = os.path.join(self.directory, "animation.html")
                events = {d: uav.events(self.env) for d, uav in enumerate(self.uavs)}
                if self.params.plot_delta:
                    sa = SimulationAnimator(self.sc, events, self.all_schedules, self.params.plot_delta)
                    sa.animate(fname)

        return result

    def debug(self, env, msg):
        self.logger.debug(self._craft_msg(env, msg))

    def info(self, env, msg):
        self.logger.info(self._craft_msg(env, msg))

    def _craft_msg(self, env, msg):
        return f"[{datetime.now().strftime('%H:%M:%S')}] [{env.now:.2f}] {msg}"


def events_to_occupancy(events: List[List[Event]]) -> Dict[int, List[Dict[str, float]]]:
    res = {}
    for eventlist in events:
        for ev in eventlist:
            if ev.type == EventType.charged:
                identifier = int(ev.node.identifier)
                occupancies = res.get(identifier, [])
                occupancies.append(dict(
                    t_start=ev.t_start,
                    t_end=ev.t_end,
                ))
                res[identifier] = occupancies
    for k, v in res.items():
        res[k] = sorted(v, key=lambda x: x['t_start'])
    return res


def plot_events_battery(result: SimResult, fname: str):
    """
    Plots the battery over time for the given events
    """
    events = result.events
    r_charge = result.params.r_charge.min()

    execution_times = []
    for d in range(len(events)):
        execution_times.append(events[d][-1].t_end)
    max_execution_time = max(execution_times)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, (len(events), 1), axes_pad=0.15, aspect=True, share_all=True)

    uav_colors = gen_colors(len(events))

    # Do not plot the arrival scatter plot when the amount of waypoints is too high
    n_wp_arrivals = []
    for evlist in events:
        n_wp_arrivals_for_d = sum([e.type == EventType.reached and e.node.node_type == NodeType.Waypoint for e in evlist])
        n_wp_arrivals.append(n_wp_arrivals_for_d)
    do_plot_scatter = max(n_wp_arrivals) <= 15

    station_ids = []
    for d in range(len(events)):
        for e in events[d]:
            if type(e.node) == ChargingStation:
                if e.node.identifier not in station_ids:
                    station_ids.append(e.node.identifier)
    station_colors = {}
    if len(station_ids) == 1:
        # make grey
        station_colors[station_ids[0]] = [0.5] * 3
    else:
        for i, color in zip(station_ids, gen_colors(len(station_ids))):
            station_colors[station_ids[i]] = color

    for d in range(len(events)):
        X_line = []
        Y_line = []
        X_scatter_wp = []
        Y_scatter_wp = []
        for i, e in enumerate(events[d]):
            ts = e.t_end
            X_line.append(ts)
            Y_line.append(e.battery)

            if e.type == EventType.charged:
                rect = Rectangle(
                    (e.t_start, 0),
                    e.duration,
                    1,
                    color=station_colors[e.node.identifier],
                    ec=None,
                    alpha=0.3,
                    zorder=-1
                )
                grid[d].add_patch(rect)
            elif e.type == EventType.waited:
                rect = Rectangle(
                    (e.t_start, 0),
                    e.duration,
                    1,
                    color=station_colors[e.node.identifier],
                    fill=None,
                    linewidth=0,
                    alpha=0.3,
                    hatch="/" * 6,
                    ec=station_colors[e.node.identifier],
                    zorder=-1
                )
                grid[d].add_patch(rect)
            elif e.type == EventType.reached and e.node.node_type == NodeType.Waypoint:
                X_scatter_wp.append(e.t_end)
                Y_scatter_wp.append(e.battery)

        grid[d].plot(X_line, Y_line, c=uav_colors[d])
        if do_plot_scatter:
            grid[d].scatter(X_scatter_wp, Y_scatter_wp, c=[uav_colors[d]], s=10)
        grid[d].set_ylabel(f"UAV {d + 1}", fontsize=9)
        grid[d].set_ylim([0, 1])
        grid[d].spines.right.set_visible(False)
        for ts, _ in result.schedules[d]:
            grid[d].axvline(ts, color='black', linestyle=":", alpha=0.5, zorder=1)

    # add vertical lines
    for d in range(len(events)):
        grid[d].axvline(max_execution_time, color='red', zorder=-10)
    grid[np.argmin(execution_times)].text(max_execution_time, 0.5, f'{max_execution_time:.1f}s', color='red',
                                          backgroundcolor='white', fontsize='xx-small', ha='left', zorder=-9)

    N_d = len(events)
    aspect = 0.8 / r_charge
    for d in range(N_d):
        grid[d].set_aspect(aspect)
    grid[N_d - 1].set_xlabel("Time (s)")

    # set figure height
    x = 0.6
    figheight = ((1 + grid[0].figure.subplotpars.hspace) * x * N_d - grid[0].figure.subplotpars.hspace * x) / (1 - grid[0].figure.subplotpars.bottom - (1 - grid[0].figure.subplotpars.top))
    grid[0].figure.set_figheight(figheight)

    plt.savefig(fname, bbox_inches='tight')


def plot_station_occupancy(result: SimResult, fname: str):
    """
    Plot the number of UAVs that use a charging station over time
    """
    events = result.events
    nstations = result.scenario.N_s
    r_charge = result.params.r_charge.min()
    total_duration = result.execution_time

    colors = gen_colors(nstations)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, (nstations, 1), axes_pad=0.15, aspect=True, share_all=True)

    for station in range(nstations):
        X = [0]
        Y = [0]
        cur_charged = 0

        changes = {}
        for d in range(len(events)):
            for ev in events[d]:
                if ev.type == EventType.charged and ev.node.identifier == station:
                    changes[ev.t_start] = changes.get(ev.t_start, 0) + 1
                    changes[ev.t_end] = changes.get(ev.t_end, 0) - 1

        for ts, change in sorted(changes.items()):
            prev_charged = cur_charged
            cur_charged = cur_charged + change

            X += [ts, ts]
            Y += [prev_charged, cur_charged]
        X.append(total_duration)
        Y.append(cur_charged)
        grid[station].set_ylabel(f"Station {station + 1}", fontsize=9)
        grid[station].plot(X, Y, colors[station])
        grid[station].fill_between(X, Y, facecolor=colors[station], alpha=0.2)

    # correct aspect
    aspect = 0.8 / r_charge
    for d in range(nstations):
        grid[d].set_aspect(aspect)
    grid[nstations - 1].set_xlabel("Time (s)")

    # set figure height
    x = 0.6
    figheight = ((1 + grid[0].figure.subplotpars.hspace) * x * nstations - grid[0].figure.subplotpars.hspace * x) / (1 - grid[0].figure.subplotpars.bottom - (1 - grid[0].figure.subplotpars.top))
    grid[0].figure.set_figheight(figheight)

    plt.savefig(fname, bbox_inches='tight')
