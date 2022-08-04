import logging
import os.path
import pickle
from datetime import datetime
from typing import List

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
from simulate.scheduling import Scheduler
from simulate.uav import UAV, UavStateType
from simulate.util import gen_colors
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

        # for outputting simulation
        self.plot_params = {}
        self.pdfs = []
        self.solve_times = []
        self.all_schedules = {}

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
                self.debug(env, f"simulator is locking charging station [{resource_id}] for {epsilon:.2f} after a UAV finished charging")
                t_before = env.now
                req = resource.request(priority=0)
                yield req
                elapsed = env.now - t_before
                self.debug(env, f"simulator acquired lock on charging station [{resource_id}] (after {elapsed:.2f}s)")

                yield env.timeout(epsilon)
                resource.release(req)
                self.debug(env, f"simulator released lock on charging station [{resource_id}]")

            env.process(release_after_epsilon(env, epsilon, resource, resource_id))

        self.uavs = []
        for d in range(self.sc.N_d):
            uav = UAV(d, self.charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d], self.sc.positions_w[d][0])
            uav.add_release_lock_cb(release_lock_cb)
            self.uavs.append(uav)
        self.debug(env, f"visiting {self.sc.N_w - 1} waypoints per UAV in total")

        # get initial schedule
        def reschedule_cb(uavs_to_schedule):
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

            t_solve, (optimal, schedules) = self.scheduler.schedule(start_positions, batteries, state_types, uavs_to_schedule)
            self.debug(env, f"rescheduled {'non-' if not optimal else ''}optimal drone paths in {t_solve:.2}s")
            n_remaining_waypoints = [self.scheduler.n_remaining_waypoints(d) for d in range(self.sc.N_d)]
            self.solve_times.append((env.now, optimal, t_solve, n_remaining_waypoints))

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

        reschedule_cb('all')
        self.strategy.set_cb(reschedule_cb)
        strat_proc = env.process(self.strategy.sim(env))

        def uav_finished_cb(uav):
            self.debug(env, f"UAV [{uav.uav_id}] finished")
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

    def sim(self):
        try:
            self.env.run(until=self.strat_proc)
        finally:
            self.logger.info(f"finished simulation in {self.env.now:.2f}s")
            for d in range(self.sc.N_d):
                self.logger.info(f"UAV [{d}] has {self.scheduler.n_remaining_waypoints(d)} remaining waypoints")

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
                plot_events_battery([u.events for u in self.uavs], fname, r_charge=self.params.r_charge.min())

                # plot occupancy
                fname = os.path.join(self.directory, "occupancy.pdf")
                plot_station_occupancy([u.events for u in self.uavs], self.sc.N_s, self.env.now, fname, r_charge=self.params.r_charge.min())

                # output events
                event_dir = os.path.join(self.directory, "events")
                os.makedirs(event_dir, exist_ok=True)
                for d, uav in enumerate(self.uavs):
                    fname = os.path.join(event_dir, f"{d}.csv")
                    with open(fname, "w") as f:
                        f.write("t_start,t_end,duration,event_type,node_type,node_type,node_identifier,node_x,node_y,node_z,uav_id,battery_start,battery_end,depletion,forced\n")
                        for ev in uav.events:
                            t_start = ev.t_start
                            t_end = ev.t_end
                            duration = ev.duration
                            event_type = ev.type.value
                            node_type = ev.node.node_type.value
                            node_identifier = ev.node.identifier
                            node_x = ev.node.x
                            node_y = ev.node.y
                            node_z = ev.node.z
                            uav_id = ev.uav.uav_id
                            battery_start = ev.pre_battery
                            battery_end = ev.battery
                            depletion = ev.depletion
                            forced = ev.forced
                            data = [t_start, t_end, duration, event_type, node_type, node_identifier, node_x, node_y, node_z, uav_id, battery_start, battery_end, depletion, forced]
                            data = [str(v) for v in data]
                            f.write(f"{','.join(data)}\n")

                fname = os.path.join(self.directory, "schedules.pkl")
                with open(fname, 'wb') as f:
                    pickle.dump(self.all_schedules, f)

                fname = os.path.join(self.directory, "animation.html")
                events = {d: uav.events for d, uav in enumerate(self.uavs)}
                if self.params.plot_delta:
                    sa = SimulationAnimator(self.sc, events, self.all_schedules, self.params.plot_delta)
                    sa.animate(fname)

        return self.solve_times, self.env, [u.events for u in self.uavs]

    def debug(self, env, msg):
        self.logger.debug(f"[{datetime.now()}] [{env.now:.2f}] {msg}")


def plot_events_battery(events: List[List[Event]], fname: str, r_charge: float = 0.00067):

    """
    Plots the battery over time for the given events
    """
    execution_times = []
    for d in range(len(events)):
        execution_times.append(events[d][-1].t_end)
    max_execution_time = max(execution_times)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, (len(events), 1), axes_pad=0.15, aspect=True, share_all=True)

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
            ts = e.t_end
            X.append(ts)
            Y.append(e.battery)

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

        grid[d].plot(X, Y, c=uav_colors[d])
        grid[d].set_ylabel(f"UAV {d + 1}", fontsize=9)
        grid[d].set_ylim([0, 1])

    # add vertical lines
    for d in range(len(events)):
        grid[d].axvline(max_execution_time, color='red', zorder=-10)
    grid[np.argmin(execution_times)].text(max_execution_time, 0.5, f'{max_execution_time:.1f}s', color='red',
                                          backgroundcolor='white', fontsize='xx-small', ha='center', zorder=-9)

    N_d = len(events)
    aspect = 1 / r_charge
    for d in range(N_d):
        grid[d].set_aspect(aspect)
    grid[N_d - 1].set_xlabel("Time (s)")

    # set figure height
    x = 1
    figheight = ((1 + grid[0].figure.subplotpars.hspace) * x * N_d - grid[0].figure.subplotpars.hspace * x) / (1 - grid[0].figure.subplotpars.bottom - (1 - grid[0].figure.subplotpars.top))
    grid[0].figure.set_figheight(figheight)

    plt.savefig(fname, bbox_inches='tight')


def plot_station_occupancy(events: List[List[Event]], nstations: int, total_duration: float, fname: str, r_charge: float = 0.00067):
    """
    Plot the number of UAVs that use a charging station over time
    """
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
        grid[station].set_ylabel(f"Station {station+1}", fontsize=9)
        grid[station].plot(X, Y, colors[station])
        grid[station].fill_between(X, Y, facecolor=colors[station], alpha=0.2)

    # correct aspect
    aspect = 1 / r_charge
    for d in range(nstations):
        grid[d].set_aspect(aspect)
    grid[nstations - 1].set_xlabel("Time (s)")

    # set figure height
    x = 1
    figheight = ((1 + grid[0].figure.subplotpars.hspace) * x * nstations - grid[0].figure.subplotpars.hspace * x) / (1 - grid[0].figure.subplotpars.bottom - (1 - grid[0].figure.subplotpars.top))
    grid[0].figure.set_figheight(figheight)

    plt.savefig(fname, bbox_inches='tight')
