import logging
from enum import Enum

import simpy.exceptions

from simulate.event import Event
from simulate.node import AuxWaypoint, NodeType, Waypoint, ChargingStation
from util.scenario import Scenario
import numpy as np


class UavStateType(Enum):
    Idle = "idle"
    Moving = "moving"
    Waiting = "waiting"
    Charging = "charging"


class UavState:
    def __init__(self, state_type: UavStateType, pos: list, battery: float):
        self.state_type = state_type
        self.pos = pos
        self.battery = battery

    @property
    def pos_str(self):
        return f"({self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f})"


class _EventGenerator:
    """
    Internal class for generating the events by executing a schedule
    """

    def __init__(self, pos, nodes, v, battery, r_deplete, r_charge, charging_stations):
        self.cur_node = AuxWaypoint(*pos)
        self.nodes = nodes
        self.v = v
        self.battery = battery
        self.r_deplete = r_deplete
        self.r_charge = r_charge
        self.charging_stations = charging_stations
        self.pre_move_cbs = []
        self.pre_charge_cbs = []
        self.pre_wait_cbs = []

    def add_pre_move_cb(self, cb):
        self.pre_move_cbs.append(cb)

    def add_pre_charge_cb(self, cb):
        self.pre_charge_cbs.append(cb)

    def add_pre_wait_cb(self, cb):
        self.pre_wait_cbs.append(cb)

    def sim(self, env):
        while True:
            if len(self.nodes) == 0:
                break
            # move to node
            node_next = self.nodes[0]
            distance = self.cur_node.dist(node_next)
            t_move = distance / self.v

            if t_move > 0:
                event = env.timeout(t_move, value=Event(env.now, "reached", node_next,
                                                        battery=self.battery - t_move * self.r_deplete))

                for cb in self.pre_move_cbs:
                    cb(event)
                yield event
                self.battery -= t_move * self.r_deplete

            self.cur_node = node_next
            self.nodes = self.nodes[1:] if len(self.nodes) > 1 else []

            # wait at node
            waiting_time = self.cur_node.wt
            if waiting_time > 0:
                event = env.timeout(waiting_time, value=Event(env.now, "waited", self.cur_node, battery=self.battery))
                for cb in self.pre_wait_cbs:
                    cb(event)
                yield event

            # charge at node
            charging_time = self.cur_node.ct
            if charging_time > 0:
                event = env.timeout(charging_time, value=Event(env.now, "charged", self.cur_node,
                                                               battery=self.battery + charging_time * self.r_charge))
                for cb in self.pre_charge_cbs:
                    cb(event)
                yield event
                self.battery += charging_time * self.r_charge


class UAV:
    def __init__(self, uav_id: int, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 battery: float = 1):
        self.logger = logging.getLogger(__name__)
        self.uav_id = uav_id
        self.charging_stations = charging_stations
        self.v = v
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.battery = battery

        self.cur_node = None
        self.state_type = UavStateType.Idle
        self.t_start = 0

        self.events = []

        def add_ev_cb(ev):
            self.events.append(ev)

        self.arrival_cbs = [add_ev_cb]
        self.waited_cbs = [add_ev_cb]
        self.charged_cbs = [add_ev_cb]
        self.finish_cbs = []

    def _get_battery(self, env):
        t_passed = env.now - self.t_start
        battery = self.battery
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
        elif self.state_type == UavStateType.Charging:
            battery = min(self.battery + t_passed * self.r_charge, 1)
        return battery

    @property
    def dest_node(self):
        raise NotImplementedError

    def get_state(self, env):
        """
        Returns the position and battery charge of the UAV.
        """
        t_passed = env.now - self.t_start
        battery = self._get_battery(env)
        if self.state_type == UavStateType.Moving:
            dir_vector = self.cur_node.direction(self.dest_node)
            traveled_distance = t_passed * self.v
            travel_vector = dir_vector * traveled_distance
            pos = self.cur_node.pos + travel_vector
            res = UavState(
                state_type=self.state_type,
                pos=pos,
                battery=battery,
            )
        elif self.state_type == UavStateType.Charging:
            res = UavState(
                state_type=self.state_type,
                pos=self.cur_node.pos,
                battery=battery,
            )
        elif self.state_type in [UavStateType.Waiting, UavStateType.Idle]:
            res = UavState(
                state_type=self.state_type,
                pos=self.cur_node.pos,
                battery=battery,
            )
        return res

    def add_arrival_cb(self, cb):
        self.arrival_cbs.append(cb)

    def add_waited_cb(self, cb):
        self.waited_cbs.append(cb)

    def add_charged_cb(self, cb):
        self.charged_cbs.append(cb)

    def add_finish_cb(self, cb):
        self.finish_cbs.append(cb)


class MilpUAV(UAV):
    def __init__(self, uav_id: int, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 battery: float = 1):
        """
        :param nodes: list of Waypoints and ChargingStations to visit in order
        :param charging_stations: list of simpy.Resources to allocate
        :param v: velocity of the UAV
        :param r_charge: charging rate
        :param r_deplete: depletion rate
        """
        super().__init__(uav_id, charging_stations, v, r_charge, r_deplete, battery=battery)
        self.resource = None
        self.req = None

        self.proc = None
        self.eg = None
        self.waypoint_id = 0

    @property
    def dest_node(self):
        return self.eg.nodes[0]

    def changes_course(self, nodes: list):
        """
        Returns whether the current drone changes its course towards given the new schedule
        :param nodes:
        :return:
        """
        non_aux_nodes = [n for n in nodes if n.node_type != NodeType.AuxWaypoint]
        # TODO: fix bug in 'test_simulator_no_charging' test
        if len(self.eg.nodes) == 0 and len(non_aux_nodes) > 0:
            return True
        return not self.eg.nodes[0].equal_pos(non_aux_nodes[0])

    def set_schedule(self, env, pos: list, nodes: list):
        self.battery = self._get_battery(env)
        self.cur_node = AuxWaypoint(*pos)
        self.t_start = env.now

        if self.state_type == UavStateType.Idle:
            # add start event
            event = Event(env.now, "started", self.cur_node, self)
            self.events.append(env.timeout(0, value=event))
        elif self.state_type == UavStateType.Moving and self.changes_course(nodes):
            # add event with current position to events list when moving
            self.logger.debug(
                f"[{env.now:.2f}] UAV [{self.uav_id}] changed course from {self.eg.nodes[0]} to {nodes[0]}")
            event = Event(env.now, "changed_course", self.cur_node, self)
            self.events.append(env.timeout(0, value=event))

        eg = _EventGenerator(pos, nodes, self.v, self.battery, self.r_deplete, self.r_charge, self.charging_stations)

        def pre_move_cb(ev):
            self.state_type = UavStateType.Moving

        def pre_charge_cb(ev):
            self.state_type = UavStateType.Charging

            self.resource = self.charging_stations[ev.value.node.identifier]
            self.req = self.resource.request()
            yield self.req

        def pre_wait_cb(ev):
            self.state_type = UavStateType.Waiting

        eg.add_pre_move_cb(pre_move_cb)
        eg.add_pre_charge_cb(pre_charge_cb)
        eg.add_pre_wait_cb(pre_wait_cb)
        if self.proc:
            try:
                self.proc.interrupt()
            except RuntimeError as e:
                self.logger.warning(f"[{env.now}] failed to interrupt process: {e}")
        self.eg = eg

    def _sim(self, env):
        for ev in self.eg.sim(env):
            if ev.value.name == 'reached':
                self.logger.debug(
                    f"[{env.now:.2f}] drone [{self.uav_id}] is moving from {self.cur_node} to {self.eg.nodes[0]}")
                self.state_type = UavStateType.Moving

            elif ev.value.name == 'waited':
                self.logger.debug(f"[{env.now:.2f}] drone [{self.uav_id}] is waiting at {self.cur_node}")
                self.state_type = UavStateType.Waiting
            elif ev.value.name == 'charged':

                self.state_type = UavStateType.Waiting

                self.logger.debug(f"[{env.now:.2f}] drone [{self.uav_id}] is charging at {self.cur_node}")
                self.state_type = UavStateType.Charging

            ev.value.uav = self

            yield ev

            self.state_type = UavStateType.Idle

            if ev.value.name == 'reached':
                self.cur_node = ev.value.node
                self.battery -= ev._delay * self.r_deplete
                if ev.value.node.node_type == NodeType.Waypoint:
                    self.waypoint_id += 1
                for cb in self.arrival_cbs:
                    cb(ev)
            elif ev.value.name == 'waited':
                for cb in self.waited_cbs:
                    cb(ev)
            elif ev.value.name == 'charged':
                self.battery = min(self.battery + ev.value.node.ct * self.r_charge, 1)
                if self.resource:
                    self.resource.release(self.req)
                for cb in self.charged_cbs:
                    cb(ev)

            self.t_start = env.now

    def sim(self, env):
        while True:
            try:
                self.proc = env.process(self._sim(env))
                yield self.proc
                for cb in self.finish_cbs:
                    cb(self)
                break
            except simpy.exceptions.Interrupt:
                pass


class NaiveUAV(UAV):
    def __init__(self, uav_id: int, sc: Scenario, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 B_min: float, battery: float = 1):
        super().__init__(uav_id, charging_stations, v, r_charge, r_deplete, battery=battery)
        self.logger = logging.getLogger(__name__)
        self.sc = sc
        self.B_min = B_min
        self.n_visited = 0
        self._dest_node = None
        self.cur_node = AuxWaypoint(*self.sc.positions_w[self.uav_id][0])

    @property
    def n_remaining(self):
        return self.sc.N_w - self.n_visited - 1

    @property
    def dest_node(self):
        return self._dest_node

    def sim(self, env):
        def set_idle(_):
            self.state_type = UavStateType.Idle

        for w_s in range(self.sc.N_w_s):
            pos_wp_next = self.sc.positions_w[self.uav_id][w_s + 1]
            station_idx, dist_to_closest_station = self._dist_to_closest_station(w_s)

            if w_s == self.sc.N_w_s - 1:
                # last waypoint
                self.state_type = UavStateType.Moving
                dist_to_wp_next = self.sc.D_N[self.uav_id, -1, w_s]
                t_to_wp_next = dist_to_wp_next / self.v
                depletion = t_to_wp_next * self.r_deplete
                if self.battery - depletion > self.B_min:
                    # move to last waypoint
                    self._dest_node = AuxWaypoint(*pos_wp_next)

                    node = Waypoint(*pos_wp_next)
                    ev = env.timeout(t_to_wp_next,
                                     value=Event(env.now, "reached", node, uav=self, battery=self.battery - depletion))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.n_visited += 1

                    # reached the end, so break from loop
                    break
                else:
                    # move to closest charging station
                    self.state_type = UavStateType.Moving
                    self._dest_node = AuxWaypoint(*self.sc.positions_S[station_idx])
                    t_to_closest_station = dist_to_closest_station / self.v

                    pos_station = self.sc.positions_S[station_idx]
                    node = ChargingStation(*pos_station, identifier=station_idx)
                    ev = env.timeout(t_to_closest_station,
                                     value=Event(env.now, "reached", node, uav=self, battery=self.battery))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.cur_node = AuxWaypoint(*self.sc.positions_S[station_idx])
                    self.battery -= t_to_closest_station * self.r_deplete

                    # wait for station availability
                    resource = self.charging_stations[station_idx]
                    req = resource.request()
                    t_start = env.now

                    yield req

                    waited_time = t_start - env.now
                    if waited_time > 0:
                        self.state_type = UavStateType.Waiting
                        node.wt = waited_time
                        ev = env.timeout(0, value=Event(env.now, "waited", node, uav=self))
                        for cb in self.waited_cbs:
                            ev.callbacks.append(cb)
                        ev.callbacks.append(set_idle)
                        yield ev

                    # charge at station
                    self.state_type = UavStateType.Charging
                    amount_to_charge = 1 - self.battery
                    t_to_charge = amount_to_charge / self.r_charge
                    node.ct = t_to_charge
                    ev = env.timeout(t_to_charge, value=Event(env.now, "charged", node, uav=self, battery=1))
                    for cb in self.charged_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.battery = 1
                    resource.release(req)

                    # move to next waypoint
                    self._dest_node = AuxWaypoint(*pos_wp_next)
                    self.state_type = UavStateType.Moving
                    dist_to_wp_next = self.sc.D_W[self.uav_id, station_idx, w_s]
                    t_to_wp_next = dist_to_wp_next / self.v
                    depletion = t_to_wp_next * self.r_deplete

                    node = Waypoint(*pos_wp_next)
                    ev = env.timeout(t_to_wp_next,
                                     value=Event(env.now, "reached", node, uav=self, battery=self.battery - depletion))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.cur_node = AuxWaypoint(*pos_wp_next)
                    self.n_visited += 1
                    self.battery -= depletion
            else:
                dist_to_wp_next = self.sc.D_N[self.uav_id, -1, w_s]
                depletion = (dist_to_wp_next + dist_to_closest_station) / self.v * self.r_deplete
                if self.battery - depletion > self.B_min:
                    # visit waypoint directly
                    self._dest_node = AuxWaypoint(*pos_wp_next)
                    self.state_type = UavStateType.Moving
                    t_to_wp_next = dist_to_wp_next / self.v
                    node = Waypoint(*pos_wp_next)
                    ev = env.timeout(t_to_wp_next, value=Event(env.now, "reached", node, uav=self,
                                                               battery=self.battery - t_to_wp_next * self.r_deplete))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.cur_node = AuxWaypoint(*pos_wp_next)
                    self.n_visited += 1
                    self.battery -= t_to_wp_next * self.r_deplete
                else:
                    # move to closest charging station
                    self.state_type = UavStateType.Moving
                    self._dest_node = AuxWaypoint(*self.sc.positions_S[station_idx])
                    t_to_closest_station = dist_to_closest_station / self.v

                    pos_station = self.sc.positions_S[station_idx]
                    node = ChargingStation(*pos_station, identifier=station_idx)
                    ev = env.timeout(t_to_closest_station, value=Event(env.now, "reached", node, uav=self,
                                                                       battery=self.battery - t_to_closest_station * self.r_deplete))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.cur_node = AuxWaypoint(*self.sc.positions_S[station_idx])
                    self.battery -= t_to_closest_station * self.r_deplete

                    # wait for station availability
                    resource = self.charging_stations[station_idx]
                    req = resource.request()
                    t_start = env.now

                    yield req

                    waited_time = env.now - t_start
                    if waited_time > 0:
                        self.state_type = UavStateType.Waiting
                        node.wt = waited_time
                        ev = env.timeout(0, value=Event(env.now, "waited", node, uav=self, battery=self.battery))
                        for cb in self.waited_cbs:
                            ev.callbacks.append(cb)
                        ev.callbacks.append(set_idle)
                        yield ev

                    # charge at station
                    self.state_type = UavStateType.Charging
                    amount_to_charge = 1 - self.battery
                    t_to_charge = amount_to_charge / self.r_charge
                    node.ct = t_to_charge
                    ev = env.timeout(t_to_charge, value=Event(env.now, "charged", node, uav=self, battery=1))
                    for cb in self.charged_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.battery = 1
                    resource.release(req)

                    # move to next waypoint
                    self._dest_node = AuxWaypoint(*pos_wp_next)
                    self.state_type = UavStateType.Moving
                    dist_to_wp_next = self.sc.D_W[self.uav_id, station_idx, w_s]
                    t_to_wp_next = dist_to_wp_next / self.v
                    depletion = t_to_wp_next * self.r_deplete

                    node = Waypoint(*pos_wp_next)
                    ev = env.timeout(t_to_wp_next,
                                     value=Event(env.now, "reached", node, uav=self, battery=self.battery - depletion))
                    for cb in self.arrival_cbs:
                        ev.callbacks.append(cb)
                    ev.callbacks.append(set_idle)
                    yield ev

                    self.cur_node = AuxWaypoint(*pos_wp_next)
                    self.n_visited += 1
                    self.battery -= depletion

        for cb in self.finish_cbs:
            cb(self)

    def _dist_to_closest_station(self, w_s):
        distances = self.sc.D_N[self.uav_id, :-1, w_s]
        return np.argmin(distances), np.min(distances)
