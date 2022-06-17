import logging
from enum import Enum

import simpy.exceptions

from simulate.event import Event
from simulate.node import AuxWaypoint, NodeType


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

            event = env.timeout(t_move, value=Event(env.now, "reached", node_next))

            for cb in self.pre_move_cbs:
                cb(event)
            yield event

            self.cur_node = node_next
            self.nodes = self.nodes[1:] if len(self.nodes) > 1 else []

            # wait at node
            waiting_time = self.cur_node.wt
            if waiting_time > 0:
                event = env.timeout(waiting_time, value=Event(env.now, "waited", self.cur_node))
                for cb in self.pre_wait_cbs:
                    cb(event)
                yield event

            # charge at node
            charging_time = self.cur_node.ct
            if charging_time > 0:
                event = env.timeout(charging_time, value=Event(env.now, "charged", self.cur_node))
                for cb in self.pre_charge_cbs:
                    cb(event)
                yield event


class UAV:
    def __init__(self, uav_id: int, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 battery: float = 1):
        """
        :param nodes: list of Waypoints and ChargingStations to visit in order
        :param charging_stations: list of simpy.Resources to allocate
        :param v: velocity of the UAV
        :param r_charge: charging rate
        :param r_deplete: depletion rate
        """
        self.logger = logging.getLogger(__name__)
        self.uav_id = uav_id

        self.charging_stations = charging_stations  # simpy shared resources
        self.resource = None
        self.req = None
        self.battery = battery
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.v = v

        self.t_start = 0
        self.state_type = UavStateType.Idle
        self.cur_node = None

        self.proc = None
        self.eg = None
        self.events = []
        self.waypoint_id = 0

        self.arrival_cbs = []
        self.waited_cbs = []
        self.charged_cbs = []
        self.finish_cbs = []

    def add_arrival_cb(self, cb):
        self.arrival_cbs.append(cb)

    def add_waited_cb(self, cb):
        self.waited_cbs.append(cb)

    def add_charged_cb(self, cb):
        self.charged_cbs.append(cb)

    def add_finish_cb(self, cb):
        self.finish_cbs.append(cb)

    def _get_battery(self, env):
        t_passed = env.now - self.t_start
        battery = self.battery
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
        elif self.state_type == UavStateType.Charging:
            battery = min(self.battery + t_passed * self.r_charge, 1)
        return battery

    def get_state(self, env):
        """
        Returns the position and batter charge of the UAV.
        """
        t_passed = env.now - self.t_start
        battery = self._get_battery(env)
        if self.state_type == UavStateType.Moving:
            dir_vector = self.cur_node.direction(self.eg.nodes[0])
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

    def set_schedule(self, env, pos: list, nodes: list):
        self.battery = self._get_battery(env)
        self.cur_node = AuxWaypoint(*pos)
        self.t_start = env.now

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
            self.proc.interrupt()
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
            self.events.append(ev)

    def sim(self, env):
        while True:
            try:
                self.proc = env.process(self._sim(env))
                yield self.proc
                for cb in self.finish_cbs:
                    cb()
                break
            except simpy.exceptions.Interrupt:
                pass
