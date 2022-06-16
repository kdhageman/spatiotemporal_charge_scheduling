import logging
from enum import Enum

import simpy.exceptions

from simulate.event import Event
from simulate.node import AuxWaypoint
from simulate.parameters import Parameters


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

    def sim(self, env):
        while True:
            if len(self.nodes) == 0:
                break
            # move to node
            node_next = self.nodes[0]
            distance = self.cur_node.dist(node_next)
            t_move = distance / self.v

            event = env.timeout(t_move, value=Event(env.now, self, "reached", node_next))

            def cb(ev):
                self.cur_node = node_next
                self.nodes = self.nodes[1:] if len(self.nodes) > 1 else []

            event.callbacks.append(cb)
            yield event

            # wait at node
            waiting_time = self.cur_node.wt
            if waiting_time > 0:
                event = env.timeout(waiting_time, value=Event(env.now, self, "waited", self.cur_node))
                yield event

            # charge at node
            charging_time = self.cur_node.ct
            if charging_time > 0:
                event = env.timeout(charging_time, value=Event(env.now, self, "charged", self.cur_node))
                yield event
        for cb in self.finish_cbs:
            cb()


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

    def get_state(self, env):
        """
        Returns the position and batter charge of the UAV.
        """
        t_passed = env.now - self.t_start
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
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
            battery = min(self.battery + t_passed * self.r_charge, 1)
            res = UavState(
                state_type=self.state_type,
                pos=self.cur_node.pos,
                battery=battery,
            )
        elif self.state_type in [UavStateType.Waiting, UavStateType.Idle]:
            res = UavState(
                state_type=self.state_type,
                pos=self.cur_node.pos,
                battery=self.battery,
            )
        return res

    def set_schedule(self, env, pos: list, nodes: list):
        self.cur_node = AuxWaypoint(*pos)

        eg = _EventGenerator(pos, nodes, self.v, self.battery, self.r_deplete, self.r_charge, self.charging_stations)
        if self.proc:
            self.proc.interrupt()
        self.eg = eg

    def _sim(self, env):
        for ev in self.eg.sim(env):
            if ev.value.name == 'reached':
                def arrival_cb(ev):
                    self.cur_node = ev.value.node
                    self.battery -= ev._delay * self.r_deplete
                    self.state_type = UavStateType.Idle

                ev.callbacks.append(arrival_cb)

                for cb in self.arrival_cbs:
                    ev.callbacks.append(cb)
                self.state_type = UavStateType.Moving

            elif ev.value.name == 'waited':
                def waited_cb(_):
                    self.state_type = UavStateType.Idle

                ev.callbacks.append(waited_cb)
                for cb in self.waited_cbs:
                    ev.callbacks.append(cb)

                self.state_type = UavStateType.Waiting
            elif ev.value.name == 'charged':
                resource = self.charging_stations[self.cur_node.identifier]
                req = resource.request()
                yield req

                def charged_cb(ev):
                    self.battery = min(self.battery + ev.value.node.ct * self.r_charge, 1)
                    self.state_type = UavStateType.Idle
                    resource.release(req)

                ev.callbacks.append(charged_cb)

                for cb in self.charged_cbs:
                    ev.callbacks.append(cb)
                self.state_type = UavStateType.Charging

            yield ev
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
