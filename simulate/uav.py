import logging
from enum import Enum

import simpy.exceptions

from simulate.event import ReachedEvent, WaitedEvent, ChargedEvent, StartedEvent, ChangedCourseEvent
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


class UAV:
    def __init__(self, uav_id: int, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 initial_pos: list,
                 battery: float = 1, B_max: float = 1):
        self.logger = logging.getLogger(__name__)
        self.uav_id = uav_id
        self.charging_stations = charging_stations
        self.v = v
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.B_max = B_max
        self.battery = battery

        self.last_known_pos = AuxWaypoint(*initial_pos)
        self.remaining_nodes = []
        self.state_type = UavStateType.Idle
        self.t_start = 0
        self.waypoint_id = 0

        self.events = []
        self.buffered_events = []

        self.proc = None
        self.resource = None
        self.req = None
        self.resource_id = None

        def add_ev_cb(ev):
            self.events.append(ev.value)

        self.arrival_cbs = [add_ev_cb]
        self.waited_cbs = [add_ev_cb]
        self.charged_cbs = [add_ev_cb]
        self.changed_course_cbs = [add_ev_cb]
        self.finish_cbs = []

    def _get_battery(self, env, offset=0):
        """
        Returns the state of the battery given the simpy environment (+ an offset)
        :param env: simpy.Environment
        :param offset: offset in seconds
        """
        t_passed = env.now - self.t_start + offset
        battery = self.battery
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
        elif self.state_type == UavStateType.Charging:
            battery = min(self.battery + t_passed * self.r_charge, 1)
        return battery

    @property
    def dest_node(self):
        return self.remaining_nodes[0]

    def get_state(self, env):
        """
        Returns the position and battery charge of the UAV.
        """
        t_passed = env.now - self.t_start
        battery = self._get_battery(env)
        if self.state_type == UavStateType.Moving:
            dir_vector = self.last_known_pos.direction(self.dest_node)
            traveled_distance = t_passed * self.v
            travel_vector = dir_vector * traveled_distance
            pos = self.last_known_pos.pos + travel_vector
            res = UavState(
                state_type=self.state_type,
                pos=pos,
                battery=battery,
            )
        elif self.state_type == UavStateType.Charging:
            res = UavState(
                state_type=self.state_type,
                pos=self.last_known_pos.pos,
                battery=battery,
            )
        elif self.state_type in [UavStateType.Waiting, UavStateType.Idle]:
            res = UavState(
                state_type=self.state_type,
                pos=self.last_known_pos.pos,
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

    def set_schedule(self, env, nodes: list):
        state = self.get_state(env)
        self.battery = state.battery

        if self.state_type == UavStateType.Charging:
            duration = env.now - self.t_start
            event = env.timeout(0, value=ChargedEvent(self.t_start, duration, self.last_known_pos, self, self.battery))
            for cb in self.charged_cbs:
                event.callbacks.append(cb)
            self.buffered_events.append(event)
        elif self.state_type == UavStateType.Waiting:
            duration = env.now - self.t_start
            event = env.timeout(0, value=WaitedEvent(self.t_start, duration, self.last_known_pos, self, self.battery))
            for cb in self.waited_cbs:
                event.callbacks.append(cb)
            self.buffered_events.append(event)
        elif self.state_type == UavStateType.Moving and nodes:
            target_cur = self.remaining_nodes[0]
            target_new = nodes[0]
            if not target_cur.same_pos(target_new):
                # course is being changed
                duration = env.now - self.t_start
                event = env.timeout(0, value=ChangedCourseEvent(self.t_start, duration, self.last_known_pos, self,
                                                                self.battery))
                for cb in self.changed_course_cbs:
                    event.callbacks.append(cb)
                self.buffered_events.append(event)

        self.remaining_nodes = nodes
        self.last_known_pos = AuxWaypoint(*state.pos)
        self.t_start = env.now

        self._release_lock(env)

        if self.proc:
            try:
                self.proc.interrupt()
                self.debug(env, f"is interrupted")
            except RuntimeError as e:
                self.debug(env, f"failed to interrupt process: {e}")

    def _release_lock(self, env):
        if self.resource and self.req:
            self.resource.release(self.req)
            self.debug(env, f"released lock ({self.resource_id})")
        self.resource = None
        self.req = None
        self.resource_id = None

    def _sim(self, env):
        """
        Simulate the following of the internal schedule of the UAV
        :return:
        """
        for ev in self.buffered_events:
            yield ev
        self.buffered_events = []

        while len(self.remaining_nodes) > 0:
            node_next = self.remaining_nodes[0]

            # move to node
            distance = self.last_known_pos.dist(node_next)
            t_move = distance / self.v

            if t_move > 0:
                self.debug(env, f"is moving from {self.last_known_pos} to {node_next}")
                self.state_type = UavStateType.Moving

                post_move_battery = self.battery - t_move * self.r_deplete
                event = env.timeout(t_move,
                                    value=ReachedEvent(env.now, t_move, node_next, self, battery=post_move_battery))
                event.callbacks += self.arrival_cbs
                yield event

                # for cb in self.arrival_cbs:
                #     cb(event)

                self.debug(env, f"reached {node_next}")
                self.state_type = UavStateType.Idle
                self.t_start = env.now
                self.battery = post_move_battery

            self.last_known_pos = node_next
            if node_next.node_type == NodeType.Waypoint:
                self.waypoint_id += 1

            self.remaining_nodes = self.remaining_nodes[1:]

            waiting_time = node_next.wt
            if waiting_time > 0:
                self.debug(env, f"is waiting at {node_next}")
                self.state_type = UavStateType.Waiting

                event = env.timeout(waiting_time,
                                    value=WaitedEvent(env.now, waiting_time, node_next, self, battery=self.battery))
                def cb(event):
                    self.debug(env, f"finished waiting at station {node_next.identifier} for {node_next.wt:.2f}s")
                    self.state_type = UavStateType.Idle
                    self.t_start = env.now
                event.callbacks.append(cb)
                event.callbacks += self.waited_cbs
                yield event

                # self.debug(env, f"finished waiting at station {node_next.identifier} for {node_next.wt:.2f}s")
                # self.state_type = UavStateType.Idle
                # self.t_start = env.now

                # for cb in self.waited_cbs:
                #     cb(event)

            # charge at node
            charging_time = node_next.ct

            if charging_time == 'full':
                # charge to full
                charging_time = (self.B_max - self.battery) / self.r_charge

            if charging_time > 0:
                # wait for station availability
                before = env.now
                self.state_type = UavStateType.Waiting

                self.resource_id = node_next.identifier
                self.resource = self.charging_stations[self.resource_id]
                self.req = self.resource.request()
                self.debug(env, f"is trying to lock for charging station {self.resource_id}")
                yield self.req
                self.debug(env, f"got lock for charging station {self.resource_id}")

                elapsed = env.now - before
                if elapsed > 0:
                    event = env.timeout(0,
                                        value=WaitedEvent(before, elapsed, node_next, self, battery=self.battery))
                    def cb(event):
                        self.state_type = UavStateType.Idle
                        self.t_start = env.now

                        self.debug(env,
                                   f"finished waiting at station {node_next.identifier} for {elapsed:.2f}s to become available")
                    event.callbacks.append(cb)
                    event.callbacks += self.waited_cbs
                    yield event

                    # self.state_type = UavStateType.Idle
                    # self.t_start = env.now
                    #
                    # self.debug(env,
                    #            f"finished waiting at station {node_next.identifier} for {elapsed:.2f}s to become available")

                    # for cb in self.waited_cbs:
                    #     cb(event)

                self.debug(env, f"is charging at {node_next}")
                self.state_type = UavStateType.Charging

                post_charge_battery = min(self.battery + charging_time * self.r_charge, 1)
                event = env.timeout(charging_time, value=ChargedEvent(env.now, charging_time, node_next, self,
                                                                      battery=post_charge_battery))
                def cb(event):
                    self._release_lock(env)
                    ct_str = charging_time if charging_time == 'until full' else f"for {charging_time:.2f}s"
                    self.debug(env, f"finished charging at station {node_next.identifier} {ct_str}")
                    self.battery = post_charge_battery
                    self.state_type = UavStateType.Idle
                    self.t_start = env.now
                event.callbacks.append(cb)
                event.callbacks += self.charged_cbs
                yield event

                # self._release_lock(env)
                # ct_str = charging_time if charging_time == 'until full' else f"for {charging_time:.2f}s"
                # self.debug(env, f"finished charging at station {node_next.identifier} {ct_str}")
                # self.battery = post_charge_battery
                # self.state_type = UavStateType.Idle
                # self.t_start = env.now

                # for cb in self.charged_cbs:
                #     cb(event)

    def debug(self, env, msg):
        self.logger.debug(f"[{env.now:.2f}] UAV [{self.uav_id}] {msg}")

    def sim(self, env):
        ev = StartedEvent(env.now, 0, self.last_known_pos, self, battery=self.battery)
        self.events.append(ev)

        while True:
            try:
                self.debug(env, f"is starting new simpy process")
                self.proc = env.process(self._sim(env))
                yield self.proc
                for cb in self.finish_cbs:
                    cb(self)
                break
            except simpy.exceptions.Interrupt:
                pass
