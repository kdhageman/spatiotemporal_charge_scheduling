import logging
from datetime import datetime
from enum import Enum
from typing import List, Callable

import jsons
import simpy.exceptions
from simpy import PriorityResource

from simulate.environment import Environment, DeterministicEnvironment
from simulate.event import ReachedEvent, WaitedEvent, ChargedEvent, StartedEvent, ChangedCourseEvent, CrashedEvent
from simulate.instruction import MoveInstruction, WaitInstruction, ChargeInstruction, InstructionType
from simulate.node import AuxWaypoint, NodeType, Node


class UavStateType(Enum):
    Idle = "idle"
    Moving = "moving"
    Waiting = "waiting"
    Charging = "charging"
    Finished = "finished"
    FinishedCharging = "finished_charging"  # indicator that the drone cannot charge again
    Crashed = "crashed"


class UavState:
    def __init__(self, state_type: UavStateType, node: Node, battery: float):
        self.state_type = state_type
        self.node = node
        self.battery = battery

    @property
    def pos_str(self):
        return f"{self.node}"


class UAV:
    def __init__(self, uav_id: int, charging_stations: List[PriorityResource], v: float, r_charge: float, r_deplete: float, initial_pos: list, battery: float = 1, B_max: float = 1):
        self.logger = logging.getLogger(__name__)
        self.uav_id = uav_id
        self.charging_stations = charging_stations
        self.v = v
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.B_max = B_max
        self.battery = battery

        self.last_known_pos = AuxWaypoint(*initial_pos)
        self.instructions = []
        self.state_type = UavStateType.Idle
        self.t_start = 0
        self.waypoint_id = 0
        self.dest_node = None

        self._events = []
        self.time_spent = {
            "moving": 0,
            "waiting": 0,
            "charging": 0,
        }

        self.proc = None
        self.resource = None
        self.req = None
        self.resource_id = None

        def add_ev_cb(ev):
            self._events.append(ev.value)

        self.arrival_cbs = [add_ev_cb]
        self.waited_cbs = [add_ev_cb]
        self.charged_cbs = [add_ev_cb]
        self.changed_course_cbs = [add_ev_cb]
        self.release_lock_cbs = []
        self.finish_cbs = []

    def _get_battery(self, env: simpy.Environment) -> float:
        """
        Returns the state of the battery given the simpy environment (+ an offset)
        :param env: simpy.Environment
        """
        t_passed = env.now - self.t_start
        battery = self.battery
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
        elif self.state_type == UavStateType.Charging:
            battery = min(self.battery + t_passed * self.r_charge, 1)
        return battery

    def get_state(self, env: simpy.Environment) -> UavState:
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
                node=AuxWaypoint(*pos),
                battery=battery,
            )
        elif self.state_type in [UavStateType.Charging, UavStateType.Finished, UavStateType.Waiting, UavStateType.Idle, UavStateType.FinishedCharging, UavStateType.Crashed]:
            res = UavState(
                state_type=self.state_type,
                node=self.last_known_pos,
                battery=battery,
            )
        else:
            raise NotImplementedError()
        return res

    def nodes_to_visit(self):
        """
        :return:
        """
        res = []
        if self.dest_node:
            res.append(self.dest_node)
        for ins in self.instructions:
            if ins.node not in res:
                res.append(ins.node)
        return res

    def events(self, env: simpy.Environment):
        res = self._events
        duration = env.now - self.t_start
        state = self.get_state(env)
        pre_battery = self.battery
        depletion = pre_battery - state.battery

        if self.state_type == UavStateType.Waiting:
            ev = WaitedEvent(self.t_start, duration, state.node, uav=self, battery=state.battery, depletion=depletion, forced=True)
            res.append(ev)
        elif self.state_type == UavStateType.Charging:
            ev = ChargedEvent(self.t_start, duration, state.node, uav=self, battery=state.battery, depletion=depletion, forced=True)
            res.append(ev)
        return res

    def add_arrival_cb(self, cb: Callable[[simpy.Event], None]):
        self.arrival_cbs.append(cb)

    def add_waited_cb(self, cb: Callable[[simpy.Event], None]):
        self.waited_cbs.append(cb)

    def add_charged_cb(self, cb: Callable[[simpy.Event], None]):
        self.charged_cbs.append(cb)

    def add_finish_cb(self, cb: Callable[[simpy.Event], None]):
        self.finish_cbs.append(cb)

    # cb(env, self.resource_id)
    def add_release_lock_cb(self, cb: Callable[[simpy.Environment, float], None]):
        self.release_lock_cbs.append(cb)

    def set_schedule(self, env: simpy.Environment, nodes: List[Node]):
        instructions = []
        for node in nodes:
            ins = MoveInstruction(node)
            if not ins.node.same_pos(self.last_known_pos) or ins.node.node_type == NodeType.Waypoint:
                # ignore auxiliary waypoints at the UAV's current position
                instructions.append(ins)

            if node.node_type == NodeType.ChargingStation:
                if node.wt:
                    ins = WaitInstruction(node, node.wt)
                    instructions.append(ins)

                if node.ct == 'full' or node.ct:
                    ins = ChargeInstruction(node, node.ct)
                    instructions.append(ins)

        if not instructions:
            return

        self.instructions = instructions

        if self.proc and self.proc.is_alive and self.state_type not in [UavStateType.Idle, UavStateType.FinishedCharging, UavStateType.Finished]:
            try:
                self.proc.interrupt()
                self.debug(env, f"is interrupted")
            except RuntimeError as e:
                self.debug(env, f"failed to interrupt process: {e}")

    def _release_lock(self, env: simpy.Environment) -> None:
        if self.resource and self.req:
            self.resource.release(self.req)
            self.debug(env, f"released lock for charging station [{self.resource_id}]")
            for cb in self.release_lock_cbs:
                cb(env, self.uav_id, self.resource_id)
        self.resource = None
        self.req = None
        self.resource_id = None

    def _sim(self, env: simpy.Environment, delta_t: float, flyenv: Environment = DeterministicEnvironment(), ):
        """
        Simulate the following of the internal schedule of the UAV
        :return:
        """
        while len(self.instructions) > 0:
            cur_instruction = self.instructions[0]
            self.instructions = self.instructions[1:]
            self.dest_node = cur_instruction.node

            self.t_start = env.now

            if cur_instruction.type == InstructionType.move:
                # move
                self.debug(env, f"is moving from {self.last_known_pos} to {self.dest_node} (distance = {self.last_known_pos.dist(self.dest_node):.2f})")
                self.state_type = UavStateType.Moving

                interrupted = False
                distance_to_dest = self.last_known_pos.dist(self.dest_node)
                reached_destination = False
                # move towards the destination node in timesteps
                while not reached_destination:
                    self.t_start = env.now
                    real_delta_t, real_distance, real_depletion, reached_destination = flyenv.move(delta_t, distance_to_dest, self.v, self.r_deplete, self.battery)
                    pre_move_battery = self.battery
                    post_move_battery = self.battery - real_depletion
                    new_pos = self.last_known_pos.pos + self.last_known_pos.direction(self.dest_node) * real_distance
                    new_node = self.dest_node if reached_destination else AuxWaypoint(*new_pos)
                    event = env.timeout(real_delta_t, value=ReachedEvent(env.now, real_delta_t, new_node, self, battery=post_move_battery, depletion=real_depletion))

                    try:
                        yield event

                        if new_node.node_type != NodeType.AuxWaypoint:
                            self.debug(env, f"reached {new_node}, B={event.value.battery:.2f} (-{event.value.depletion:.2f})")
                        self.last_known_pos = new_node
                        self.battery = event.value.battery
                        self.time_spent['moving'] += real_delta_t

                        if not reached_destination:
                            self._events.append(event.value)

                        if self.battery <= 0:
                            # terminate because the UAV is crashed
                            event = env.timeout(0, value=CrashedEvent(env.now, 0, self, self.last_known_pos, battery=0))
                            yield event
                            self._events.append(event.value)
                            self.state_type = UavStateType.Crashed
                            return
                    except simpy.Interrupt:
                        self.debug(env, f"is interrupted during moving to a node")
                        self.last_known_pos = self.get_state(env).node
                        self.battery = self._get_battery(env)
                        elapsed = env.now - self.t_start
                        self.time_spent['moving'] += elapsed

                        # change direction
                        depletion = pre_move_battery - self.battery
                        event = env.timeout(0, value=ChangedCourseEvent(self.t_start, elapsed, self.last_known_pos, self, battery=self.battery, depletion=depletion, forced=True))
                        yield event
                        self.debug(env, f"changed direction from {self.dest_node} to {self.instructions[0].node} at {self.last_known_pos}")
                        self.t_start = env.now

                        for cb in self.changed_course_cbs:
                            cb(event)

                        # step out of the loop for moving small time steps
                        interrupted = True
                        break
                    distance_to_dest = self.last_known_pos.dist(self.dest_node)

                if not interrupted:
                    # reached the destination node
                    self.state_type = UavStateType.Idle

                    if self.dest_node.node_type == NodeType.Waypoint:
                        self.waypoint_id += 1

                    for cb in self.arrival_cbs:
                        cb(event)

            elif cur_instruction.type == InstructionType.wait:
                # wait
                self.last_known_pos = self.dest_node

                waiting_time = cur_instruction.t
                self.debug(env, f"is waiting at {self.dest_node}")
                self.state_type = UavStateType.Waiting

                event = env.timeout(waiting_time, value=WaitedEvent(env.now, waiting_time, self.dest_node, self, battery=self.battery))

                try:
                    yield event

                    self.debug(env, f"finished waiting at station {self.dest_node.identifier} for {waiting_time:.2f}s")
                    self.state_type = UavStateType.Idle
                    self.time_spent['waiting'] += waiting_time

                    for cb in self.waited_cbs:
                        cb(event)
                except simpy.Interrupt:
                    elapsed = env.now - self.t_start
                    event = env.timeout(0, value=WaitedEvent(self.t_start, elapsed, self.dest_node, self, battery=self.battery, forced=True))
                    yield event

                    self.time_spent['waiting'] += elapsed
                    self.debug(env, f"forcefully finished waiting at station {self.dest_node.identifier} for {elapsed:.2f}s")
                    for cb in self.waited_cbs:
                        cb(event)

            elif cur_instruction.type == InstructionType.charge:
                # charge
                self.last_known_pos = self.dest_node

                charging_time = cur_instruction.t
                if charging_time == 'full':
                    # charge to full
                    charging_time = (self.B_max - self.battery) / self.r_charge

                # wait for station availability
                before = env.now
                self.state_type = UavStateType.Waiting

                if self.resource_id is not None and self.resource_id == self.dest_node.identifier:
                    # already have lock
                    self.debug(env, f"continuing with previously acquired lock for charging station {self.resource_id}")
                else:
                    self.resource_id = self.dest_node.identifier
                    self.resource = self.charging_stations[self.resource_id]
                    if self.resource.count == self.resource.capacity:
                        self.debug(env, f"must wait to get lock for charging station [{self.resource_id}]")
                    self.req = self.resource.request(priority=1)
                    try:
                        yield self.req
                        elapsed = before - env.now
                        if elapsed > 0:
                            self.debug(env, f"got lock for charging station [{self.resource_id}] after {elapsed:.2f}s")
                        else:
                            self.debug(env, f"got lock for charging station [{self.resource_id}] immediately")
                    except simpy.Interrupt:
                        self.req.cancel()
                        self._release_lock(env)

                        elapsed = env.now - before
                        event = env.timeout(0, value=WaitedEvent(before, elapsed, self.dest_node, self, battery=self.battery, forced=True))
                        yield event

                        self.debug(env, f"finished waiting at station {self.dest_node.identifier} for {event.value.duration:.2f}s to become available")
                        self.state_type = UavStateType.Idle
                        self.time_spent['waiting'] += elapsed

                        for cb in self.waited_cbs:
                            cb(event)

                        continue

                elapsed = env.now - before
                if elapsed > 0:
                    event = env.timeout(0, value=WaitedEvent(before, elapsed, self.dest_node, self, battery=self.battery))

                    try:
                        yield event
                        self.debug(env, f"finished waiting at station {self.dest_node.identifier} for {event.value.duration:.2f}s to become available")
                        self.state_type = UavStateType.Idle
                        self.time_spent['waiting'] += elapsed

                        for cb in self.waited_cbs:
                            cb(event)
                    except simpy.Interrupt:
                        elapsed = env.now - self.t_start
                        event = env.timeout(0, value=WaitedEvent(self.t_start, elapsed, self.dest_node, self, battery=self.battery, forced=True))
                        yield event

                        self.debug(env, f"forcefully finished waiting at station {self.dest_node.identifier} for {elapsed:.2f}s to become available")
                        self.time_spent['waiting'] += elapsed

                        for cb in self.waited_cbs():
                            cb(event)

                        continue

                remaining_charge = charging_time * self.r_charge
                self.debug(env, f"is charging at {self.dest_node} for an estimated {charging_time:.2f}s (charging {remaining_charge*100:.2f}%)")
                self.state_type = UavStateType.Charging

                # charge the UAV in timesteps
                finished_charging = False
                ct_sum = 0
                charge_interrupted = False
                while not finished_charging:
                    pre_charge_battery = self.battery
                    self.t_start = env.now
                    real_delta_t, real_charge, finished_charging = flyenv.charge(delta_t, remaining_charge, self.r_charge, self.battery)
                    post_charge_battery = self.battery + real_charge
                    event = env.timeout(real_delta_t, value=ChargedEvent(self.t_start, real_delta_t, self.dest_node, self, battery=post_charge_battery, depletion=-real_charge))

                    try:
                        yield event

                        remaining_charge -= real_charge
                        ct_sum += real_delta_t
                        self.battery = post_charge_battery
                        self.time_spent['charging'] += real_delta_t

                        if not finished_charging:
                            self._events.append(event.value)

                    except simpy.Interrupt:
                        if not self.instructions or not (self.instructions[0].type == InstructionType.charge and event.value.node.identifier == self.instructions[0].node.identifier):
                            # if the UAV is finished or does not keep charging afterwards at the same charging station, release the lock
                            self._release_lock(env)
                        t_charged = env.now - self.t_start
                        ct_sum += t_charged
                        self.battery = self.battery + self.r_charge * t_charged  # TODO: simulate this too?
                        depletion = pre_charge_battery - self.battery
                        event = env.timeout(0, value=ChargedEvent(self.t_start, t_charged, self.dest_node, self, battery=self.battery, depletion=depletion, forced=True))
                        yield event

                        self.debug(env, f"forcefully finished charging at station [{self.dest_node.identifier}] for {ct_sum:.2f}s")
                        self.state_type = UavStateType.Idle
                        self.time_spent['charging'] += t_charged

                        # step out of the loop for moving small time steps
                        charge_interrupted = True

                        for cb in self.charged_cbs:
                            cb(event)

                        break

                if not charge_interrupted:
                    self._release_lock(env)
                    ct_str = charging_time if charging_time == 'until full' else f"for {ct_sum:.2f}s"
                    self.debug(env, f"finished charging at station {self.dest_node.identifier} {ct_str}")
                    self.state_type = UavStateType.FinishedCharging

                    for cb in self.charged_cbs:
                        cb(event)

    def debug(self, env: simpy.Environment, msg: str):
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] [{env.now:.2f}] UAV [{self.uav_id}] {msg}")

    def sim(self, env: simpy.Environment, delta_t: float, flyenv: Environment = DeterministicEnvironment()):
        ev = StartedEvent(env.now, 0, self.last_known_pos, self, battery=self.battery)
        self._events.append(ev)

        self.debug(env, f"is starting new simpy process")
        self.proc = env.process(self._sim(env, delta_t, flyenv))
        yield self.proc
        if self.state_type != UavStateType.Crashed:
            self.state_type = UavStateType.Finished

        for cb in self.finish_cbs:
            cb(self)


def uav_serializer(obj: UAV, *args, **kwargs):
    return dict(id=obj.uav_id)
    # return obj.uav_id


jsons.set_serializer(uav_serializer, UAV)
