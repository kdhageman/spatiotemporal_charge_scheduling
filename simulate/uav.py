import logging

from prometheus_client import Enum

from simulate.event import Event
from simulate.node import NodeType, AuxWaypoint


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
        self.nodes = None
        self.cur_node = None

        self.charging_stations = charging_stations  # simpy shared resources

        self.t_start = 0
        self.state_type = UavStateType.Idle
        self.battery = battery
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.v = v

        self.node_idx_sched = 0  # node index in schedule
        self.node_idx_mission = 0  # node index for total mission
        self.events = []

    def get_state(self, env):
        """
        Returns the position and batter charge of the UAV.
        """
        t_passed = env.now - self.t_start
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
            dir_vector = self.cur_node.direction(self.nodes[0])
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
        # TODO: stop already yielding event, and start a new one
        self.cur_node = AuxWaypoint(*pos)
        self.nodes = nodes

    # TODO: add event for when the UAV crashes
    def sim(self, env, callbacks=[], finish_callbacks=[]):
        while True:
            if len(self.nodes) == 0:
                break
            node_next = self.nodes[0]
            distance = self.cur_node.dist(node_next)
            t_move = distance / self.v

            # move to node
            self.logger.debug(
                f"[{env.now:.1f}] UAV {self.uav_id} is moving from {self.cur_node} to {node_next} [expected duration = {t_move:.1f}]")
            self.state_type = UavStateType.Moving
            self.t_start = env.now
            event = env.timeout(t_move, value=Event(env.now, self, "reached", node_next))

            def cb(event):
                self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} reached {event.value.node}")
                self.cur_node = event.value.node
                self.nodes = self.nodes[1:] if len(self.nodes) > 1 else []
                self.battery -= t_move * self.r_deplete
                self.state_type = UavStateType.Idle
                self.events.remove(event)

            event.callbacks.append(cb)
            for cb in callbacks:
                event.callbacks.append(cb)

            self.events.append(event)
            yield event

            # wait at node
            waiting_time = self.cur_node.wt
            if waiting_time > 0:
                self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} is waiting at {self.cur_node}")

                self.state_type = UavStateType.Waiting
                self.t_start = env.now
                event = env.timeout(waiting_time, value=Event(env.now, self, "waited", self.cur_node))

                def cb(_):
                    self.state_type = UavStateType.Idle
                    self.events.remove(event)

                event.callbacks.append(cb)
                for cb in callbacks:
                    event.callbacks.append(cb)

                self.events.append(event)
                yield event

            # charge at node
            charging_time = self.cur_node.ct
            if charging_time > 0:
                self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} is charging at {self.cur_node}")

                self.state_type = UavStateType.Charging
                self.t_start = env.now
                resource = self.charging_stations[self.cur_node.identifier]
                req = resource.request()
                yield req

                event = env.timeout(charging_time,
                                    value=Event(env.now, self, "charged", self.cur_node))

                def cb(_):
                    self.battery = min(self.battery + charging_time * self.r_charge, 1)
                    self.state_type = UavStateType.Idle
                    resource.release(req)
                    self.events.remove(event)

                event.callbacks.append(cb)
                for cb in callbacks:
                    event.callbacks.append(cb)

                self.events.append(event)
                yield event
        for cb in finish_callbacks:
            cb()
