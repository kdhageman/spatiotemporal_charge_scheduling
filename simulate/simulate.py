import copy
import logging
from enum import Enum
import numpy as np
import simpy
from pyomo.opt import SolverFactory
from pyomo_models.multi_uavs import MultiUavModel
from util.distance import dist3
from util.scenario import Scenario


class NodeType(Enum):
    Waypoint = "waypoint"
    ChargingStation = "charging_station"


class Node:
    def __init__(self, x, y, z, wt, ct):
        self.x = x
        self.y = y
        self.z = z
        self.wt = wt
        self.ct = ct

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    def dist(self, other):
        return dist3(self.pos, other.pos)

    def direction(self, other):
        """
        Unit vector in the direction of the other node
        """
        dir_vector = other.pos - self.pos
        return dir_vector / np.linalg.norm(dir_vector)

    def node_type(self):
        raise NotImplementedError


class ChargingStation(Node):
    def __init__(self, x, y, z, identifier, wt=0, ct=0):
        super().__init__(x, y, z, wt, ct)
        self.identifier = identifier

    def node_type(self):
        return NodeType.ChargingStation


class Waypoint(Node):
    def __init__(self, x, y, z, aux=False):
        super().__init__(x, y, z, 0, 0)
        self.aux = aux

    def node_type(self):
        return NodeType.Waypoint


class Schedule:
    def __init__(self, decisions: np.ndarray, charging_times: np.ndarray, waiting_times: np.ndarray):
        assert (decisions.ndim == 2)
        assert (charging_times.ndim == 1)
        assert (waiting_times.ndim == 1)

        self.decisions = decisions
        self.charging_times = charging_times
        self.waiting_times = waiting_times


class Parameters:
    def __init__(self, v: float, r_charge: float, r_deplete: float, B_start: float, B_min: float, B_max: float,
                 epsilon: float = 0.1):
        self.v = np.array(v)
        self.r_charge = np.array(r_charge)
        self.r_deplete = np.array(r_deplete)
        self.B_start = np.array(B_start)
        self.B_min = np.array(B_min)
        self.B_max = np.array(B_max)
        self.epsilon = np.array(epsilon)

    def as_dict(self):
        return dict(
            v=self.v,
            r_charge=self.r_charge,
            r_deplete=self.r_deplete,
            B_start=self.B_start,
            B_min=self.B_min,
            B_max=self.B_max,
            epsilon=self.epsilon,
        )

    def copy(self):
        return copy.deepcopy(self)


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


class Event:
    def __init__(self, ts, uav, name, battery, node):
        self.ts = ts
        self.uav = uav
        self.name = name
        self.battery = battery
        self.node = node


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
        self.uav_id = uav_id
        self.nodes = None
        self.start_node = None

        self.charging_stations = charging_stations  # simpy shared resources

        self.t_start = 0
        self.state_type = None
        self.battery = battery
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.v = v
        self.node_idx = 0  # node index

    def set_schedule(self, nodes: list):
        self.nodes = nodes
        self.start_node = nodes[0]
        self.node_idx = 0

    def get_state(self, env):
        """
        Returns the position and batter charge of the UAV.
        """
        t_passed = env.now - self.t_start
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
            start_pos = self.nodes[self.node_idx].pos
            dir_vector = self.nodes[self.node_idx].direction(self.nodes[self.node_idx + 1])
            traveled_distance = t_passed * self.v
            travel_vector = dir_vector * traveled_distance
            pos = start_pos + travel_vector
            res = UavState(
                state_type=self.state_type,
                pos=pos,
                battery=battery,
            )
        elif self.state_type == UavStateType.Charging:
            battery = min(self.battery + t_passed * self.r_charge, 1)
            res = UavState(
                state_type=self.state_type,
                pos=self.nodes[self.node_idx].pos,
                battery=battery,
            )
        elif self.state_type in [UavStateType.Waiting, UavStateType.Idle]:
            res = UavState(
                state_type=self.state_type,
                pos=self.nodes[self.node_idx].pos,
                battery=self.battery,
            )
        return res

    # TODO: add event for when the UAV crashes
    def sim(self, env, callbacks=[], finish_callbacks=[]):
        while True:
            if self.node_idx >= len(self.nodes) - 1:
                break
            node_cur = self.nodes[self.node_idx]
            node_next = self.nodes[self.node_idx + 1]
            distance = node_cur.dist(node_next)
            t_move = distance / self.v

            # move to node
            self.state_type = UavStateType.Moving
            self.t_start = env.now
            event = env.timeout(t_move, value=Event(env.now, self.uav_id, "reached", self.battery, node_next))
            for cb in callbacks:
                event.callbacks.append(cb)

            def cb(_):
                self.node_idx += 1
                self.battery -= t_move * self.r_deplete
                self.state_type = UavStateType.Idle

            event.callbacks.append(cb)
            yield event

            # wait at node
            waiting_time = node_next.wt
            if waiting_time > 0:
                self.state_type = UavStateType.Waiting
                self.t_start = env.now
                event = env.timeout(waiting_time, value=Event(env.now, self.uav_id, "waited", self.battery, node_next))
                for cb in callbacks:
                    event.callbacks.append(cb)

                def cb(_):
                    self.state_type = UavStateType.Idle

                event.callbacks.append(cb)
                yield event

            # charge at node
            charging_time = node_next.ct
            if charging_time > 0:
                self.state_type = UavStateType.Charging
                self.t_start = env.now
                resource = self.charging_stations[node_next.identifier]
                req = resource.request()
                yield req

                event = env.timeout(charging_time,
                                    value=Event(env.now, self.uav_id, "charged", self.battery, node_next))
                for cb in callbacks:
                    event.callbacks.append(cb)

                def cb(_):
                    self.battery = min(self.battery + charging_time * self.r_charge, 1)
                    self.state_type = UavStateType.Idle
                    resource.release(req)

                event.callbacks.append(cb)
                yield event
        for cb in finish_callbacks:
            cb()


class TimeStepper:
    def __init__(self, interval):
        self.interval = interval

    def sim(self, env, callbacks=[]):
        while True:
            event = env.timeout(self.interval)
            for cb in callbacks:
                event.callbacks.append(cb)
            try:
                yield event
            except simpy.exceptions.Interrupt:
                break


class NotSolvableException(Exception):
    pass


class Scheduler:
    def __init__(self, params: Parameters, scenario: Scenario):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.scenario = scenario

    def schedule(self, solver=SolverFactory("gurobi")):
        """
        Return the most recent schedule
        :return: path to follow, dim = N_d x (W - 1) x (N_s + 1)
        :return: charging times, dim = N_d x (W - 1)
        :return: waiting times, dim = N_d x (W - 1)
        """
        model = MultiUavModel(scenario=self.scenario, parameters=self.params.as_dict())
        solution = solver.solve(model)
        if solution['Solver'][0]['Termination condition'] != 'optimal':
            raise NotSolvableException("non-optimal solution")
        solve_time = solution['Solver'][0]['Time']
        self.logger.debug(f"solved MILP in {solve_time:.2f}s")

        res = []
        for d in model.d:
            start_pos = self.scenario.positions_w[d][0]
            nodes = [
                Waypoint(*self.scenario.positions_w[d][0], aux=True)
            ]
            for w_s in model.w_s:
                n = model.P_np[d, :, w_s].tolist().index(1)
                if n < model.N_s:
                    ct = model.C_np[d][w_s]
                    wt = model.W_np[d][w_s]
                    nodes.append(
                        ChargingStation(*self.scenario.positions_S[n], n, wt, ct)
                    )
                wp = Waypoint(*self.scenario.positions_w[d][w_s + 1])
                if np.array_equal(wp.pos, start_pos):
                    wp.aux = True
                nodes.append(wp)
            res.append(nodes)
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


class Simulator:
    def __init__(self, scheduler_cls, params: Parameters, sc: Scenario, delta: float, W: int):
        self.logger = logging.getLogger(__name__)
        self.scheduler_cls = scheduler_cls
        self.delta = delta
        self.params = params
        self.sc = sc
        self.W = W

        self.timestepper = TimeStepper(delta)

        # prepare waypoint indices
        self.current_waypoint_idx = []
        self.reset_waypoint_indices()

        # used in callbacks
        self.remaining = []

        # maintain history of simulation
        self.events = []
        for _ in range(self.sc.N_d):
            self.events.append([])

    def reset_waypoint_indices(self):
        self.current_waypoint_idx = []
        for d in range(self.sc.N_d):
            self.current_waypoint_idx.append(0)

    def sim(self):
        env = simpy.Environment()

        # prepare waypoint indices
        self.reset_waypoint_indices()

        # prepare shared resources
        charging_stations = []
        for s in range(self.sc.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sc.N_d):
            uav = UAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sc.N_w} waypoints per UAV in total")
        # convert scenario
        sc = self.prepare_scenario(first=True)
        self.logger.debug("")

        # get initial schedule
        scheduler = self.scheduler_cls(self.params, sc)
        schedules = scheduler.schedule()

        for i, nodes in enumerate(schedules):
            uavs[i].set_schedule(nodes)

        def uav_cb(event):
            if event.value.name == "reached" and type(event.value.node) == Waypoint:
                if not event.value.node.aux:
                    self.current_waypoint_idx[event.value.uav] += 1
                    wp = self.current_waypoint_idx[event.value.uav]
                else:
                    wp = f"{self.current_waypoint_idx[event.value.uav]} (aux)"
                self.logger.debug(f"[{event.value.ts:.1f}] UAV {event.value.uav} reached a new waypoint ({wp})")
                self.events[event.value.uav].append(event.value)

        def ts_cb(event):
            positions = []
            batteries = []
            for uav in uavs:
                state = uav.get_state(env)
                positions.append(state.pos)
                batteries.append(state.battery)
            sc = self.prepare_scenario(positions)

            params = self.params.copy()
            params.B_start = np.array(batteries)

            scheduler = Scheduler(params, sc)
            schedules = scheduler.schedule()

            for i, nodes in enumerate(schedules):
                uavs[i].set_schedule(nodes)

        ts = env.process(self.timestepper.sim(env, callbacks=[ts_cb]))

        self.remaining = self.sc.N_d

        def uav_finished_cb():
            self.logger.debug("uav finished")
            self.remaining -= 1
            if self.remaining == 0:
                ts.interrupt()

        # run simulation
        for uav in uavs:
            env.process(uav.sim(env, callbacks=[uav_cb], finish_callbacks=[uav_finished_cb]))
        env.run(until=100)

        return env

    def prepare_scenario(self, positions: list = [], first=False):
        """
        Return a scenario based on the current state of the UAVs
        :param positions: (x,y,z)-coorindates of all UAVs
        :param first: true if the first W waypoints need to be used
        :return:
        """
        positions_S = self.sc.positions_S
        positions_w = []
        for d in range(self.sc.N_d):
            if not first:
                waypoints = [positions[d]]
                waypoints += self.sc.positions_w[d][
                             self.current_waypoint_idx[d]:self.current_waypoint_idx[d] + self.W - 1]
            else:
                waypoints = self.sc.positions_w[d][:self.W]
            # pad waypoints with first waypoint
            while len(waypoints) != self.W:
                waypoints = [waypoints[0]] + waypoints
            positions_w.append(waypoints)
        return Scenario(positions_S, positions_w)

    def plot(self, axes):
        assert self.sc.N_d == len(axes)
