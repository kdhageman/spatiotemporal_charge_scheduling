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
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0, 0)

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


class EventValue:
    def __init__(self, name, value):
        self.name = name
        self.value = value


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
    def __init__(self, charging_stations: list, v: float, r_charge: float, r_deplete: float,
                 battery: float = 1):
        """
        :param nodes: list of Waypoints and ChargingStations to visit in order
        :param charging_stations: list of simpy.Resources to allocate
        :param v: velocity of the UAV
        :param r_charge: charging rate
        :param r_deplete: depletion rate
        """
        self.nodes = None

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
    def sim(self, env, callbacks=[]):
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
            event = env.timeout(t_move, value=EventValue("reached", node_next))
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
                event = env.timeout(waiting_time, value=EventValue("waited", node_next))
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

                event = env.timeout(charging_time, value=EventValue("charged", node_next))
                for cb in callbacks:
                    event.callbacks.append(cb)

                def cb(_):
                    self.battery = min(self.battery + charging_time * self.r_charge, 1)
                    self.state_type = UavStateType.Idle
                    resource.release(req)

                event.callbacks.append(cb)
                yield event


class TimeStepper:
    def __init__(self, interval):
        self.interval = interval

    def sim(self, env, callbacks=[]):
        while True:
            event = env.timeout(self.interval)
            for cb in callbacks:
                event.callbacks.append(cb)
            yield event


class Scheduler:
    def __init__(self, params: Parameters, scenario: Scenario):
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
        _ = solver.solve(model, tee=True)
        # TODO: handle solution of model solving

        res = []
        for d in model.d:
            nodes = [
                Waypoint(*self.scenario.positions_w[d][0])
            ]
            for w_s in model.w_s:
                n = model.P_np[d, :, w_s].tolist().index(1)
                if n < model.N_s:
                    ct = model.C_np[d][w_s]
                    wt = model.W_np[d][w_s]
                    nodes.append(
                        ChargingStation(*self.scenario.positions_S[n], n, wt, ct)
                    )
                nodes.append(
                    Waypoint(*self.scenario.positions_w[d][w_s + 1])
                )
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
        self.scheduler_cls = scheduler_cls
        self.delta = delta
        self.params = params
        self.sc = sc
        self.W = W

        self.timestepper = TimeStepper(delta)

    def sim(self):
        env = simpy.Environment()

        # prepare waypoint indices
        current_waypoint_idx = []
        for d in range(self.sc.N_d):
            current_waypoint_idx.append(0)

        # prepare shared resources
        charging_stations = []
        for s in range(self.sc.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sc.N_d):
            uav = UAV(charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        # convert scenario
        positions_S = self.sc.positions_S
        positions_w = []
        for d in range(self.sc.N_d):
            positions_w.append(self.sc.positions_w[d][:self.W])
        sc = Scenario(positions_S, positions_w)

        # get initial schedule
        scheduler = self.scheduler_cls(self.params, self.sc)
        schedules = scheduler.schedule()

        for i, nodes in enumerate(schedules):
            uavs[i].set_schedule(nodes)

        # define callbacks
        def uav_cb(event):
            print(f"{event}")

        # run simulation
        for uav in uavs:
            env.process(uav.sim(env, callbacks=[uav_cb]))
        env.run()

        print(env.now)