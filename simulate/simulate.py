import copy
import logging
from enum import Enum
import numpy as np
import simpy
from matplotlib import pyplot as plt
from pyomo.opt import SolverFactory
from pyomo_models.multi_uavs import MultiUavModel
from util.decorators import timed
from util.distance import dist3
from util.scenario import Scenario


class NodeType(Enum):
    AuxWaypoint = "auxiliary_waypoint"
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
        if np.linalg.norm(dir_vector) == 0:
            return np.array([0, 0, 0])
        return dir_vector / np.linalg.norm(dir_vector)

    @property
    def node_type(self):
        raise NotImplementedError

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"


class ChargingStation(Node):
    def __init__(self, x, y, z, identifier, wt=0, ct=0):
        super().__init__(x, y, z, wt, ct)
        self.identifier = identifier

    @property
    def node_type(self):
        return NodeType.ChargingStation

    def __repr__(self):
        return f"charging station ({self.identifier}) {super().__repr__()}"


class Waypoint(Node):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0, 0)

    @property
    def node_type(self):
        return NodeType.Waypoint


class AuxWaypoint(Node):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0, 0)

    @property
    def node_type(self):
        return NodeType.AuxWaypoint


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
    def __init__(self, ts, uav, name, node):
        self.ts = ts
        self.uav = uav
        self.name = name
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
        self.logger = logging.getLogger(__name__)
        self.uav_id = uav_id
        self.nodes = None
        self.start_node = None

        self.charging_stations = charging_stations  # simpy shared resources

        self.t_start = 0
        self.state_type = UavStateType.Idle
        self.battery = battery
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.v = v
        self.node_idx_sched = 0  # node index in schedule
        self.node_idx_mission = 0  # node index for total mission

    def set_schedule(self, nodes: list):
        self.nodes = nodes
        self.start_node = nodes[0]
        self.node_idx_sched = 0

    def get_state(self, env):
        """
        Returns the position and batter charge of the UAV.
        """
        t_passed = env.now - self.t_start
        if self.state_type == UavStateType.Moving:
            battery = self.battery - t_passed * self.r_deplete
            start_pos = self.nodes[self.node_idx_sched].pos
            dir_vector = self.nodes[self.node_idx_sched].direction(self.nodes[self.node_idx_sched + 1])
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
                pos=self.nodes[self.node_idx_sched].pos,
                battery=battery,
            )
        elif self.state_type in [UavStateType.Waiting, UavStateType.Idle]:
            res = UavState(
                state_type=self.state_type,
                pos=self.nodes[self.node_idx_sched].pos,
                battery=self.battery,
            )
        return res

    # TODO: add event for when the UAV crashes
    def sim(self, env, callbacks=[], finish_callbacks=[]):
        while True:
            if self.node_idx_sched >= len(self.nodes) - 1:
                break
            node_cur = self.nodes[self.node_idx_sched]
            node_next = self.nodes[self.node_idx_sched + 1]
            distance = node_cur.dist(node_next)
            t_move = distance / self.v

            # move to node
            self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} is moving from {node_cur} to {node_next} [expected duration = {t_move:.1f}]")
            self.state_type = UavStateType.Moving
            self.t_start = env.now
            event = env.timeout(t_move, value=Event(env.now, self, "reached", node_next))

            def cb(event):
                self.node_idx_sched += 1
                if event.value.node.node_type == NodeType.Waypoint:
                    self.node_idx_mission += 1
                self.battery -= t_move * self.r_deplete
                self.state_type = UavStateType.Idle

            event.callbacks.append(cb)
            for cb in callbacks:
                event.callbacks.append(cb)

            yield event

            # wait at node
            waiting_time = node_next.wt
            if waiting_time > 0:
                self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} is waiting at {node_next}")

                self.state_type = UavStateType.Waiting
                self.t_start = env.now
                event = env.timeout(waiting_time, value=Event(env.now, self, "waited", node_next))

                def cb(_):
                    self.state_type = UavStateType.Idle

                event.callbacks.append(cb)
                for cb in callbacks:
                    event.callbacks.append(cb)

                yield event

            # charge at node
            charging_time = node_next.ct
            if charging_time > 0:
                self.logger.debug(f"[{env.now:.1f}] UAV {self.uav_id} is charging at {node_next}")

                self.state_type = UavStateType.Charging
                self.t_start = env.now
                resource = self.charging_stations[node_next.identifier]
                req = resource.request()
                yield req

                event = env.timeout(charging_time,
                                    value=Event(env.now, self, "charged", node_next))

                def cb(_):
                    self.battery = min(self.battery + charging_time * self.r_charge, 1)
                    self.state_type = UavStateType.Idle
                    resource.release(req)

                event.callbacks.append(cb)
                for cb in callbacks:
                    event.callbacks.append(cb)
                yield event
        for cb in finish_callbacks:
            cb()


class TimeStepper:
    def __init__(self, interval):
        self.timestep = 0
        self.interval = interval

    def _inc(self, _):
        self.timestep += 1

    def sim(self, env, callbacks=[]):
        while True:
            event = env.timeout(self.interval)
            for cb in callbacks:
                event.callbacks.append(cb)
            event.callbacks.append(self._inc)
            try:
                yield event
            except simpy.exceptions.Interrupt:
                break


class NotSolvableException(Exception):
    pass


class ScenarioFactory:
    """
    Creates new scenarios on the fly based on
    """

    def __init__(self, scenario: Scenario, W: int):
        self.original_start_pos = [wps[0] for wps in scenario.positions_w]
        self.positions_S = scenario.positions_S
        self.positions_w = [wps[1:] for wps in scenario.positions_w]
        self.offsets = [0] * scenario.N_d
        self.sc_orig = scenario
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w
        self.W = W

    def incr(self, d):
        """
        Increments the
        :param d: identifier of the UAV
        """
        self.offsets[d] += 1

    def next(self, start_positions):
        """
        Returns the next scenario
        """
        positions_w = []
        for i, wps in enumerate(self.positions_w):
            wps_truncated = [start_positions[i]] + wps[self.offsets[i]:self.offsets[i] + self.W - 1]
            while len(wps_truncated) < self.W:
                wps_truncated = [wps_truncated[0]] + wps_truncated
            positions_w.append(wps_truncated)
        return Scenario(positions_S=self.positions_S, positions_w=positions_w)


class Scheduler:
    def __init__(self, params: Parameters, scenario: Scenario):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.scenario = scenario

    @timed
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

        res = []
        for d in model.d:
            start_pos = self.scenario.positions_w[d][0]
            nodes = [
                AuxWaypoint(*self.scenario.positions_w[d][0])
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
        self.sf = ScenarioFactory(sc, W)

        self.timestepper = TimeStepper(delta)

        # prepare waypoint indices
        self.current_waypoint_idx = []
        self.reset_waypoint_indices()

        # used in callbacks
        self.remaining = []

        # maintain history of simulation
        self.events = []
        for _ in range(self.sf.N_d):
            self.events.append([])

    def reset_waypoint_indices(self):
        self.current_waypoint_idx = []
        for d in range(self.sf.N_d):
            self.current_waypoint_idx.append(0)

    def sim(self):
        env = simpy.Environment()

        # prepare waypoint indices
        self.reset_waypoint_indices()

        # prepare shared resources
        charging_stations = []
        for s in range(self.sf.N_s):
            charging_stations.append(simpy.Resource(env, capacity=1))

        # prepare UAVs
        uavs = []
        for d in range(self.sf.N_d):
            uav = UAV(d, charging_stations, self.params.v[d], self.params.r_charge[d], self.params.r_deplete[d])
            uavs.append(uav)

        self.logger.info(f"visiting {self.sf.N_w} waypoints per UAV in total")

        # convert scenario
        sc = self.sf.next(self.sf.original_start_pos)

        # get initial schedule
        scheduler = self.scheduler_cls(self.params, sc)
        t_solve, schedules = scheduler.schedule()
        self.logger.debug(f"[{env.now:.1f}] scheduled in {t_solve:.1f}s")
        for i, nodes in enumerate(schedules):
            n_wp_aux = len([x for x in nodes if x.node_type == NodeType.AuxWaypoint])
            n_stations = len([x for x in nodes if x.node_type == NodeType.ChargingStation])
            # self.logger.debug(
            #     f"UAV {i} is scheduled {n_wp_aux} auxiliary waypoints and {n_stations} charging stations, starting with {uavs[i].battery * 100:.1f}% battery charge")

        for i, nodes in enumerate(schedules):
            uavs[i].set_schedule(nodes)

        _, ax = plt.subplots()
        fname = f"out/simulation/scenarios/scenario_{self.timestepper.timestep:03}.pdf"
        self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname)
        self.timestepper._inc(_)

        def uav_cb(event):
            if event.value.name == "reached":
                if event.value.node.node_type == NodeType.Waypoint:
                    self.sf.incr(event.value.uav.uav_id)
                    # TODO: add end statement (UAV must keep state itself)
                    reached_name = f'new waypoint ({event.value.uav.node_idx_mission})'
                elif event.value.node.node_type == NodeType.AuxWaypoint:
                    reached_name = 'aux waypoint'
                elif event.value.node.node_type == NodeType.ChargingStation:
                    reached_name = f"charging station ({event.value.node.identifier})"
                self.logger.debug(f"[{env.now:.1f}] UAV {event.value.uav.uav_id} reached a {reached_name}")
            elif event.value.name == 'waited':
                self.logger.debug(
                    f"[{env.now:.1f}] UAV {event.value.uav.uav_id} finished waiting at station {event.value.node.identifier} for {event.value.node.wt:.1f}s")
            elif event.value.name == 'charged':
                self.logger.debug(
                    f"[{env.now:.1f}] UAV {event.value.uav.uav_id} finished charging at station {event.value.node.identifier} for {event.value.node.ct:.1f}s")
            self.events[event.value.uav.uav_id].append(event.value)

        def ts_cb(_):
            start_positions = []
            batteries = []
            for i, uav in enumerate(uavs):
                state = uav.get_state(env)
                start_positions.append(state.pos)
                batteries.append(state.battery)
            sc = self.sf.next(start_positions)

            params = self.params.copy()
            params.B_start = np.array(batteries)

            scheduler = Scheduler(params, sc)
            t_solve, schedules = scheduler.schedule()
            self.logger.debug(f"[{env.now}] scheduled in {t_solve:.1f}s")
            for i, nodes in enumerate(schedules):
                n_wp_aux = len([x for x in nodes if x.node_type == NodeType.AuxWaypoint])
                n_stations = len([x for x in nodes if x.node_type == NodeType.ChargingStation])
                # self.logger.debug(
                #     f"UAV {i} is scheduled {n_wp_aux} auxiliary waypoints and {n_stations} charging stations, starting with {uavs[i].battery * 100:.1f}% battery charge")

            for i, nodes in enumerate(schedules):
                uavs[i].set_schedule(nodes)

            _, ax = plt.subplots()
            fname = f"out/simulation/scenarios/scenario_{self.timestepper.timestep:03}.pdf"
            self.plot(schedules, [uav.get_state(env).battery for uav in uavs], ax=ax, fname=fname)

        ts = env.process(self.timestepper.sim(env, callbacks=[ts_cb]))

        self.remaining = self.sf.N_d

        def uav_finished_cb():
            self.logger.debug("uav finished")
            self.remaining -= 1
            if self.remaining == 0:
                ts.interrupt()

        # run simulation
        for uav in uavs:
            env.process(uav.sim(env, callbacks=[uav_cb], finish_callbacks=[uav_finished_cb]))
        env.run(until=ts)

        return env

    def plot(self, schedules, batteries, ax=None, fname=None):
        if not ax:
            _, ax = plt.subplots()

        colors = ['red', 'blue']
        for i, nodes in enumerate(schedules):
            x_all = [n.pos[0] for n in nodes]
            y_all = [n.pos[1] for n in nodes]
            x_wp = [n.pos[0] for n in nodes if n.node_type == NodeType.Waypoint]
            y_wp = [n.pos[1] for n in nodes if n.node_type == NodeType.Waypoint]
            x_c = [n.pos[0] for n in nodes if n.node_type == NodeType.ChargingStation]
            y_c = [n.pos[1] for n in nodes if n.node_type == NodeType.ChargingStation]
            x_aux = [n.pos[0] for n in nodes if n.node_type == NodeType.AuxWaypoint]
            y_aux = [n.pos[1] for n in nodes if n.node_type == NodeType.AuxWaypoint]
            alphas = np.linspace(1, 0.2, len(x_all) - 1)
            for j in range(len(x_all) - 1):
                x = x_all[j:j + 2]
                y = y_all[j:j + 2]
                label = f"{batteries[i] * 100:.1f}%" if j == 0 else None
                ax.plot(x, y, color=colors[i], label=label, alpha=alphas[j])
            ax.scatter(x_wp, y_wp, c=colors[i])
            ax.scatter(x_c, y_c, marker='s', c=colors[i], facecolor='white')
            ax.scatter(x_aux, y_aux, marker='o', c=colors[i], facecolor='white', zorder=10)
        for i, positions in enumerate(self.sf.sc_orig.positions_w):
            x = [x for x, _, _ in positions]
            y = [y for _, y, _ in positions]
            ax.scatter(x, y, marker='x', s=10, c=colors[i], zorder=-1, alpha=0.2)
        ax.legend()

        if fname:
            plt.savefig(fname, bbox_inches='tight')
