from unittest import TestCase

import numpy as np
import simpy

from simulate.simulate import Parameters, Scheduler, Simulator, UAV, Waypoint, ChargingStation, TimeStepper, Node
from util.scenario import Scenario


class TestSimulator(TestCase):

    def test_simulator(self):
        # sc = Scenario.from_file("scenarios/two_longer_path.yml")
        sc = Scenario.from_file("scenarios/two_drones.yml")

        p = dict(

            v=[1, 1],
            r_charge=[0.15, 0.15],
            r_deplete=[0.3, 0.3],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
        )
        delta = 3
        W = 4

        params = Parameters(**p)

        simulator = Simulator(Scheduler, params, sc, delta, W)
        simulator.sim()


class TestScheduler(TestCase):
    def test_scheduler(self):
        sc = Scenario.from_file("scenarios/two_drones.yml")
        p = dict(
            v=[1, 1],
            r_charge=[0.15, 0.15],
            r_deplete=[0.3, 0.3],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
        )
        params = Parameters(**p)
        W = 4

        scheduler = Scheduler(params=params, scenario=sc, W=W)
        scheduler.schedule()
        # TODO: add assertions


class TestUAV(TestCase):
    def test_uav(self):
        env = simpy.Environment()

        nodes = np.array([
            Waypoint(0, 0, 0),
            ChargingStation(1, 0, 0, 0, ct=1, wt=3),
            Waypoint(2, 0, 0),
            ChargingStation(3, 0, 0, 1, ct=2, wt=2),
            Waypoint(4, 0, 0),
        ])
        v = 1

        charging_stations = []
        for i in range(2):
            resource = simpy.Resource(env, capacity=1)
            charging_stations.append(resource)

        def uav_cb(event):
            print(f"{event.env.now} {event.value.name}")

        uav = UAV(charging_stations, v, r_charge=0.1, r_deplete=0.1)
        uav.set_schedule(nodes)
        uav_proc = env.process(uav.sim(env, callbacks=[uav_cb]))

        def ts_cb(event):
            uav_state = uav.get_state(env)
            print(f"{event.env.now}: {uav_state.state_type} {uav_state.battery} {uav_state.pos}")

        ts = TimeStepper(interval=0.5)
        env.process(ts.sim(env, callbacks=[ts_cb]))

        env.run(until=uav_proc)

        self.assertEqual(env.now, 12)


class TestNode(TestCase):
    def test_direction(self):
        node1 = Node(0, 0, 0, 0, 0)
        node2 = Node(0, 0, 3, 0, 0)
        dir_vector = node1.direction(node2)
        expected = np.array([0, 0, 1])
        self.assertTrue(np.array_equal(dir_vector, expected))
