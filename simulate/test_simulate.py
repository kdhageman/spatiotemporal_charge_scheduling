import logging
from unittest import TestCase

import numpy as np
import simpy
from simpy import Resource

from simulate.simulate import Parameters, Scheduler, Simulator, UAV, Waypoint, ChargingStation, TimeStepper, Node
from util.scenario import Scenario


class TestSimulator(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)

    def test_simulator(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")
        # sc = Scenario.from_file("scenarios/two_drones.yml")

        p = dict(

            v=[1, 1],
            r_charge=[0.15, 0.15],
            r_deplete=[0.3, 0.3],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
        )
        # delta = 10
        delta = 1
        W = 10

        params = Parameters(**p)

        simulator = Simulator(Scheduler, params, sc, delta, W)
        env = simulator.sim()
        print(env.now)

    def test_prepare_scenario(self):
        original_sc = Scenario.from_file("scenarios/two_longer_path.yml")

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

        simulator = Simulator(Scheduler, params, original_sc, delta, W)
        sc = simulator.prepare_scenario(first=True)
        self.assertTrue(len(sc.positions_w), 2)
        for wps in sc.positions_w:
            self.assertEqual(len(wps), W)

        positions = [(0, 0, 0), (0, 0, 0)]
        sc = simulator.prepare_scenario(positions=positions)
        self.assertTrue(len(sc.positions_w), 2)
        for wps in sc.positions_w:
            self.assertEqual(len(wps), W)

        # test for padding (# of waypoints = 20)
        simulator.current_waypoint_idx[0] = 19
        simulator.current_waypoint_idx[1] = 18
        sc = simulator.prepare_scenario(positions=positions)
        self.assertTrue(len(sc.positions_w), 2)
        for wps in sc.positions_w:
            self.assertEqual(len(wps), W)


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

        scheduler = Scheduler(params=params, scenario=sc)
        _, schedules = scheduler.schedule()
        self.assertTrue(len(schedules), 2)


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

        uav = UAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
        uav.set_schedule(nodes)
        uav_proc = env.process(uav.sim(env, callbacks=[uav_cb]))

        def ts_cb(event):
            uav_state = uav.get_state(env)
            print(f"{event.env.now}: {uav_state.state_type} {uav_state.battery} {uav_state.pos}")

        ts = TimeStepper(interval=0.5)
        env.process(ts.sim(env, callbacks=[ts_cb]))

        env.run(until=uav_proc)

        self.assertEqual(env.now, 12)

    def test_get_state(self):
        """
        Moving a UAV in the following pattern, with the 'o' representing the time stepper evaluation
        2 --o------ 3
        |           |
        |           o
        |           |
        |           |
        1 --o------ 4
        """
        env = simpy.Environment()
        charging_stations = [
            Resource(env)
        ]
        uav = UAV(0, charging_stations, 1, 0.1, 0.1, 1)
        nodes = [
            Waypoint(0, 0, 0),
            Waypoint(0, 2, 0),
            ChargingStation(0, 5, 0, identifier=0, wt=0, ct=0),
            Waypoint(1, 5, 0),
            Waypoint(5, 5, 0),
            ChargingStation(5, 3, 0, identifier=0, wt=0, ct=0),
            ChargingStation(5, 0, 0, identifier=0, wt=0, ct=0),
            Waypoint(4, 0, 0),
            Waypoint(1, 0, 0),
            Waypoint(0, 0, 0),
        ]
        uav.set_schedule(nodes)
        proc = env.process(uav.sim(env))

        self.counter = 0
        expected_arrs = [
            np.array([1, 5, 0]),
            np.array([5, 3, 0]),
            np.array([2, 0, 0]),
        ]

        timestepper = TimeStepper(interval=6)

        def ts_cb(_):
            state = uav.get_state(env)
            actual = state.pos
            expected = expected_arrs[self.counter]
            self.assertTrue(np.array_equal(actual, expected))
            self.counter += 1

        env.process(timestepper.sim(env, callbacks=[ts_cb]))

        env.run(until=proc)


class TestNode(TestCase):
    def test_direction(self):
        node1 = Node(0, 0, 0, 0, 0)
        node2 = Node(0, 0, 3, 0, 0)
        dir_vector = node1.direction(node2)
        expected = np.array([0, 0, 1])
        self.assertTrue(np.array_equal(dir_vector, expected))
