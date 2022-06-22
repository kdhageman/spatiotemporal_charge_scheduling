import logging
import os
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from simulate.node import Node, NodeType, Waypoint
from simulate.simulate import Parameters, Scheduler, Simulator, ScenarioFactory
from util.scenario import Scenario


class TestSimulator(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    # def test_plot(self):
    #     sc = Scenario.from_file("scenarios/two_longer_path.yml")
    #     _, ax = plt.subplots()
    #     sc.plot(ax=ax, draw_distances=False)
    #     ax.axis('off')
    #     plt.savefig("out/simulation/scenario/scenario.pdf", bbox_inches='tight')

    def test_simulator_long(self):
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
        # schedule_delta = 1
        # schedule_delta = 10
        schedule_delta = 5
        plot_delta = 0.05
        W = 10

        params = Parameters(**p)

        directory = 'out/test/long'
        os.makedirs(directory, exist_ok=True)
        simulator = Simulator(Scheduler, params, sc, schedule_delta, W, plot_delta=plot_delta, directory=directory)
        env, events = simulator.sim()
        print(env.now)

    def test_simulator_short_no_charging(self):
        positions_w = [
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
            ]
        ]
        positions_S = [
            (2, 0.25, 0)
        ]
        sc = Scenario(positions_S, positions_w)

        params = Parameters(
            v=[1],
            r_charge=[0.1],
            r_deplete=[0.2],
            B_start=[1],
            B_min=[0.1],
            B_max=[1],
        )

        schedule_delta = 1.2
        W = 3

        directory = 'out/test/short_no_charging'
        os.makedirs(directory, exist_ok=True)
        simulator = Simulator(Scheduler, params, sc, schedule_delta, W, directory=directory, plot_delta=0.05)
        env, event_list = simulator.sim()
        self.assertEqual(len(event_list), 1)
        self.assertEqual(len(event_list[0]), 5)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "started"]), 1)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "reached"]), 4)
        self.assertEqual(env.now, 4)

    def test_simulator_short_charging(self):
        positions_w = [
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
            ]
        ]
        positions_S = [
            (2, 0.25, 0)
        ]
        sc = Scenario(positions_S, positions_w)

        params = Parameters(
            v=[1],
            r_charge=[0.1],
            r_deplete=[0.3],
            B_start=[1],
            B_min=[0.1],
            B_max=[1],
        )

        schedule_delta = 1.2
        W = 4

        directory = 'out/test/short_charging'
        os.makedirs(directory, exist_ok=True)
        simulator = Simulator(Scheduler, params, sc, schedule_delta, W, directory=directory, plot_delta=0.05)
        env, event_list = simulator.sim()
        self.assertEqual(len(event_list), 1)
        self.assertEqual(len(event_list[0]), 7)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "started"]), 1)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "reached"]), 5)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "charged"]), 1)

    def test_simulator_change_midmove(self):
        positions_w = [
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
            ]
        ]
        positions_S = [
            (2, 0.25, 0)
        ]
        sc = Scenario(positions_S, positions_w)

        params = Parameters(
            v=[1],
            r_charge=[0.1],
            r_deplete=[0.45],
            B_start=[1],
            B_min=[0],
            B_max=[1],
        )

        schedule_delta = 1.75
        W = 3

        directory = 'out/test/change_midmove'
        os.makedirs(directory, exist_ok=True)
        simulator = Simulator(Scheduler, params, sc, schedule_delta, W, directory=directory, plot_delta=0.05)
        env, event_list = simulator.sim()

        self.assertEqual(len(event_list), 1)
        self.assertEqual(len(event_list[0]), 7)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "reached"]), 4)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "started"]), 1)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "changed_course"]), 1)
        self.assertEqual(len([e for e in event_list[0] if e.value.name == "charged"]), 1)

    def test_plot(self):
        positions_S = [
            [0, 0, 0]
        ]
        positions_w = [
            [
                [-1, 0, 0],
                [-1, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [1, -1, 0],
                [0, -1, 0],
                [-1, -1, 0],
            ]
        ]
        sc = Scenario(positions_S, positions_w)

        sim = Simulator(None, None, sc, 0, 5)

        schedules = [
            [(-1, 0, 0), [Waypoint(-1, 1, 0), Waypoint(-1, 1, 0), Waypoint(0, 1, 0), Waypoint(1, 1, 0)]]
        ]
        batteries = [0.8]
        fname = "out/test/test_sim.pdf"
        title = "0.00s"

        _, ax = plt.subplots()
        sim.plot(schedules, batteries, ax=ax, fname=fname, title=title)


class TestScheduler(TestCase):
    def test_scheduler(self):
        sc = Scenario(
            positions_S=[
                (1.5, 1, 0),
                (3.5, 1, 0),
            ],
            positions_w=[
                [
                    (0, 2, 0),
                    (1, 2, 0),
                    (2, 2, 0),
                    (3, 2, 0),
                    (4, 2, 0),
                    (5, 2, 0),
                ],
                [
                    (0, 0, 0),
                    (1, 0, 0),
                    (2, 0, 0),
                    (3, 0, 0),
                    (4, 0, 0),
                    (5, 0, 0),
                ]
            ])
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
        self.assertEqual(schedules[0][0], (0, 2, 0))
        self.assertEqual(schedules[1][0], (0, 0, 0))
        self.assertEqual(len([x for x in schedules[0][1] if x.node_type == NodeType.Waypoint]), 5)
        self.assertEqual(len([x for x in schedules[1][1] if x.node_type == NodeType.Waypoint]), 5)


class TestNode(TestCase):
    def test_direction(self):
        node1 = Node(0, 0, 0, 0, 0)
        node2 = Node(0, 0, 3, 0, 0)
        dir_vector = node1.direction(node2)
        expected = np.array([0, 0, 1])
        self.assertTrue(np.array_equal(dir_vector, expected))


class TestScenarioFactory(TestCase):
    def test_scenario_factory(self):
        positions_S = [(0.5, 0.5, 0)]
        positions_w = [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            [(0, 0, 0), (-1, 0, 0), (-2, 0, 0), (-3, 0, 0), (-4, 0, 0)],
        ]
        sc_orig = Scenario(positions_S, positions_w)
        sf = ScenarioFactory(sc_orig, W=3)

        # first iteration
        start_positions = [(0, 0, 0), (0, 0, 0)]
        actual = sf.next(start_positions).positions_w
        expected = [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            [(0, 0, 0), (-1, 0, 0), (-2, 0, 0)],
        ]
        self.assertEqual(actual, expected)

        # after increment
        sf.incr(0)
        start_positions = [(1.5, 0, 0), (-0.5, 0, 0)]
        actual = sf.next(start_positions).positions_w
        expected = [
            [(1.5, 0, 0), (2, 0, 0), (3, 0, 0)],
            [(-0.5, 0, 0), (-1, 0, 0), (-2, 0, 0)],
        ]
        self.assertEqual(actual, expected)

        # padding iterations
        for _ in range(2):
            sf.incr(0)
        for _ in range(3):
            sf.incr(1)

        start_positions = [(3.5, 0, 0), (-3.5, 0, 0)]
        actual = sf.next(start_positions).positions_w
        expected = [
            [(3.5, 0, 0), (3.5, 0, 0), (4, 0, 0)],
            [(-3.5, 0, 0), (-3.5, 0, 0), (-4, 0, 0)],
        ]
        self.assertEqual(actual, expected)

        sf.incr(0)
        sf.incr(1)
        actual = sf.next(start_positions).positions_w
        expected = [
            [(3.5, 0, 0), (3.5, 0, 0), (3.5, 0, 0)],
            [(-3.5, 0, 0), (-3.5, 0, 0), (-3.5, 0, 0)],
        ]
        self.assertEqual(actual, expected)

    def test_remaining_waypoints(self):
        positions_S = [(0.5, 0.5, 0)]
        positions_w = [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            [(0, 0, 0), (-1, 0, 0), (-2, 0, 0), (-3, 0, 0), (-4, 0, 0)],
        ]
        sc_orig = Scenario(positions_S, positions_w)
        sf = ScenarioFactory(sc_orig, W=3)

        expected = [
            [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)],
            [(-1, 0, 0), (-2, 0, 0), (-3, 0, 0), (-4, 0, 0)]
        ]
        for d in range(2):
            actual = sf.remaining_waypoints(d)
            self.assertEqual(expected[d], actual)
        sf.incr(0)
        sf.incr(0)
        sf.incr(1)
        sf.incr(1)
        sf.incr(1)
        sf.incr(1)

        expected = [
            [(3, 0, 0), (4, 0, 0)],
            [],
        ]
        for d in range(2):
            actual = sf.remaining_waypoints(d)
            self.assertEqual(expected[d], actual)
