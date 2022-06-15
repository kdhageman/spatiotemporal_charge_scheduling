import logging
from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt

from simulate.node import Node, NodeType
from simulate.simulate import Parameters, Scheduler, Simulator, ScenarioFactory
from util.scenario import Scenario


class TestSimulator(TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_plot(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")
        _, ax = plt.subplots()
        sc.plot(ax=ax, draw_distances=False)
        plt.savefig("out/simulation/scenarios/scenario.pdf", bbox_inches='tight')

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
        delta = 0.5
        W = 10

        params = Parameters(**p)

        simulator = Simulator(Scheduler, params, sc, delta, W)
        env = simulator.sim()
        print(env.now)


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
