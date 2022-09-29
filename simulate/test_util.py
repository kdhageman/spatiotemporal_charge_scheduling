import logging
from unittest import TestCase

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from simulate.parameters import Parameters
from simulate.util import maximum_schedule_delta, is_feasible, as_graph
from util.scenario import Scenario


class TestMaximumScheduleDelta(TestCase):
    def test_maximum_schedule_delta(self):
        positions_w = [
            [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (4, 0, 0),
                (4.5, 0, 0),
            ]
        ]
        sc = Scenario([], positions_w)

        v = np.array([1])
        actual = maximum_schedule_delta(sc, v, W=3, sigma=1)
        expected = 1.5
        self.assertEqual(actual, expected)

        v = np.array([0.5])
        actual = maximum_schedule_delta(sc, v, W=3, sigma=1)
        expected = 3
        self.assertEqual(actual, expected)

        v = np.array([1])
        actual = maximum_schedule_delta(sc, v, W=2, sigma=1)
        expected = 0.5
        self.assertEqual(actual, expected)

        v = np.array([1])
        actual = maximum_schedule_delta(sc, v, W=3, sigma=2)
        expected = 3.5
        self.assertEqual(actual, expected)

        v = np.array([1])
        actual = maximum_schedule_delta(sc, v, W=10, sigma=10)
        expected = 4.5
        self.assertEqual(actual, expected)


class TestFeasibility(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_is_feasible(self):
        start_positions = [
            (0, 0)
        ]
        positions_S = [(0, 1)]
        positions_w = [
            [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
            ]
        ]

        sc = Scenario(start_positions, positions_S, positions_w)

        params = Parameters(
            v=[1],
            r_charge=[0.02],
            r_deplete=[0.05],
            B_start=[1],
            B_min=[0.5],
            B_max=[1],
            W_zero_min=np.zeros((1, 1)),
            W=5,
        )
        anchors = [
            [0, 2, 4]
        ]

        is_feasible(sc, params, anchors)

    def test_no_anchors(self):
        start_positions = [
            (0, 0)
        ]
        positions_S = [(0, 1)]
        positions_w = [
            [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
            ]
        ]

        sc = Scenario(start_positions, positions_S, positions_w)
        params = Parameters(
            v=[1],
            r_charge=[0.02],
            r_deplete=[0.05],
            B_start=[1],
            B_min=[0.5],
            B_max=[1],
            W_zero_min=np.zeros((1, 1)),
            W=5,
        )
        anchors = [
            []
        ]

        is_feasible(sc, params, anchors)

    def test_just_feasible(self):
        start_positions = [
            (0, 0)
        ]
        positions_S = [(5, 0)]  # should be doable when charging halfway
        positions_w = [
            [
                (10, 0),
            ]
        ]

        sc = Scenario(start_positions, positions_S, positions_w)
        params = Parameters(
            v=[1],
            r_charge=[0.02],
            r_deplete=[0.20],
            B_start=[1],
            B_min=[0],
            B_max=[1],
            W_zero_min=np.zeros((1, 1)),
            W=6,
        )
        anchors = [
            [0]
        ]

        expected = True
        actual = is_feasible(sc, params, anchors)
        self.assertEqual(expected, actual)

    def test_not_feasible(self):
        start_positions = [
            (0, 0)
        ]
        positions_S = [(5, 0)]  # not possible anymore with the depletion rate too high
        positions_w = [
            [
                (10, 0),
            ]
        ]

        sc = Scenario(start_positions, positions_S, positions_w)
        params = Parameters(
            v=[1],
            r_charge=[0.02],
            r_deplete=[0.21],
            B_start=[1],
            B_min=[0],
            B_max=[1],
            W_zero_min=np.zeros((1, 1)),
            W=6,
        )
        anchors = [
            [0]
        ]

        expected = False
        actual = is_feasible(sc, params, anchors)
        self.assertEqual(expected, actual)


class TestGraph(TestCase):
    def test_graph(self):
        positions_w = [
            [
                (1, 0),
                (2, 0),
                (3, 0),
            ]
        ]
        positions_S = [
            (1, 0.5),
            (2, 0.5),
        ]
        start_positions = [
            (0, 0)
        ]
        sc = Scenario(positions_S=positions_S, positions_w=positions_w, start_positions=start_positions)

        anchors = [
            [1, 2]
        ]

        g, pos = as_graph(sc, anchors, 0, [0])

        plt.subplots()
        nx.draw(g, pos)
        nx.draw_networkx_labels(g, pos, font_size=6, font_color='white')
        edge_labels = {(n1, n2): f"{dat['dist']:.1f}" for n1, n2, dat in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)
        plt.savefig("graph.pdf", bbox_inches='tight')
