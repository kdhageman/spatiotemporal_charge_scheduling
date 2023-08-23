import os
import tempfile
from unittest import TestCase

import numpy as np
import yaml

from simulate.parameters import SchedulingParameters
from simulate.util import draw_graph
from util.scenario import Scenario, ScenarioFactory


class TestScenario(TestCase):
    def test_init_single(self):
        doc = {
            "charging_stations": [
                {
                    "x": 3,
                    "y": 1
                },
                {
                    "x": 5,
                    "y": -1
                },
            ],
            'drones': [
                {"waypoints": [
                    {
                        "x": 0,
                        "y": 0
                    },
                    {
                        "x": 2.5,
                        "y": 0
                    },
                    {
                        "x": 4.5,
                        "y": 0
                    },
                    {
                        "x": 6,
                        "y": 0
                    },
                    {
                        "x": 7.5,
                        "y": 0
                    },
                ]}
            ],
        }
        with tempfile.NamedTemporaryFile(mode='w') as f:
            yaml.dump(doc, f)
            s = Scenario.from_file(f.name)

        # s = Scenario(doc)
        self.assertEqual(len(s.positions_S), 2)
        self.assertEqual(len(s.positions_w), 1)
        self.assertEqual(len(s.positions_w[0]), 4)
        self.assertEqual(len(s.start_positions), 1)

    def test_init_padding(self):
        positions_S = [
            (1, 1),
        ]
        positions_w = [
            [
                (1, 0),
            ],
            [
                (2.5, 0),
                (4.5, 0),
                (6, 0),
                (7.5, 0),
            ]
        ]
        start_positions = [(0, 0), (0, 0)]

        s = Scenario(start_positions, positions_S, positions_w)
        self.assertEqual(s.N_s, 1)
        self.assertEqual(s.N_d, 2)
        self.assertEqual(s.N_w, 4)
        for wps in s.positions_w:
            self.assertEqual(len(wps), s.N_w)


class TestCollapse(TestCase):
    def setUp(self):
        self.sched_params = SchedulingParameters.from_raw(
            v=[1] * 3,
            r_charge=[0.04] * 3,
            r_deplete=[0.3] * 3,
            B_start=[1] * 3,
            B_min=0.1 * np.ones((3, 3)),
            B_max=[1] * 3,
            omega=[[0] * 1] * 3,
            rho=[0] * 3,
            W_hat=3,
            sigma=1,
            epsilon=1e-2,
            pi=1,
        )

        self.wps = [
            [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
            ]
        ]
        self.charging_stations = [(3, 0.5)]
        self.start_positions = [(0, 0)]

    def test_collapse_normal_draw(self):
        anchors = [
            [1, 3, 5]
        ]
        sc = Scenario(start_positions=self.start_positions, positions_S=self.charging_stations, positions_w=self.wps, anchors=anchors)

        fname = os.path.join(os.getcwd(), "not_collapsed.pdf")
        draw_graph(sc, self.sched_params, fname)

        collapsed = sc.collapse()
        fname = os.path.join(os.getcwd(), "collapsed.pdf")
        draw_graph(collapsed, self.sched_params, fname)

    def test_collapse_first_anchor_draw(self):
        anchors = [
            [0, 1, 3, 5]
        ]
        sc = Scenario(start_positions=self.start_positions, positions_S=self.charging_stations, positions_w=self.wps, anchors=anchors)

        offsets = [0]
        fname = os.path.join(os.getcwd(), "not_collapsed.pdf")
        draw_graph(sc, self.sched_params, fname)

        collapsed = sc.collapse()
        fname = os.path.join(os.getcwd(), "collapsed.pdf")
        draw_graph(collapsed, self.sched_params, fname)

    def test_check_d_n(self):
        anchors = [[2, 5]]
        sc = Scenario(start_positions=self.start_positions, positions_S=self.charging_stations, positions_w=self.wps, anchors=anchors)
        offsets = [0]
        draw_graph(sc, self.sched_params, "not_collapsed.pdf")
        collapsed = sc.collapse()
        draw_graph(collapsed, self.sched_params, "collapsed.pdf")

        # w_s = 0 (D_N)
        expected = sc.D_N[0, :, 2] + 2
        actual = collapsed.D_N[0, :, 0]
        self.assertTrue(np.array_equal(expected, actual))

        # w_s = 1 (D_N)
        expected = sc.D_N[0, :, 5] + 2
        actual = collapsed.D_N[0, :, 1]
        self.assertTrue(np.array_equal(expected, actual))

        # w_s = 0 (D_W)
        expected = sc.D_W[0, :, 2]
        actual = collapsed.D_W[0, :, 0]
        self.assertTrue(np.array_equal(expected, actual))

        # w_s = 1 (D_W)
        expected = sc.D_W[0, :, 5] + 1
        actual = collapsed.D_W[0, :, 1]
        self.assertTrue(np.array_equal(expected, actual))

    def test_collapse_uneven_nr_anchors(self):
        wps = [
            [
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
            ],
            [
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
            ]
        ]
        start_positions = [
            (0, 0),
            (0, 1)
        ]
        anchors = [
            [0, 3],
            [1],
        ]
        sc = Scenario(start_positions=start_positions, positions_S=self.charging_stations, positions_w=wps, anchors=anchors)
        fname = os.path.join(os.getcwd(), "not_collapsed.uneven.pdf")
        draw_graph(sc, self.sched_params, fname)

        collapsed = sc.collapse()
        fname = os.path.join(os.getcwd(), "collapsed.uneven.pdf")
        draw_graph(collapsed, self.sched_params, fname)

        positions_w_expected = [
            [(1, 0, 0), (4, 0, 0)],
            [(4, 1, 0), (4, 1, 0)]
        ]
        positions_w_actual = collapsed.positions_w
        self.assertEqual(positions_w_expected, positions_w_actual)


class TestScenarioFactory(TestCase):
    def test_next(self):
        positions_S = []
        positions_w = [
            [
                (0, 0),
                (1, 0),
                (3, 0),
                (6, 0),
                (10, 0),
                (15, 0),
                (21, 0),
                (28, 0),
                (36, 0),
                (45, 0),
            ]
        ]
        start_positions = [(-0.5, 0)]
        sc = Scenario(start_positions, positions_S, positions_w)
        sf = ScenarioFactory(sc, W_hat=3, sigma=2)

        offsets = [0]
        sc_new, _ = sf.next(start_positions, offsets)

        D_N_expected = np.reshape([0.5, 1, 2], (1, 1, 3))
        D_N_actual = sc_new.D_N
        self.assertTrue(np.array_equal(D_N_expected, D_N_actual))

        actual = sc_new.anchors[0]
        expected = [0, 2]
        self.assertEqual(expected, actual)

        offsets = [1]
        sc_new, _ = sf.next(start_positions, offsets)

        D_N_expected = np.reshape([1.5, 2, 3], (1, 1, 3))
        D_N_actual = sc_new.D_N
        self.assertTrue(np.array_equal(D_N_expected, D_N_actual))

        actual = sc_new.anchors[0]
        expected = [1]
        self.assertEqual(expected, actual)

    def test_anchors(self):
        positions_S = []
        positions_w = [
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
                (8, 0),
                (9, 0),
            ]
        ]
        start_positions = [(-1, 0)]
        sc = Scenario(start_positions, positions_S, positions_w)
        sf = ScenarioFactory(sc, W_hat=3, sigma=3)
        actual = sf.anchors()
        expected = [0, 3, 6, 9]
        self.assertEqual(expected, actual)

        sf = ScenarioFactory(sc, W_hat=10, sigma=2)
        actual = sf.anchors()
        expected = [0, 2, 4, 6, 8]
        self.assertEqual(expected, actual)
