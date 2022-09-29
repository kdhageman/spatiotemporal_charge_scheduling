import os
import tempfile
from unittest import TestCase

import yaml

from simulate.parameters import Parameters
from simulate.scheduling import draw_graph
from simulate.util import as_graph
from util.scenario import Scenario


class TestScenario(TestCase):
    def setUp(self) -> None:
        p = dict(
            v=[1] * 3,
            r_charge=[0.04] * 3,
            r_deplete=[0.3] * 3,
            B_min=[0.1] * 3,
            B_max=[1] * 3,
            B_start=[1] * 3,
            plot_delta=0.1,
            # plot_delta=0,
            W=3,
            sigma=1,
            epsilon=1e-2,
            W_zero_min=None
        )
        self.params = Parameters(**p)

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
        self.assertEqual(len(s.positions_w[0]), 5)

    def test_init_padding(self):
        positions_S = [
            (1, 1),
        ]
        positions_w = [
            [
                (0, 0),
                (1, 0),
            ],
            [
                (0, 0),
                (2.5, 0),
                (4.5, 0),
                (6, 0),
                (7.5, 0),
            ]
        ]

        s = Scenario(positions_S, positions_w)
        self.assertEqual(s.N_s, 1)
        self.assertEqual(s.N_d, 2)
        self.assertEqual(s.N_w, 4)
        for wps in s.positions_w:
            self.assertEqual(len(wps), s.N_w + 1)

    def test_receding_horizon(self):
        wps = [
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
            ], [
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
            ]
        ]
        stations = [(3, 0.5)]
        sc = Scenario(positions_S=stations, positions_w=wps)

        starting_positions = [(0.5, 0), (0.5, 1)]
        progress = [1, 1]
        N_w = 3
        sc_rhc = sc.receding_horizon(starting_positions, progress, N_w)
        self.assertEqual(sc_rhc.N_w, N_w)
        self.assertEqual(sc_rhc.positions_S, sc.positions_S)
        for d in range(sc.N_d):
            self.assertEqual(sc_rhc.positions_w[d][0][0:2], starting_positions[d][0:2])

    def test_collapse(self):
        wps = [
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
        charging_stations = [(3, 0.5)]

        start_positions = [(0, 0)]

        sc = Scenario(start_positions=start_positions, positions_S=charging_stations, positions_w=wps)

        anchors = [
            [1, 3, 5]
        ]
        offsets = [0]
        fname = os.path.join(os.getcwd(), "not_collapsed.pdf")
        draw_graph(sc, self.params, anchors, offsets, fname)

        collapsed = sc.collapse(anchors)
        fname = os.path.join(os.getcwd(), "collapsed.pdf")
        anchors = [list(range(collapsed.N_w))]
        draw_graph(collapsed, self.params, anchors, offsets, fname)

