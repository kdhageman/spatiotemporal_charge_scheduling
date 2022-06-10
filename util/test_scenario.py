import tempfile
from unittest import TestCase

import yaml

from util.scenario import Scenario


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
        self.assertEqual(len(s.positions_w[0]), 5)

    def test_init_padding(self):
        doc = {
            'drones': [
                {
                    "waypoints": [
                        {
                            "x": 0,
                            "y": 0,
                        }
                    ]
                },
                {
                    "waypoints": [
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
                    ]
                },
            ],
        }

        s = Scenario(doc)
        self.assertEqual(s.N_s, 0)
        self.assertEqual(s.N_d, 2)
        self.assertEqual(s.N_w, 5)
        for wps in s.positions_w:
            self.assertEqual(len(wps), s.N_w)