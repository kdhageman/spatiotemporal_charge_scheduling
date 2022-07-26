from unittest import TestCase

import numpy as np

from simulate.util import maximum_schedule_delta
from util.scenario import Scenario


class Test(TestCase):
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
