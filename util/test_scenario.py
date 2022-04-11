from unittest import TestCase

from util.scenario import Scenario


class TestScenario(TestCase):
    def test_init(self):
        s = Scenario("../scenarios/single_drone.yml")
        self.assertEqual(len(s.positions_S), 2)
        self.assertEqual(len(s.positions_w), 1)
        self.assertEqual(len(s.positions_w[0]), 5)
