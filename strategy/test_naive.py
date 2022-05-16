from unittest import TestCase

import numpy as np

from strategy.naive import NaiveStrategy
from util.scenario import Scenario


class TestNaiveStrategy(TestCase):
    def test_schedule(self):
        # sc = Scenario.from_file("../scenarios/single_longer_path.yml")
        sc = Scenario.from_file("../scenarios/three_drones_circling.yml")
        params = dict(
            B_start=[1, 1, 1],
            B_min=0,
            B_max=1,
            v=[1, 1.2, 1],
            r_charge=np.array([0.15, 0.15, 0.15]),
            r_deplete=np.array([0.3, 0.3, 0.3]),
        )
        strat = NaiveStrategy(sc, params)
        outcome = strat.simulate()
