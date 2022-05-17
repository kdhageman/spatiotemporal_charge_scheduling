import logging
from unittest import TestCase

import numpy as np

from pyomo_models.base import BaseModel
from simulate.simulate import Simulation, Schedule
from strategy.naive import NaiveStrategy
from util.scenario import Scenario


class TestNaiveStrategy(TestCase):
    def test_schedule(self):
        logging.basicConfig(level=logging.INFO)

        # sc = Scenario.from_file("../scenarios/single_longer_path.yml")
        sc = Scenario.from_file("../scenarios/three_drones_circling.yml")
        parameters = dict(
            B_start=[1, 1, 1],
            B_min=0,
            B_max=1,
            v=[1, 1.2, 1],
            r_charge=np.array([0.15, 0.15, 0.15]),
            r_deplete=np.array([0.3, 0.3, 0.3]),
        )
        strat = NaiveStrategy(sc, parameters)
        decisions, waiting_times, charging_times = strat.simulate()
        for d in range(sc.N_d):
            schedule = Schedule(decisions[d], waiting_times[d], charging_times[d])

            model = BaseModel(sc, parameters)
            params = Simulation.from_base_model(model, 0).params

            sim = Simulation(schedule, params)
            sim.simulate()