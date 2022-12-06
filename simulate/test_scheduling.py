import logging
from unittest import TestCase

import numpy as np
from pyomo.opt import SolverFactory

from simulate.parameters import SchedulingParameters
from simulate.scheduling import MilpScheduler, ScenarioFactory
from util.scenario import Scenario


class TestMilpScheduler(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

        self.params = SchedulingParameters.from_raw(
            v=[1],
            r_charge=[1],
            r_deplete=[0.15],
            B_start=[1],
            B_min=[0],
            B_max=[1],
            W_hat=10,
            sigma=5,
            pi=np.inf,
        )

    def test_scheduling(self):
        start_positions = [
            [0, 0, 0]
        ]
        positions_S = [
            [5.5, 1, 0]
        ]
        positions_w = [
            [
                [1, 0, 0],
                [2, 0, 0],
                [3, 0, 0],
                [4, 0, 0],
                [5, 0, 0],
                [6, 0, 0],
                [7, 0, 0],
                [8, 0, 0],
                [9, 0, 0],
                [10, 0, 0],
            ]
        ]
        sc = Scenario(start_positions, positions_S=positions_S, positions_w=positions_w)

        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        scheduler = MilpScheduler(self.params, sc, solver)

        cs_locks = np.array(
            [
                [0],
            ]
        )

        start_positions_dict = {i: v for i, v in enumerate(start_positions)}
        batteries = {0: 1}
        t_solve, (optimal, schedules) = scheduler.schedule(start_positions_dict, batteries=batteries, cs_locks=cs_locks, uavs_to_schedule=[0])
        logging.getLogger("test").info(schedules[0])
