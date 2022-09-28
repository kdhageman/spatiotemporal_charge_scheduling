import logging
from unittest import TestCase

import numpy as np
from pyomo.opt import SolverFactory

from simulate.parameters import Parameters
from simulate.scheduling import MilpScheduler, ScenarioFactory
from util.scenario import Scenario


class TestMilpScheduler(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_three_drones_circling(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")
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
        params = Parameters(**p)

        solver = SolverFactory('gurobi')
        sched = MilpScheduler(params, sc, solver=solver)

        start_positions = {i: x[0] for i, x in enumerate(sc.positions_w)}
        batteries = {d: 1 for d in range(sc.N_d)}
        cs_locks = np.zeros((sc.N_d, sc.N_s))
        _, (_, schedules) = sched.schedule(start_positions, batteries, cs_locks, uavs_to_schedule=list(range(sc.N_d)))


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
        sf = ScenarioFactory(sc, W=3, sigma=3)

        offsets = [0]
        sc_new, _ = sf.next(start_positions, offsets)

        D_N_expected = np.reshape([0.5, 1, 2], (1, 1, 3))
        D_N_actual = sc_new.D_N
        self.assertTrue(np.array_equal(D_N_expected, D_N_actual))

        offsets = [2]
        sc_new, _ = sf.next(start_positions, offsets)

        D_N_expected = np.reshape([3.5, 3, 4], (1, 1, 3))
        D_N_actual = sc_new.D_N
        self.assertTrue(np.array_equal(D_N_expected, D_N_actual))

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
        sf = ScenarioFactory(sc, W=3, sigma=3)
        actual = sf.anchors()
        expected = [0, 3, 6, 9]
        self.assertEqual(expected, actual)

        sf = ScenarioFactory(sc, W=10, sigma=2)
        actual = sf.anchors()
        expected = [0, 2, 4, 6, 8]
        self.assertEqual(expected, actual)
