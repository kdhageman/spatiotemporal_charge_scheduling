import logging
from unittest import TestCase

from pyomo.opt import SolverFactory

from simulate.parameters import Parameters
from simulate.scheduling import MilpScheduler
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
            W=4,
            sigma=1,
            epsilon=1e-2,
        )
        params = Parameters(**p)

        solver = SolverFactory('gurobi')
        sched = MilpScheduler(params, sc, solver=solver)

        start_positions = {i: x[0] for i, x in enumerate(sc.positions_w)}
        batteries = {d: 1 for d in range(sc.N_d)}
        _, _, schedules = sched.schedule(start_positions, batteries, uavs_to_schedule=list(range(sc.N_d)))
        return


