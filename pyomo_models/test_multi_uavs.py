import logging
from unittest import TestCase

import numpy as np
from pyomo.opt import SolverFactory

from pyomo_models.multi_uavs import MultiUavModel
from simulate.parameters import Parameters
from util.scenario import Scenario


class TestMultiUavModel(TestCase):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("pyomo").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("gurobi").setLevel(logging.ERROR)
    sc = Scenario.from_file("scenarios/two_longer_path.yml")

    params = Parameters(
        v=[1, 1],
        r_charge=[0.04, 0.04],
        r_deplete=[0.28, 0.28],
        B_min=[0.1, 0.1],
        B_max=[1, 1],
        B_start=[1, 1],
        plot_delta=0,
        W=8,
        sigma=2,
        epsilon=1,
        W_zero_min=np.zeros((sc.N_d, sc.N_s))
    )

    model = MultiUavModel(sc, params)
    solver = SolverFactory("gurobi")
    solver.solve(model)

    print(model.P_np)
    print(model.C_np)
    print(model.W_np)
