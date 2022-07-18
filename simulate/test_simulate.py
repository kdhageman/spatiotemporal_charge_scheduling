import logging
import os
from unittest import TestCase

from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Parameters, \
    Simulator
from simulate.strategy import IntervalStrategy, ArrivalStrategy
from util.scenario import Scenario


class TestSimulator(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_milp_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.3, 0.3],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            # plot_delta=0.1,
            plot_delta=0,
            W=8,
            sigma=2,
            epsilon=1e-6,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = IntervalStrategy(5)
        simulator = Simulator(MilpScheduler, strat, params, sc, directory=directory)
        _, env, events = simulator.sim()
        print(env.now)

    def test_naive_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.3, 0.3],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            # plot_delta=2,
            plot_delta=0,
            W=8,
            sigma=2,
        )
        params = Parameters(**p)

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = ArrivalStrategy()
        simulator = Simulator(NaiveScheduler, strat, params, sc, directory=directory)
        _, env, events = simulator.sim()
        print(env.now)
