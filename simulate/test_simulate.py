import json
import logging
import os
from itertools import product
from unittest import TestCase

import jsons
from matplotlib import pyplot as plt
from pyomo.opt import SolverFactory

from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Parameters, Simulator
from simulate.strategy import OnEventStrategySingle, AfterNEventsStrategyAll
from simulate.environment import NormalDistributedEnvironment
from util.scenario import Scenario


class TestSimulator(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def test_milp_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.28, 0.28],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            # plot_delta=3,
            plot_delta=0,
            # W=4,
            W=5,
            sigma=2,
            epsilon=1,
            W_zero_min=None,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_simulator_long'
        os.makedirs(directory, exist_ok=True)

        if directory:
            _, ax = plt.subplots()
            sc.plot(ax=ax, draw_distances=False)
            plt.savefig(os.path.join(directory, "scenario.pdf"), bbox_inches='tight')

        strat = AfterNEventsStrategyAll(1)
        # strat = AfterNEventsStrategyAll(sc.N_w + 1)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        scheduler = MilpScheduler(params, sc, solver=solver)
        scale = 0.03
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_milp_three_drones_circling_W4(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            # plot_delta=0.1,
            plot_delta=0,
            W=4,
            sigma=1,
            epsilon=5,
            W_zero_min=None,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_three_drones_circling_W4'
        os.makedirs(directory, exist_ok=True)
        strat = AfterNEventsStrategyAll(3)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['outlev'] = 1
        solver.options['iisfind'] = 1
        solver.options['DualReductions'] = 0
        # solver = SolverFactory("gurobi")
        # solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc, solver=solver)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_milp_three_drones_circling_W5(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            plot_delta=0.1,
            # plot_delta=0,
            W=5,
            sigma=1,
            epsilon=5,
            W_zero_min=None,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_three_drones_circling_W5'
        os.makedirs(directory, exist_ok=True)
        # strat = OnEventStrategyAll(interval=3)
        strat = AfterNEventsStrategyAll(params.sigma * (params.W - 1))
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc, solver=solver)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_naive_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.28, 0.28],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            plot_delta=0,
            W=sc.N_w,
            sigma=1,
            epsilon=1,
            W_zero_min=None,
            delta_t=2,
        )
        params = Parameters(**p)

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        scale = 0.2
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_naive_three_drones_circling(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            # plot_delta=0.1,
            plot_delta=0,
            W=5,
            sigma=1,
            epsilon=5,
            W_zero_min=None,
        )
        params = Parameters(**p)

        directory = 'out/test/naive_three_drones_circling'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        scale = 1
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_find_failing_case(self):
        return  # TODO: this test case can be used to try a variety of parameters to make the test fail
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.28, 0.28],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            plot_delta=0,
            W_zero_min=None,
        )
        sigmas = [1, 2, 3, 4, 5]
        Ws = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        pis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        # Ws = [3]
        # sigmas = [1]
        # pis = [1]

        for sigma, W, pi in product(sigmas, Ws, pis):
            if sigma >= W - 1:
                # impossible
                continue
            if pi >= W - 1:
                # impossible
                continue
            p['W'] = W
            p['sigma'] = sigma

            directory = 'out/test/test_find_failing_case'
            os.makedirs(directory, exist_ok=True)
            directory = None  # TODO: comment out to produce output
            params = Parameters(**p)
            strat = AfterNEventsStrategyAll(pi)
            scheduler = MilpScheduler(params, sc)
            simulator = Simulator(scheduler, strat, params, sc, directory=directory)
            result = simulator.sim()

            if directory:
                with open(os.path.join(directory, "result.json"), 'w') as f:
                    json.dump(jsons.dump(result), f)
