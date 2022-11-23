import json
import logging
import os
from unittest import TestCase

import jsons
from matplotlib import pyplot as plt
from pyomo.opt import SolverFactory

from simulate.environment import NormalDistributedEnvironment
from simulate.parameters import SchedulingParameters, SimulationParameters
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Simulator
from simulate.strategy import OnEventStrategySingle, AfterNEventsStrategyAll
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

        p_sim = dict(
            plot_delta=0,
        )
        p_sched = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.28, 0.28],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            W_hat=10,
            sigma=2,
            epsilon=1,
            pi=9,
        )

        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = 'out/test/milp_simulator_long'
        os.makedirs(directory, exist_ok=True)

        if directory:
            _, ax = plt.subplots()
            sc.plot(ax=ax, draw_distances=False)
            plt.savefig(os.path.join(directory, "scenario.pdf"), bbox_inches='tight')

        strat = AfterNEventsStrategyAll(sched_params.pi)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        scheduler = MilpScheduler(sched_params, sc, solver=solver)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_milp_three_drones_circling_W4(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p_sim = dict(
            plot_delta=0,
        )
        p_sched = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            W_hat=4,
            sigma=1,
            epsilon=5,
        )

        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = 'out/test/milp_three_drones_circling_W4'
        os.makedirs(directory, exist_ok=True)
        strat = AfterNEventsStrategyAll(3)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['outlev'] = 1
        solver.options['iisfind'] = 1
        solver.options['DualReductions'] = 0
        # solver = SolverFactory("gurobi")
        # solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(sched_params, sc, solver=solver)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_milp_three_drones_circling_W5(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p_sim = dict(
            plot_delta=0,
        )
        p_sched = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            W_hat=5,
            sigma=1,
            epsilon=5,
        )

        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = 'out/test/milp_three_drones_circling_W5'
        os.makedirs(directory, exist_ok=True)
        # strat = OnEventStrategyAll(interval=3)
        strat = AfterNEventsStrategyAll(5)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(sched_params, sc, solver=solver)
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(jsons.dump(result), f)

    def test_naive_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p_sim = dict(
            plot_delta=0,
        )
        p_sched = dict(
            v=[1, 1],
            r_charge=[0.04, 0.04],
            r_deplete=[0.28, 0.28],
            B_min=[0.1, 0.1],
            B_max=[1, 1],
            B_start=[1, 1],
            W_hat=5,
            sigma=2,
            epsilon=1,
        )

        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(sched_params, sc)
        scale = 0.2
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                dumped = jsons.dump(result)
                json.dump(dumped, f)
                # json.dump(result, f)

    def test_naive_three_drones_circling(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p_sim = dict(
            plot_delta=0,
        )
        p_sched = dict(
            v=[1, 1, 1],
            r_charge=[0.2, 0.2, 0.2],
            r_deplete=[0.3, 0.3, 0.3],
            B_min=[0.1, 0.1, 0.1],
            B_max=[1, 1, 1],
            B_start=[1, 1, 1],
            W_hat=5,
            sigma=1,
            epsilon=5,
        )

        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = 'out/test/naive_three_drones_circling'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(sched_params, sc)
        scale = 1
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory, simenvs=simenvs)
        result = simulator.sim()

        if directory:
            with open(os.path.join(directory, "result.json"), 'w') as f:
                json.dump(result, f)
