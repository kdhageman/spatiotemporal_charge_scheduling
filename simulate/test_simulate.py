import logging
import os
from unittest import TestCase

from pyomo.opt import SolverFactory

from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Parameters, Simulator
from simulate.strategy import OnEventStrategySingle, OnEventStrategyAll
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
            epsilon=1,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategyAll(interval=5)
        scheduler = MilpScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        solve_times, env, events = simulator.sim()

        # write solve times to disk
        if directory:
            with open(os.path.join(directory, 'solve_times.csv'), 'w') as f:
                f.write("iteration,sim_timestamp,optimal,solve_time,n_remaining_waypoints\n")
                for i, (sim_timestamp, optimal, solve_time, n_remaining_waypoints) in enumerate(solve_times):
                    f.write(f"{i},{sim_timestamp},{optimal},{solve_time},{n_remaining_waypoints}\n")

            # write mission execution time to disk
            with open(os.path.join(directory, "execution_time.txt"), 'w') as f:
                f.write(f"{env.now}")


    def test_milp_three_drones_circling(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p = dict(
            v=[1] * 3,
            r_charge=[0.04] * 3,
            r_deplete=[0.3] * 3,
            B_min=[0.1] * 3,
            B_max=[1] * 3,
            B_start=[1] * 3,
            # plot_delta=0.1,
            plot_delta=0,
            W=4,
            sigma=1,
            epsilon=1,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_three_drones_circling'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategyAll(interval=3)
        solver = SolverFactory("gurobi")
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        solve_times, env, events = simulator.sim()

        # write solve times to disk
        if directory:
            with open(os.path.join(directory, 'solve_times.csv'), 'w') as f:
                f.write("iteration,sim_timestamp,optimal,solve_time,n_remaining_waypoints\n")
                for i, (sim_timestamp, optimal, solve_time, n_remaining_waypoints) in enumerate(solve_times):
                    f.write(f"{i},{sim_timestamp},{optimal},{solve_time},{n_remaining_waypoints}\n")

            # write mission execution time to disk
            with open(os.path.join(directory, "execution_time.txt"), 'w') as f:
                f.write(f"{env.now}")

    def test_naive_three_drones_circling(self):
        sc = Scenario.from_file("scenarios/three_drones_circling.yml")

        p = dict(
            v=[1] * 3,
            r_charge=[0.04] * 3,
            r_deplete=[0.3] * 3,
            B_min=[0.1] * 3,
            B_max=[1] * 3,
            B_start=[1] * 3,
            # plot_delta=0.1,
            plot_delta=0,
            W=5,
            sigma=1,
            epsilon=1,
        )
        params = Parameters(**p)

        directory = 'out/test/naive_three_drones_circling'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        solve_times, env, events = simulator.sim()

        # write solve times to disk
        if directory:
            with open(os.path.join(directory, 'solve_times.csv'), 'w') as f:
                f.write("iteration,sim_timestamp,optimal,solve_time,n_remaining_waypoints\n")
                for i, (sim_timestamp, optimal, solve_time, n_remaining_waypoints) in enumerate(solve_times):
                    f.write(f"{i},{sim_timestamp},{optimal},{solve_time},{n_remaining_waypoints}\n")

            # write mission execution time to disk
            with open(os.path.join(directory, "execution_time.txt"), 'w') as f:
                f.write(f"{env.now}")


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
            epsilon=1
        )
        params = Parameters(**p)

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        solve_times, env, events = simulator.sim()

        # write solve times to disk
        if directory:
            with open(os.path.join(directory, 'solve_times.csv'), 'w') as f:
                f.write("iteration,sim_timestamp,optimal,solve_time,n_remaining_waypoints\n")
                for i, (sim_timestamp, optimal, solve_time, n_remaining_waypoints) in enumerate(solve_times):
                    f.write(f"{i},{sim_timestamp},{optimal},{solve_time},{n_remaining_waypoints}\n")

            # write mission execution time to disk
            with open(os.path.join(directory, "execution_time.txt"), 'w') as f:
                f.write(f"{env.now}")
