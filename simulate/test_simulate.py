import logging
import os
from unittest import TestCase

from pyomo.opt import SolverFactory

from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Parameters, \
    Simulator
from simulate.strategy import IntervalStrategy, OnEventStrategySingle
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
        strat = IntervalStrategy(5)
        scheduler = MilpScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        _, env, events = simulator.sim()
        print(env.now)

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
        strat = IntervalStrategy(3)
        solver = SolverFactory("gurobi")
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        solve_times, env, events = simulator.sim()
        print(env.now)

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
            epsilon=1
        )
        params = Parameters(**p)

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        _, env, events = simulator.sim()
        print(env.now)

# class TestPlot(TestCase):
#     station1 = ChargingStation(0, 0, 0, identifier=0)
#     station2 = ChargingStation(0, 0, 0, identifier=1)
#     events = [
#         [
#             ChargedEvent(1, 2, station1),
#             ChargedEvent(5, 1, station1),
#         ],
#         [
#             ChargedEvent(2, 5, station1),
#             ChargedEvent(8, 1, station1),
#         ]
#     ]
#
#     directory = 'out/test/plot'
#     os.makedirs(directory, exist_ok=True)
#     plot_station_occupancy(events, 2, 10, os.path.join(directory, "plot_station_occupancy.pdf"))
