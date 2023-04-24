import json
import logging
import os
from unittest import TestCase

import numpy as np
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
            delta_t=5,
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
            pi=5,
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
        simulator.sim()

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
        simulator.sim()

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
        simulator.sim()

    def test_naive_simulator_long(self):
        sc = Scenario.from_file("scenarios/two_longer_path.yml")

        p_sim = dict(
            plot_delta=0,
            delta_t=1,
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

        directory = 'out/test/naive_simulator_long'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(sched_params, sc)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory, simenvs=simenvs)
        simulator.sim()

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
        simulator.sim()

    def test_generate_demo_video_milp(self):
        start_positions = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        positions_S = [
            [-5, -10, 0],
            [10, 5, 0],
        ]

        with open("out/flight_sequences/villalvernia_4.demo.fine/flight_sequences.json", 'r') as f:
            position_w = json.load(f)

        sc = Scenario(start_positions, positions_S, position_w)

        p_sim = dict(
            plot_delta=5,
        )
        p_sched = dict(
            r_charge=[1 / 240]*4,
            r_deplete=[1 / 120]*4,
            v=[1]*4,
            B_min=[0.1]*4,
            B_max=[1]*4,
            B_start=[1]*4,
            W_hat=100,
            sigma=10,
            epsilon=5,
        )
        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = "out/test/demo/milp"
        os.makedirs(directory, exist_ok=True)
        strat = AfterNEventsStrategyAll(75)
        # solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver = SolverFactory("gurobi")
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(sched_params, sc, solver=solver)
        # scheduler = NaiveScheduler(sched_params, sc)
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory)
        simulator.sim()

    def test_generate_demo_video_greedy(self):
        start_positions = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        positions_S = [
            [-5, -10, 0],
            [10, 5, 0],
        ]

        with open("out/flight_sequences/villalvernia_4.demo.fine/flight_sequences.json", 'r') as f:
            position_w = json.load(f)

        sc = Scenario(start_positions, positions_S, position_w)

        p_sim = dict(
            plot_delta=5,
        )
        p_sched = dict(
            r_charge=[1 / 240] * 4,
            r_deplete=[1 / 120] * 4,
            v=[1] * 4,
            B_min=[0.1] * 4,
            B_max=[1] * 4,
            B_start=[1] * 4,
            W_hat=100,
            sigma=10,
            epsilon=5,
        )
        sim_params = SimulationParameters(**p_sim)
        sched_params = SchedulingParameters.from_raw(**p_sched)

        directory = "out/test/demo/greedy"
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(sched_params, sc)
        simulator = Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory)
        simulator.sim()

class TestFailingCase(TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

        self.start_positions = [[-13.559854178252943, 0.5133199052103233, 2.377043178757954],
                                [-5.1825760641060015, 1.8900243448648735, 4.642737359572407],
                                [9.257831604149622, 11.839546599945958, 3.6446305736185862]]
        self.positions_S = [[-5, -15, 0], [10, 5, 0]]
        self.positions_w = [[[-19.24280667403044, 1.8735211464345176, 3.3644812505658717],
                             [-21.21667738363538, -3.568748832651963, 4.351919317234395],
                             [-23.480754998433873, 1.9802376098561227, 5.339357381782671],
                             [-24.563758408131527, 8.845918963074363, 6.326795444358716],
                             [-26.109955013964928, 2.269913313624652, 7.314233504622253],
                             [-20.991168181673636, -6.77875686465932, 7.687906135244146],
                             [-23.33179992479578, -5.483171584532944, 8.061578759409805],
                             [-28.224083429897426, 0.718418967156339, 8.435251373911708],
                             [-28.652370725551158, 4.614121687178963, 8.808923980183373],
                             [-28.70724222347531, 8.063672431313158, 9.182596585491503],
                             [-22.472050074473998, -8.68078906915031, 9.556269190553405],
                             [-22.26341977512514, -9.303482448244004, 9.929941795558543]],
                            [[-3.740306621983244, 4.679121201454113, 9.285474719044021],
                             [-2.2980371798797896, 7.468218058063673, 10.797843328652435],
                             [-3.1734170269798554, 10.62104425338005, 11.729121429580745],
                             [-6.250050744009471, 12.873071176509024, 12.66039952999343],
                             [-8.903457334292062, 6.920231601262658, 12.439904301430685],
                             [-10.659846434578995, 0.7276069316311177, 12.219409072536372],
                             [-17.54775187937615, 12.054722007638235, 11.998913843568584],
                             [-12.786493918788471, 16.352012705336683, 11.778418614545584],
                             [-7.9404347361131995, 20.543972613699918, 11.557923385477851],
                             [-1.5617835669920122, 20.80288226765961, 11.337428156370843],
                             [6.05699270267436, 14.698058165902697, 11.116932927228127],
                             [-3.219825573995506, 24.905979204022508, 10.896437698057802],
                             [-14.350322761396333, 17.853652291523566, 10.675942468875602]],
                            [[6.69243699739841, 23.83239945151522, 3.931028218345826],
                             [5.897108142102886, 26.46055918914454, 5.5498897536823835],
                             [13.397153990641442, 23.457524952460258, 7.168751293538472],
                             [11.952547292880933, 19.072245742633644, 8.787612833707655],
                             [7.753714274306875, 22.321174875558196, 10.40647437041311],
                             [4.853894116314534, 28.112513655151684, 11.237488013505327],
                             [11.680648963421552, 25.198928451064663, 12.06850162480142],
                             [18.50740379237156, 22.2853432247376, 12.299502767081616],
                             [15.371478018937946, 27.805733408592534, 11.758247889385872],
                             [12.235552213062142, 33.32612360699796, 11.216992783027825],
                             [12.38515003082737, 33.42816485399117, 10.675737668025862],
                             [12.779088126411525, 33.46828029205288, 10.134482546116834],
                             [12.247274283290995, 37.78086737192405, 9.593227422661386]]]
        self.sc = Scenario(self.start_positions, self.positions_S, self.positions_w)

        self.sched_params = SchedulingParameters.from_raw(
            v=[0.6, 0.6, 0.6],
            r_charge=[0.0005025, 0.0005025, 0.0005025],
            r_deplete=[0.0045, 0.0045, 0.0045],
            B_start=[1, 1, 1],
            B_min=[0.4, 0.4, 0.4],
            B_max=[1, 1, 1],
            W_hat=13,
            sigma=1,
            pi=np.inf,
            time_limit=600,
            int_feas_tol=1e-12,
            epsilon=10,
        )
        self.sim_params = SimulationParameters(
            plot_delta=0
        )

    def test_milp(self):
        directory = 'out/test/failing_case/milp'
        os.makedirs(directory, exist_ok=True)
        strat = AfterNEventsStrategyAll(self.sc.N_w + 1)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['IntFeasTol'] = 1e-09
        scheduler = MilpScheduler(self.sched_params, self.sc, solver)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, self.sched_params, self.sim_params, self.sc, directory=directory, simenvs=simenvs)
        simulator.sim()

    def test_naive(self):
        directory = 'out/test/failing_case/naive'
        os.makedirs(directory, exist_ok=True)
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(self.sched_params, self.sc)
        scale = 0
        simenvs = [
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1),
            NormalDistributedEnvironment.from_seed(scale, seed=1)
        ]
        simulator = Simulator(scheduler, strat, self.sched_params, self.sim_params, self.sc, directory=directory, simenvs=simenvs)
        simulator.sim()
