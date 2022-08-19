import logging
import os
import pickle
from datetime import datetime
from unittest import TestCase

import yaml
from pyomo.opt import SolverFactory

from experiments.util_funcs import ChargingStrategy
from simulate.event import EventType, StartedEvent, ReachedEvent, ChargedEvent
from simulate.node import NodeType, Node, ChargingStation, AuxWaypoint, Waypoint
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Parameters, Simulator, plot_events_battery
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
        strat = AfterNEventsStrategyAll(params.sigma * (params.W - 1))
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
            # epsilon=1.9,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_three_drones_circling_W4'
        os.makedirs(directory, exist_ok=True)
        # strat = OnEventStrategyAll(interval=3)
        strat = AfterNEventsStrategyAll(5)
        solver = SolverFactory("gurobi_ampl", solver_io='nl')
        solver.options['outlev'] = 1
        solver.options['iisfind'] = 1
        solver.options['DualReductions'] = 0
        # solver = SolverFactory("gurobi")
        # solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc, solver=solver)
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
            epsilon=2,
        )
        params = Parameters(**p)

        directory = 'out/test/milp_three_drones_circling_W5'
        os.makedirs(directory, exist_ok=True)
        # strat = OnEventStrategyAll(interval=3)
        strat = AfterNEventsStrategyAll(params.sigma * (params.W - 1))
        solver = SolverFactory("gurobi")
        solver.options['MIPFocus'] = 1
        scheduler = MilpScheduler(params, sc, solver=solver)
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

    def test_naive_three_drones_circling(self):
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
            epsilon=2,
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

    def test_villalvernia_3_milp_sigma6_w4(self):
        fpath_conf = "config/charge_scheduling/villalvernia_3_milp_sigma6_w4.yml"
        with open(fpath_conf, 'r') as f:
            conf = yaml.load(f, Loader=yaml.Loader)

        co = conf["charging_optimization"]
        n_drones = conf['n_drones']
        B_min = [co["B_min"]] * n_drones
        B_max = [co["B_max"]] * n_drones
        B_start = [co["B_start"]] * n_drones
        v = [co["v"]] * n_drones
        r_charge = [co["r_charge"]] * n_drones
        r_deplete = [co["r_deplete"]] * n_drones
        epsilon = co.get("epsilon", None)
        schedule_delta = co.get('schedule_delta', None)
        plot_delta = co['plot_delta']
        W = co.get('W', None)
        sigma = co.get('sigma', None)
        charging_station_positions = co['charging_positions']
        directory = conf['output_directory']

        flight_sequence_fpath = conf['flight_sequence_fpath']
        with open(flight_sequence_fpath, 'rb') as f:
            seqs = pickle.load(f)

        params = Parameters(
            v=v,
            r_charge=r_charge,
            r_deplete=r_deplete,
            B_start=B_start,
            B_min=B_min,
            B_max=B_max,
            epsilon=epsilon,
            plot_delta=plot_delta,
            schedule_delta=schedule_delta,
            W=W,
            sigma=sigma,
        )
        strategy = ChargingStrategy.parse(conf['charging_strategy'])

        if directory:
            os.makedirs(directory, exist_ok=True)

        sc = Scenario(charging_station_positions, [seq.tolist() for seq in seqs])
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] # drones:               {sc.N_d}")
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] # stations:             {sc.N_s}")
        for d in range(sc.N_d):
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] # waypoints for UAV[{d}]: {len(seqs[d])}")
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] W:                      {params.W}")
        self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] sigma:                  {params.sigma}")
        if strategy == ChargingStrategy.Milp:
            strat = AfterNEventsStrategyAll(params.sigma * (params.W - 1))
            solver = SolverFactory("gurobi")
            solver.options['IntFeasTol'] = 1e-9
            solver.options['TimeLimit'] = 30
            scheduler = MilpScheduler(params, sc, solver=solver)
            simulator = Simulator(scheduler, strat, params, sc, directory=None)
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] prepared MILP simulator")
        elif strategy == ChargingStrategy.Naive:
            strat = OnEventStrategySingle()
            scheduler = NaiveScheduler(params, sc)
            simulator = Simulator(scheduler, strat, params, sc, directory=None)
            self.logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] prepared naive simulator")

        visited_positions = {}
        for d, seq in enumerate(seqs):
            visited_positions_for_d = {}
            for pos in seq[1:]:  # skip first waypoint because it is the starting point
                visited_positions_for_d[tuple(pos)] = 0
            visited_positions[d] = visited_positions_for_d

        def cb(ev):
            if ev.value.type == EventType.reached and ev.value.node.node_type == NodeType.Waypoint:
                uav_id = ev.value.uav.uav_id
                pos = tuple(ev.value.node.pos)
                if pos in visited_positions[uav_id]:
                    visited_positions[uav_id][pos] += 1
                else:
                    self.logger.warning(f"UAV [{uav_id}] arrived at unscheduled node")

        for uav in simulator.uavs:
            uav.add_arrival_cb(cb)

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

        return env, events


class TestPlotBattery(TestCase):
    def test_plot_events_battery(self):
        r_charge = 0.1

        events = [
            [
                StartedEvent(0, 0, AuxWaypoint(0, 0, 0), None, battery=1, depletion=0, forced=False),
                ReachedEvent(0, 1, Waypoint(0, 0, 0), None, battery=0.9, depletion=0.1, forced=False),
                ReachedEvent(1, 3, ChargingStation(0, 0, 0, wt=0, ct=0, identifier=0), None, battery=0.6, depletion=0.3, forced=False),
                ChargedEvent(4, 2, ChargingStation(0, 0, 0, wt=0, ct=0, identifier=0), None, battery=0.8, depletion=-0.2, forced=False),
                ReachedEvent(6, 7, Waypoint(1, 0, 0), None, battery=0.1, depletion=0.7, forced=False)
            ],
            [
                StartedEvent(0, 0, AuxWaypoint(0, 0, 0), None, battery=1, depletion=0, forced=False),
                ReachedEvent(0, 1, Waypoint(0, 0, 0), None, battery=0.9, depletion=0.1, forced=False),
                ReachedEvent(1, 3, ChargingStation(0, 0, 0, wt=0, ct=0, identifier=0), None, battery=0.6, depletion=0.3, forced=False),
                ChargedEvent(4, 2, ChargingStation(0, 0, 0, wt=0, ct=0, identifier=0), None, battery=0.8, depletion=-0.2, forced=False),
                ReachedEvent(6, 7, Waypoint(1, 0, 0), None, battery=0.1, depletion=0.7, forced=False)
            ]
        ]

        plot_events_battery(events, fname="test.pdf", r_charge=r_charge)
