import os

from simulate.parameters import Parameters
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Simulator
from simulate.strategy import AfterNEventsStrategyAll, OnEventStrategySingle
from util.scenario import Scenario


def main():
    sc = Scenario.from_file("scenarios/three_drones_circling.yml")

    p = dict(
        v=[1, 1, 1],
        r_charge=[0.15, 0.15, 0.15],
        r_deplete=[0.3, 0.3, 0.3],
        B_min=[0.1, 0.1, 0.1],
        B_max=[1, 1, 1],
        B_start=[1, 1, 1],
        # plot_delta=0.1,
        plot_delta=0,
        W=4,
        sigma=1,
        epsilon=0.1,
    )
    params = Parameters(**p)

    # MILP
    directory = 'out/simple_comparison/naive'
    os.makedirs(directory, exist_ok=True)
    strat = OnEventStrategySingle()
    scheduler = NaiveScheduler(params, sc)
    simulator = Simulator(scheduler, strat, params, sc, directory=directory)
    _, _, _ = simulator.sim()

    # MILP
    directory = 'out/simple_comparison/milp'
    os.makedirs(directory, exist_ok=True)
    strat = AfterNEventsStrategyAll(4)
    scheduler = MilpScheduler(params, sc)
    simulator = Simulator(scheduler, strat, params, sc, directory=directory)
    _, _, _ = simulator.sim()

if __name__ == "__main__":
    main()