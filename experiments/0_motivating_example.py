import logging
import math
import os

from matplotlib import pyplot as plt

from simulate.parameters import SchedulingParameters, SimulationParameters
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Simulator
from simulate.strategy import AfterNEventsStrategyAll, OnWaypointStrategySingle
from util.scenario import Scenario


def milp_simulator(rootdir, sc, sched_params, sim_params):
    directory = os.path.join(rootdir, "milp")
    os.makedirs(directory, exist_ok=True)

    scheduler = MilpScheduler(sched_params, sc)
    strat = AfterNEventsStrategyAll(10)

    return Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory)


def greedy_simulator(rootdir, sc, sched_params, sim_params):
    directory = os.path.join(rootdir, "greedy")
    os.makedirs(directory, exist_ok=True)

    scheduler = NaiveScheduler(sched_params, sc)
    strat = OnWaypointStrategySingle()

    return Simulator(scheduler, strat, sched_params, sim_params, sc, directory=directory)


def main():
    rootdir = "out/motivating_example"
    os.makedirs(rootdir, exist_ok=True)

    start_positions = [
        (0, 2),
        (0, 0),
    ]

    positions_S = [
        (1.5, 1),
        (2.5, 1),
        (3.5, 1),
    ]

    positions_w = [
        [
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
        ],  # drone 1
        [
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
        ],  # drone 2
    ]

    sched_params = SchedulingParameters.from_raw(
        v=[1, 1],
        r_charge=[.5, .5],
        r_deplete=[0.3, 0.3],
        B_start=[1, 1],
        B_min=[0.1, 0.1],
        B_max=[1, 1],
        omega=[[0] * 3] * 2,
        rho=[0, 0],
        W_hat=10,
    )
    sim_params = SimulationParameters(
        delta_t =400,
        draw_scheduling=False,
        draw_battery_profile_greyscale=False,
    )

    sc = Scenario(start_positions, positions_S, positions_w)
    _, ax = plt.subplots(figsize=(5, 3))
    ax.axis('equal')
    ax.axis('off')
    sc.plot(ax=ax, draw_distances=False)
    plt.savefig(os.path.join(rootdir, "scenario.pdf"), bbox_inches='tight')

    milp_simulator(rootdir, sc, sched_params, sim_params).sim()
    greedy_simulator(rootdir, sc, sched_params, sim_params).sim()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("pyomo").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("gurobi").setLevel(logging.ERROR)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage[utf8]{inputenc} \usepackage{lmodern}",
        }
    )

    main()
