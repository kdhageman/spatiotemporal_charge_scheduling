import logging

from matplotlib import pyplot as plt

from simulate.simulate import Parameters, Simulator, Scheduler, NotSolvableException
from util.scenario import Scenario


def main():
    logger = logging.getLogger(__name__)
    # sc = Scenario.from_file("scenarios/two_longer_path.yml")
    sc = Scenario.from_file("scenarios/two_drones.yml")

    p = dict(

        v=[1, 1],
        r_charge=[0.15, 0.15],
        r_deplete=[0.3, 0.3],
        B_min=[0.1, 0.1],
        B_max=[1, 1],
        B_start=[1, 1],
    )
    schedule_delta = 35
    W = 7

    params = Parameters(**p)

    simulator = Simulator(Scheduler, params, sc, schedule_delta, W)
    try:
        env = simulator.sim()
    except NotSolvableException as e:
        logger.fatal(f"failure during simulation: {e}")
    _, axes = plt.subplots(nrows=sc.N_d)
    simulator.plot(axes)
    plt.savefig("out/simulation/test.pdf", bbox_inches='tight')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("pyomo").setLevel(logging.FATAL)
    logging.getLogger("matplotlib").setLevel(logging.FATAL)

    main()
