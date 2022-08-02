import logging
import os
import yaml
from experiments.util_funcs import schedule_charge_from_conf


class Configuration:
    def __init__(self, baseconf: dict, basedir, charging_strategy, n_drones, W, sigma, flight_sequence_fpath):
        self.baseconf = baseconf
        self.charging_strategy = charging_strategy
        self.n_drones = n_drones
        self.W = W
        self.sigma = sigma
        self.basedir = basedir
        self.flight_sequence_fpath = flight_sequence_fpath

    def outputdir(self):
        raise NotImplementedError

    def as_dict(self):
        """
        Return the enhanced base configuration with the configuration properties
        :return:
        """
        conf = self.baseconf
        conf['flight_sequence_fpath'] = self.flight_sequence_fpath
        conf['output_directory'] = self.outputdir()
        conf['n_drones'] = self.n_drones
        conf['charging_strategy'] = self.charging_strategy
        conf['charging_optimization']['W'] = self.W
        conf['charging_optimization']['sigma'] = self.sigma
        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, n_drones, flight_sequence_fpath):
        super().__init__(baseconf, basedir, "naive", n_drones, 0, 0, flight_sequence_fpath)

    def outputdir(self):
        return os.path.join(self.basedir, f"naive_{self.n_drones}")


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, n_drones, W, sigma, flight_sequence_fpath):
        super().__init__(baseconf, basedir, "milp", n_drones, W, sigma, flight_sequence_fpath)

    def outputdir(self):
        return os.path.join(self.basedir, f"{self.charging_strategy}_ndrones{self.n_drones}_sigma{self.sigma}_W{self.W}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"  # ~160 waypoints per UAV

    basedir = "out/villalvernia/planning_horizon/increase_sigma"
    # increase sigmas
    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=1, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=2, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=4, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=5, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=7, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=8, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=11, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=12, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=13, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=14, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=15, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=17, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=18, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=19, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=20, W=10, flight_sequence_fpath=flight_sequence_fpath3),
    ]

    basedir = "out/villalvernia/planning_horizon/full_coverage"
    # full coverage, but increase W
    confs += [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=34, W=6, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=29, W=7, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=25, W=8, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=22, W=9, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=19, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=17, W=11, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=16, W=12, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=15, W=13, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=14, W=14, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=13, W=15, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=12, W=16, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=11, W=17, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=10, W=18, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=19, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=20, flight_sequence_fpath=flight_sequence_fpath3),
    ]

    for conf in confs:
        try:
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.error(f"failed to run configuration: {e}")
