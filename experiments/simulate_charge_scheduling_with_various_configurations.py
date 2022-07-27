import logging
import os
import yaml
from experiments.util_funcs import schedule_charge_from_conf


class Configuration:
    def __init__(self, baseconf: dict, basedir, charging_strategy, n_drones, W, sigma, schedule_delta, flight_sequence_fpath):
        self.baseconf = baseconf
        self.charging_strategy = charging_strategy
        self.n_drones = n_drones
        self.W = W
        self.sigma = sigma
        self.basedir = basedir
        self.schedule_delta = schedule_delta
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
        conf['charging_optimization']['schedule_delta'] = self.schedule_delta
        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, n_drones, flight_sequence_fpath):
        super().__init__(baseconf, basedir, "naive", n_drones, 0, 0, 0, flight_sequence_fpath)

    def outputdir(self):
        return os.path.join(self.basedir, f"naive_{self.n_drones}")


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, n_drones, W, sigma, schedule_delta, flight_sequence_fpath):
        super().__init__(baseconf, basedir, "milp", n_drones, W, sigma, schedule_delta, flight_sequence_fpath)

    def outputdir(self):
        return os.path.join(self.basedir, f"{self.charging_strategy}_ndrones{self.n_drones}_sigma{self.sigma}_W{self.W}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"
    flight_sequence_fpath4 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"
    flight_sequence_fpath5 = "out/flight_sequences/villalvernia_5/flight_sequences.pkl"
    flight_sequence_fpath6 = "out/flight_sequences/villalvernia_6/flight_sequences.pkl"

    confs = [
        NaiveConfiguration(baseconf, "out/villalvernia",3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=3, sigma=4, schedule_delta=0.5, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=4, sigma=6, schedule_delta=4, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=4, sigma=10, schedule_delta=40, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=3, schedule_delta=21, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=6, schedule_delta=42, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=10, schedule_delta=70, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=3, schedule_delta=30, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=6, schedule_delta=60, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=10, schedule_delta=100, flight_sequence_fpath=flight_sequence_fpath3),
        NaiveConfiguration(baseconf, "out/villalvernia",4, flight_sequence_fpath3),
        # MilpConfiguration(baseconf, "out/villalvernia", 4, W=3, sigma=4, schedule_delta=0.5, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=4, sigma=6, schedule_delta=4, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=4, sigma=10, schedule_delta=40, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=7, sigma=3, schedule_delta=21, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=7, sigma=6, schedule_delta=42, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=7, sigma=10, schedule_delta=70, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=10, sigma=3, schedule_delta=30, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=10, sigma=6, schedule_delta=60, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, W=10, sigma=10, schedule_delta=100, flight_sequence_fpath=flight_sequence_fpath4),
        NaiveConfiguration(baseconf, "out/villalvernia",5, flight_sequence_fpath5),
        # MilpConfiguration(baseconf, "out/villalvernia", 5, W=3, sigma=4, schedule_delta=0.5, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=4, sigma=6, schedule_delta=4, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=4, sigma=10, schedule_delta=40, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=7, sigma=3, schedule_delta=21, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=7, sigma=6, schedule_delta=42, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=7, sigma=10, schedule_delta=70, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=10, sigma=3, schedule_delta=30, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=10, sigma=6, schedule_delta=60, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, W=10, sigma=10, schedule_delta=100, flight_sequence_fpath=flight_sequence_fpath5),
        NaiveConfiguration(baseconf, "out/villalvernia",6, flight_sequence_fpath6),
        # MilpConfiguration(baseconf, "out/villalvernia", 5, W=3, sigma=4, schedule_delta=0.5, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=4, sigma=6, schedule_delta=4, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=4, sigma=10, schedule_delta=40, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=7, sigma=3, schedule_delta=21, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=7, sigma=6, schedule_delta=42, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=7, sigma=10, schedule_delta=70, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=10, sigma=3, schedule_delta=30, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=10, sigma=6, schedule_delta=60, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, W=10, sigma=10, schedule_delta=100, flight_sequence_fpath=flight_sequence_fpath6),
    ]

    for conf in confs:
        try:
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.error("failed to run configuration")
