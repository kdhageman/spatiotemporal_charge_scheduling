import logging
import os

import yaml

from experiments.inspection import run_from_conf


class Configuration:
    def __init__(self, baseconf: dict, basedir, charging_strategy, n_drones, W, sigma, schedule_delta):
        self.baseconf = baseconf
        self.charging_strategy = charging_strategy
        self.n_drones = n_drones
        self.W = W
        self.sigma = sigma
        self.basedir = basedir
        self.schedule_delta = schedule_delta

    def outputdir(self):
        raise NotImplementedError

    def as_dict(self):
        """
        Return the enhanced base configuration with the configuration properties
        :return:
        """
        conf = self.baseconf
        conf['output_directory'] = self.outputdir()
        conf['general']['n_drones'] = self.n_drones
        conf['charging_strategy'] = self.charging_strategy
        conf['charging_optimization']['W'] = self.W
        conf['charging_optimization']['sigma'] = self.sigma
        conf['charging_optimization']['schedule_delta'] = self.schedule_delta
        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, n_drones):
        super().__init__(baseconf, basedir, "naive", n_drones, 0, 0, 0)

    def outputdir(self):
        return os.path.join(self.basedir, f"naive_{self.n_drones}")


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, n_drones, W, sigma, schedule_delta):
        super().__init__(baseconf, basedir, "milp", n_drones, W, sigma, schedule_delta)

    def outputdir(self):
        return os.path.join(self.basedir, f"{self.charging_strategy}_ndrones{self.n_drones}_sigma{self.sigma}_W{self.W}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    confs = [
        # NaiveConfiguration(baseconf, "out/villalvernia", 3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, W=3, sigma=4, schedule_delta=0.5),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=4, sigma=6, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=4, sigma=10, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=3, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=6, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=7, sigma=10, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=3, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=6, schedule_delta=4),
        # MilpConfiguration(baseconf, "out/villalvernia", 3, W=10, sigma=10, schedule_delta=4),
    ]

    for conf in confs:
        try:
            run_from_conf(conf.as_dict())
        except Exception:
            logger.error("failed to run configuration")
