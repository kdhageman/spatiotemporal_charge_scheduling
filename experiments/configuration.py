import os

import numpy as np


class Configuration:
    def __init__(self, baseconf: dict, basedir, charging_strategy, n_drones, W, sigma, flight_sequence_fpath, v=1.5, r_charge=0.00067, r_deplete=0.006, time_limit=60, int_feas_tol=1e-9, rescheduling_frequency=None):
        self.baseconf = baseconf
        self.charging_strategy = charging_strategy
        self.n_drones = n_drones
        self.W = W
        self.sigma = sigma
        self.basedir = basedir
        self.flight_sequence_fpath = flight_sequence_fpath
        self.v = v
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.time_limit = time_limit
        self.int_feas_tol = int_feas_tol
        if rescheduling_frequency:
            self.rescheduling_frequency = rescheduling_frequency
        else:
            self.rescheduling_frequency = sigma * (int(np.ceil(W / 2)) - 1)

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
        conf['charging_optimization']['time_limit'] = self.time_limit
        conf['charging_optimization']['int_feas_tol'] = self.int_feas_tol
        conf['charging_optimization']['rescheduling_frequency'] = self.rescheduling_frequency
        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, n_drones, flight_sequence_fpath):
        super().__init__(baseconf, basedir, "naive", n_drones, 0, 0, flight_sequence_fpath)

    def outputdir(self):
        return os.path.join(self.basedir, f"naive_{self.n_drones}_rc{self.r_charge}_rd{self.r_deplete}")


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, n_drones, W, sigma, flight_sequence_fpath, time_limit=60, int_feas_tol=1e-7, rescheduling_frequency=None):
        super().__init__(baseconf, basedir, "milp", n_drones, W, sigma, flight_sequence_fpath, time_limit=time_limit, int_feas_tol=int_feas_tol, rescheduling_frequency=rescheduling_frequency)

    def outputdir(self):
        return os.path.join(self.basedir, f"{self.charging_strategy}_ndrones{self.n_drones}_sigma{self.sigma}_W{self.W}_tl{self.time_limit}_rc{self.r_charge}_rd{self.r_deplete}_n{self.rescheduling_frequency}")
