import json
import logging
import os
from datetime import datetime


class ConfigurationManager:
    """
    Class for caching prior results related to the management of different configurations
    """
    def __init__(self, rootdir):
        self.logger = logging.getLogger(__name__)

        self.fingerprints = {}

        for trialdir in os.listdir(rootdir):
            for ts_dir in os.listdir(os.path.join(rootdir, trialdir)):
                result_fpath = os.path.join(rootdir, trialdir, ts_dir, "result.json")
                try:
                    with open(result_fpath, 'r') as f:
                        parsed = json.load(f)
                except FileNotFoundError:
                    continue

                if not parsed.get('success', False):
                    # only successful runs count
                    continue

                charging_strategy = parsed['scheduler']
                n_drones = parsed['scenario']['nr_drones']
                W_hat = parsed['sched_params']['W_hat']
                sigma = parsed['sched_params']['sigma']
                flight_sequence_fpath = parsed['scenario']['source_file']
                v = parsed['sched_params']['v'][0]
                r_charge = parsed['sched_params']['r_charge'][0]
                r_deplete = parsed['sched_params']['r_deplete'][0]
                time_limit = parsed['sched_params']['time_limit']
                int_feas_tol = parsed['sched_params']['int_feas_tol']
                pi = parsed['sched_params']['pi']
                B_min = parsed['sched_params']['B_min'][0]
                N_s = parsed['scenario']['nr_charging_stations']

                fp = (charging_strategy, n_drones, W_hat, sigma, flight_sequence_fpath, v, r_charge, r_deplete, time_limit, int_feas_tol, pi, B_min, N_s, trialdir)
                self.fingerprints[fp] = os.path.join(trialdir, ts_dir)
        self.logger.info(f"extracted {len(self.fingerprints)} fingerprints from the directory")

    def seen(self, conf):
        fp = self.fingerprint(conf)
        return self.fingerprints.get(fp)

    def fingerprint(self, conf):
        """
        Return the unique fingerprint for a given configuration.
        Can be used to compare configurations against each other
        """
        charging_strategy = conf.charging_strategy + "scheduler"
        return charging_strategy, conf.n_drones, conf.W_hat, conf.sigma, conf.flight_sequence_fpath, conf.v, conf.r_charge, conf.r_deplete, conf.time_limit, conf.int_feas_tol, conf.pi, conf.B_min, conf.N_s, f"{conf.trial}"


class Configuration:
    def __init__(self, baseconf: dict, basedir, trial, charging_strategy, n_drones, W_hat, sigma, pi, flight_sequence_fpath, v=None, r_charge=None, r_deplete=None, time_limit=None, int_feas_tol=None, B_min=None):
        self.baseconf = baseconf
        self.basedir = basedir
        self.trial = trial
        self.charging_strategy = charging_strategy
        self.n_drones = n_drones
        self.W_hat = W_hat
        self.sigma = sigma
        self.pi = pi
        self.flight_sequence_fpath = flight_sequence_fpath
        self.v = v if v else baseconf['charging_optimization'].get('v')
        self.r_charge = r_charge if r_charge else baseconf['charging_optimization'].get('r_charge')
        self.r_deplete = r_deplete if r_deplete else baseconf['charging_optimization'].get('r_deplete')
        self.time_limit = time_limit if time_limit else baseconf['charging_optimization'].get('time_limit')
        self.int_feas_tol = int_feas_tol if int_feas_tol else baseconf['charging_optimization'].get('int_feas_tol')
        self.B_min = B_min if B_min else baseconf['charging_optimization'].get('B_min')
        self.ts = datetime.now()

    def outputdir(self):
        return os.path.join(self.basedir, f"{self.trial}", f"{self.ts}")

    @property
    def N_s(self):
        return len(self.baseconf['charging_optimization']['charging_positions'])

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
        conf['charging_optimization']['W_hat'] = self.W_hat
        conf['charging_optimization']['sigma'] = self.sigma
        conf['charging_optimization']['time_limit'] = self.time_limit
        conf['charging_optimization']['int_feas_tol'] = self.int_feas_tol
        conf['charging_optimization']['pi'] = self.pi
        conf['charging_optimization']['r_charge'] = self.r_charge
        conf['charging_optimization']['r_deplete'] = self.r_deplete
        conf['charging_optimization']['B_min'] = self.B_min
        conf['charging_optimization']['v'] = self.v

        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, trial, n_drones, flight_sequence_fpath, r_charge=None, r_deplete=None, B_min=None):
        super().__init__(baseconf, basedir, trial, "naive", n_drones, 0, 0, 1, flight_sequence_fpath, r_charge=r_charge, r_deplete=r_deplete, B_min=B_min)


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, trial, n_drones, W_hat, sigma, pi, flight_sequence_fpath, time_limit=None, int_feas_tol=None, r_charge=None, r_deplete=None, B_min=None):
        super().__init__(baseconf, basedir, trial, "milp", n_drones, W_hat, sigma, pi, flight_sequence_fpath, time_limit=time_limit, int_feas_tol=int_feas_tol, r_charge=r_charge, r_deplete=r_deplete, B_min=B_min)
