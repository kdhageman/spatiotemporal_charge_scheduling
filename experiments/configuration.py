import json
import os
from datetime import datetime


class Configuration:
    def __init__(self, baseconf: dict, basedir, charging_strategy, n_drones, W_hat, sigma, pi, flight_sequence_fpath, v=None, r_charge=None, r_deplete=None, time_limit=None, int_feas_tol=None, B_min=None):
        self.baseconf = baseconf
        self.basedir = basedir
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
        return os.path.join(self.basedir, f"{self.ts}")

    def experiment_already_exists(self):
        """
        Returns whether an existing experiment in the given directory with the same parameters as this configuration already exists.
        :param directory:
        :return:
        """
        if not os.path.exists(self.basedir):
            return None

        for subdir in os.listdir(self.basedir):
            result_fpath = os.path.join(self.basedir, subdir, "result.json")
            try:
                with open(result_fpath, 'r') as f:
                    parsed = json.load(f)
            except FileNotFoundError:
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

            if self.charging_strategy + "scheduler" == charging_strategy and self.n_drones == n_drones and self.W_hat == W_hat and self.sigma == sigma and self.flight_sequence_fpath == flight_sequence_fpath and self.v == v \
                    and self.r_charge == r_charge and self.r_deplete == r_deplete and self.time_limit == time_limit and self.int_feas_tol == int_feas_tol and self.pi == pi and self.B_min == B_min:
                return subdir
        return None

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

        return conf


class NaiveConfiguration(Configuration):
    def __init__(self, baseconf, basedir, n_drones, flight_sequence_fpath, r_charge=None, r_deplete=None, B_min=None):
        super().__init__(baseconf, basedir, "naive", n_drones, 0, 0, 1, flight_sequence_fpath, r_charge=r_charge, r_deplete=r_deplete, B_min=B_min)


class MilpConfiguration(Configuration):
    def __init__(self, baseconf: dict, basedir, n_drones, W_hat, sigma, pi, flight_sequence_fpath, time_limit=None, int_feas_tol=None, r_charge=None, r_deplete=None, B_min=None):
        super().__init__(baseconf, basedir, "milp", n_drones, W_hat, sigma, pi, flight_sequence_fpath, time_limit=time_limit, int_feas_tol=int_feas_tol, r_charge=r_charge, r_deplete=r_deplete, B_min=B_min)
