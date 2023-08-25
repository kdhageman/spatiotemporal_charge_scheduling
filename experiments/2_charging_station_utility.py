import copy
import logging
import math
import os
from itertools import product

import numpy as np
import yaml
from tqdm import tqdm
import sys

sys.path.append(".")
from experiments.configuration import MilpConfiguration, NaiveConfiguration, ConfigurationManager
from experiments.util_funcs import load_flight_sequences, schedule_charge_from_conf

time_limit = 300
charging_stations_compositions = {
    1: [
        [-9.2918538, 18.04343851, 0],
    ],
    2: [
        [-9.2918538, 18.04343851, 0],
        [-5.89549398, 13.3062412, 0],
    ],
    3: [
        [-9.2918538, 18.04343851, 0],
        [-5.89549398, 13.3062412, 0],
        [-2.49913417, 8.56904388, 0],
    ]
}
basedir = "out/villalvernia/charging_station_utility"


def coarse_configs(number_of_charging_stations, r_charges, r_deplete, n_trials):
    """
    Returns the configuration for the simulation of the coarse (i.e., voxel size=5.1) experiment
    """
    flight_seq_fpath = "out/flight_sequences/villalvernia_3.vs_52/flight_sequences.pkl"
    flight_sequences = load_flight_sequences(flight_seq_fpath)

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    sigma = 1
    pi = math.inf
    W_hat = max([len(x) for x in flight_sequences]) - 1

    confs = []
    for N_s, r_charge in product(number_of_charging_stations, r_charges):
        for trial in range(1, 1 + n_trials):
            conf = MilpConfiguration(
                copy.deepcopy(baseconf),
                basedir,
                trial,
                3,
                sigma=sigma,
                pi=pi,
                W_hat=W_hat,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

            conf = NaiveConfiguration(
                copy.deepcopy(baseconf),
                basedir,
                trial,
                3,
                flight_sequence_fpath=flight_seq_fpath,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

    return confs


def fine_configs(number_of_charging_stations, r_charges, r_deplete, n_trials):
    """
    Returns the configuration for the simulation of the fine-grained (i.e., voxel size=2) experiment
    """
    flight_seq_fpath = "out/flight_sequences/villalvernia_3.vs_30/flight_sequences.pkl"

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    sigma = 8
    pi = np.inf
    W_hat = 75

    confs = []
    for N_s, r_charge in product(number_of_charging_stations, r_charges):
        for trial in range(1, 1 + n_trials):
            conf = MilpConfiguration(
                copy.deepcopy(baseconf),
                basedir,
                trial,
                3,
                sigma=sigma,
                pi=pi,
                W_hat=W_hat,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

            conf = NaiveConfiguration(
                copy.deepcopy(baseconf),
                basedir,
                trial,
                3,
                flight_sequence_fpath=flight_seq_fpath,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

    return confs


def main():
    logger = logging.getLogger(__name__)

    number_of_charging_stations = [1, 2, 3]
    r_charges = [1 / 5400, 1 / 3600, 1 / 3000, 1 / 2400, 1 / 1800, 1 / 1200, 1 / 600, 1 / 300]
    r_deplete = 1 / 600
    n_trials = 1

    confs = []
    confs += coarse_configs(number_of_charging_stations, r_charges, r_deplete, n_trials)
    # confs += fine_configs(number_of_charging_stations, r_charges, r_deplete, B_min, n_trials)

    conf_manager = ConfigurationManager(basedir)
    for i, conf in enumerate(tqdm(confs)):
        try:
            existing_experiment_dir = conf_manager.seen(conf)
            if not existing_experiment_dir:
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({existing_experiment_dir})")
        except Exception as e:
            # raise e
            logger.exception(f"failed to run configuration")
            error_file = os.path.join(conf.outputdir(), "error.txt")
            os.makedirs(conf.outputdir(), exist_ok=True)
            with open(error_file, 'w') as f:
                f.write(f"failed: {e}")


if __name__ == "__main__":
    main()
