import copy
import logging
import math
import os
from itertools import product

import yaml
from tqdm import tqdm
import sys

sys.path.append(".")
from experiments.configuration import MilpConfiguration, NaiveConfiguration
from experiments.util_funcs import load_flight_sequences, schedule_charge_from_conf


def coarse_configs(r_charges, number_of_charging_stations, r_deplete, B_min, n_trials):
    """
    Returns the configuration for the simulation of the coarse (i.e., voxel size=5.1) experiment
    """
    flight_seq_fpath = "out/flight_sequences/villalvernia_3.vs_51/flight_sequences.pkl"
    flight_sequences = load_flight_sequences(flight_seq_fpath)

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    sigma = 1
    pi = math.inf
    W_hat = max([len(x) for x in flight_sequences]) - 1
    time_limit = 600
    basedir = "out/villalvernia/charging_station_utility"

    charging_stations = [
        [-10, 10, 0],
        [-10, 10, 0],
        [-10, 10, 0]
    ]

    confs = []
    for N_s, r_charge in product(number_of_charging_stations, r_charges):
        for trial in range(1, 1 + n_trials):
            basedir_trial = os.path.join(basedir, f"{trial}")
            conf = MilpConfiguration(
                copy.deepcopy(baseconf),
                basedir_trial,
                3,
                sigma=sigma,
                pi=pi,
                W_hat=W_hat,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.B_min = B_min
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations[:N_s]
            confs.append(conf)

            conf = NaiveConfiguration(
                copy.deepcopy(baseconf),
                basedir_trial,
                3,
                flight_sequence_fpath=flight_seq_fpath,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.B_min = B_min
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations[:N_s]
            confs.append(conf)

    return confs


def fine_configs(r_charges, number_of_charging_stations, r_deplete, B_min, n_trials):
    """
    Returns the configuration for the simulation of the fine-grained (i.e., voxel size=2) experiment
    """
    flight_seq_fpath = "out/flight_sequences/villalvernia_3.vs_20/flight_sequences.pkl"

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    sigma = 4
    pi = 36
    W_hat = 40
    time_limit = 600
    basedir = "out/villalvernia/charging_station_utility"

    charging_stations_compositions = {
        1: [
            [-5.89549398, 13.3062412, 0]
        ],
        2: [
            [-9.2918538, 18.04343851, 0],
            [-2.49913417, 8.56904388, 0],
        ],
        3: [
            [-9.2918538, 18.04343851, 0],
            [-5.89549398, 13.3062412, 0]
            [-2.49913417, 8.56904388, 0],
        ]
    }

    confs = []
    for N_s, r_charge in product(number_of_charging_stations, r_charges):
        for trial in range(1, 1 + n_trials):
            basedir_trial = os.path.join(basedir, f"{trial}")
            conf = MilpConfiguration(
                copy.deepcopy(baseconf),
                basedir_trial,
                3,
                sigma=sigma,
                pi=pi,
                W_hat=W_hat,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.B_min = B_min
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

            conf = NaiveConfiguration(
                copy.deepcopy(baseconf),
                basedir_trial,
                3,
                flight_sequence_fpath=flight_seq_fpath,
                r_deplete=r_deplete,
                r_charge=r_charge,
            )
            conf.B_min = B_min
            conf.baseconf['charging_optimization']['charging_positions'] = charging_stations_compositions[N_s]
            confs.append(conf)

    return confs


def main():
    logger = logging.getLogger(__name__)

    number_of_charging_stations = [1, 2, 3]
    r_charges = [1 / 5400, 1 / 3600, 1 / 3000, 1 / 2400, 1 / 1800, 1 / 1200, 1 / 600, 1 / 300]
    r_deplete = 1 / 600
    B_min = 0.73
    n_trials = 1

    confs = []
    confs += coarse_configs(r_charges, number_of_charging_stations, r_deplete, B_min, n_trials)
    confs += fine_configs(r_charges, number_of_charging_stations, r_deplete, B_min, n_trials)

    for i, conf in enumerate(tqdm(confs)):
        try:
            existing_experiment_dir = conf.experiment_already_exists()
            if not existing_experiment_dir:
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({existing_experiment_dir})")
        except Exception as e:
            raise e
            logger.exception(f"failed to run configuration")
            error_file = os.path.join(conf.outputdir(), "error.txt")
            os.makedirs(conf.outputdir(), exist_ok=True)
            with open(error_file, 'w') as f:
                f.write(f"failed: {e}")


if __name__ == "__main__":
    main()
