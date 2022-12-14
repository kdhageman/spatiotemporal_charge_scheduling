import logging
import math
import os
import sys

import yaml

sys.path.append(".")
from experiments.configuration import MilpConfiguration, NaiveConfiguration, ConfigurationManager
from experiments.util_funcs import load_flight_sequences, schedule_charge_from_conf

sys.setrecursionlimit(100000)


def main():
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)
        baseconf['charging_optimization']['charging_positions'] = baseconf['charging_optimization']['charging_positions'][:2]

    basedir = "out/villalvernia/optimal_perf"
    r_charge = 1 / 3600
    r_deplete = 1 / 600
    time_limit = 300

    confs = []

    flight_seq_dirs = [
        # "villalvernia_3.vs_100",
        # "villalvernia_3.vs_90",
        # "villalvernia_3.vs_80",
        # "villalvernia_3.vs_75",
        "villalvernia_3.vs_70",
        "villalvernia_3.vs_65",
        "villalvernia_3.vs_60",
        "villalvernia_3.vs_55",
        "villalvernia_3.vs_52",
        "villalvernia_3.vs_51",
        "villalvernia_3.vs_50",
        "villalvernia_3.vs_49",
    ]
    n_trials = 1

    # optimal
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")
        flight_sequences = load_flight_sequences(flight_seq_fpath)

        for trial in range(1, 1 + n_trials):
            W_hat = max([len(x) for x in flight_sequences]) - 1
            sigma = 1
            pi = math.inf
            conf = MilpConfiguration(
                baseconf,
                basedir,
                trial,
                3,
                sigma=sigma,
                W_hat=W_hat,
                pi=pi,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    # suboptimal (sigma=2)
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")
        flight_sequences = load_flight_sequences(flight_seq_fpath)

        for trial in range(1, 1 + n_trials):
            W_hat = max([len(x) for x in flight_sequences]) - 1
            sigma = 2
            pi = math.inf
            conf = MilpConfiguration(
                baseconf,
                basedir,
                trial,
                3,
                sigma=sigma,
                W_hat=W_hat,
                pi=pi,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    # suboptimal (sigma=3)
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")
        flight_sequences = load_flight_sequences(flight_seq_fpath)

        for trial in range(1, 1 + n_trials):
            W_hat = max([len(x) for x in flight_sequences]) - 1
            sigma = 3
            pi = math.inf
            conf = MilpConfiguration(
                baseconf,
                basedir,
                trial,
                3,
                sigma=sigma,
                W_hat=W_hat,
                pi=pi,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    # suboptimal(W_hat=10, sigma=1, pi=8)
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")

        for trial in range(1, 1 + n_trials):
            W_hat = 10
            sigma = 1
            pi = 8
            conf = MilpConfiguration(
                baseconf,
                basedir,
                trial,
                3,
                sigma=sigma,
                W_hat=W_hat,
                pi=pi,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    # suboptimal(W_hat=15, sigma=1, pi=13)
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")

        for trial in range(1, 1 + n_trials):
            W_hat = 15
            sigma = 1
            pi = 13
            conf = MilpConfiguration(
                baseconf,
                basedir,
                trial,
                3,
                sigma=sigma,
                W_hat=W_hat,
                pi=pi,
                flight_sequence_fpath=flight_seq_fpath,
                time_limit=time_limit,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    # Naive
    for flight_seq_dir in flight_seq_dirs:
        flight_seq_fpath = os.path.join("out/flight_sequences", flight_seq_dir, "flight_sequences.pkl")

        for trial in range(1, 1 + n_trials):
            conf = NaiveConfiguration(
                baseconf,
                basedir,
                trial,
                n_drones=3,
                flight_sequence_fpath=flight_seq_fpath,
                r_charge=r_charge,
                r_deplete=r_deplete,
            )
            confs.append(conf)

    conf_manager = ConfigurationManager(basedir)
    for conf in confs:
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
