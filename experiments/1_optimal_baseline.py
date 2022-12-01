import logging
import math
import os

import yaml
from experiments.configuration import MilpConfiguration, NaiveConfiguration
from experiments.util_funcs import load_flight_sequences, schedule_charge_from_conf

import sys

sys.setrecursionlimit(100000)


def main():
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    confs = []

    # optimal solution
    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3.fewervoxels/flight_sequences.pkl"  # ~34 waypoints per vehicle
    flight_sequences = load_flight_sequences(flight_sequence_fpath3)
    basedir = "out/villalvernia/baseline"
    W_hat = max([len(x) for x in flight_sequences])
    sigma = 1
    pi = math.inf
    time_limit = 3600
    conf = MilpConfiguration(
        baseconf,
        basedir,
        3,
        sigma=sigma,
        W_hat=W_hat,
        pi=pi,
        flight_sequence_fpath=flight_sequence_fpath3,
        time_limit=time_limit,
        r_charge=0.0005025,
        r_deplete=0.0045,
    )
    confs.append(conf)

    # suboptimal solution
    conf = MilpConfiguration(
        baseconf,
        basedir,
        3,
        sigma=2,
        W_hat=25,
        pi=20,
        flight_sequence_fpath=flight_sequence_fpath3,
        time_limit=time_limit,
        r_charge=0.0005025,
        r_deplete=0.0045,
    )
    confs.append(conf)

    # naive solution
    conf = NaiveConfiguration(
        baseconf,
        basedir,
        3, flight_sequence_fpath=flight_sequence_fpath3,
        r_charge=0.0005025,
        r_deplete=0.0045,
    )
    confs.append(conf)

    for conf in confs:
        try:
            # schedule_charge_from_conf(conf.as_dict())
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.exception(f"failed to run configuration")
            error_file = os.path.join(conf.outputdir(), "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"failed: {e}")


if __name__ == "__main__":
    main()
