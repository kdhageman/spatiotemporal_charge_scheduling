import logging
import os
import pickle

import yaml

from experiments.configuration import NaiveConfiguration, MilpConfiguration
from experiments.util_funcs import schedule_charge_from_conf, load_flight_sequences

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"  # ~160 waypoints per UAV
    flight_sequences = load_flight_sequences(flight_sequence_fpath3)
    min_nr_waypoints = min(len(seq) for seq in flight_sequences)

    basedir = "out/villalvernia/grid_search"
    # increase sigmas
    sigmas = [1, 4, 7, 10, 13, 16, 19, 25, 31, 36]
    # fix rescheduling frequencies vs Ws
    rescheduling_frequencies = [3, 5, 7, 9, 11, 13, 25, 50, 75, 100, 125, 150]
    Ws = [5, 10, 15]

    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
    ]

    for sigma in sigmas:
        for rescheduling_frequency in rescheduling_frequencies:
            for W in Ws:
                # if (W == 15 and rescheduling_frequency == 3 and sigma == 10):
                #     continue
                conf = MilpConfiguration(baseconf, basedir, 3, sigma=sigma, W=W, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=rescheduling_frequency)
                confs.append(conf)

    for conf in confs:
        try:
            # schedule_charge_from_conf(conf.as_dict())
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.error(f"failed to run configuration: {e}")
            error_file = os.path.join(conf.outputdir(), "error.txt")
            with open(error_file, 'w') as f:
                f.write(f"failed: {e}")
            raise e
