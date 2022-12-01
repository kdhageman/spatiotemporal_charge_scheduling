import logging
import os

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

    basedir = "out/villalvernia/grid_search.test"
    sigmas = [7, 13, 19, 31, 43]
    pis = [9, 19, 49, 99, 149]
    Ws = [10, 20, 50, 100, 170]

    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
    ]

    for sigma in sigmas:
        for pi in pis:
            for W_hat in Ws:
                if pi > W_hat:
                    # impossible
                    continue
                if sigma > W_hat:
                    # impossible
                    continue
                conf = MilpConfiguration(baseconf, basedir, 3, sigma=sigma, W_hat=W_hat, pi=pi, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30)
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
            # raise e
