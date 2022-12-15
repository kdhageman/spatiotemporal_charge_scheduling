import logging
import math
import os
from itertools import product

import numpy as np
import yaml
from tqdm import tqdm

from experiments.configuration import NaiveConfiguration, MilpConfiguration, ConfigurationManager
from experiments.util_funcs import schedule_charge_from_conf, load_flight_sequences


def main():
    logger = logging.getLogger(__name__)

    basedir = "out/villalvernia/grid_search"

    with open("config/charge_scheduling/base.fewervoxels.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_seq_fpath = "out/flight_sequences/villalvernia_3.vs_30/flight_sequences.pkl"

    W_hats = [75, 60, 45, 30, 15]
    pis = [np.inf, 74, 59, 45, 30, 14]
    sigmas = [1, 2, 3, 4, 5, 6, 7, 8]
    time_limit = 300
    r_charge = 1 / 3600
    r_deplete = 1 / 600
    n_trials = 1

    confs = []
    for W_hat, pi, sigma in tqdm(product(W_hats, pis, sigmas)):
        anchorcount = 1 + math.floor(W_hat / sigma)
        if (pi > W_hat) and W_hat != 75:
            # cannot run simulation where rescheduling frequency is larger than horizon
            continue
        elif anchorcount >= 25:
            # this might become infeasible, so don't run
            continue
        for trial in range(1, 1 + n_trials):
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

        conf = NaiveConfiguration(
            baseconf,
            basedir,
            1,
            3,
            flight_sequence_fpath=flight_seq_fpath,
            r_charge=r_charge,
            r_deplete=r_deplete,
        )
        confs.append(conf)

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
