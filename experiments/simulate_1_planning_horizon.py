import logging
import os
import yaml

from experiments.configuration import NaiveConfiguration, MilpConfiguration
from experiments.util_funcs import schedule_charge_from_conf

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"  # ~160 waypoints per UAV

    basedir = "out/villalvernia/1_planning_horizon/increase_sigma"
    # increase sigmas
    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=1, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=2, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=4, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=5, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=7, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=8, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=11, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=12, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=13, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=14, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=15, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=17, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=18, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=19, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
        MilpConfiguration(baseconf, basedir, 3, sigma=20, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30),
    ]

    basedir = "out/villalvernia/1_planning_horizon/increase_W"
    # full coverage, but increase W
    confs += [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, basedir, 3, sigma=34, W=6, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=29, W=7, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=25, W=8, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=22, W=9, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=19, W=10, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=17, W=11, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=16, W=12, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=15, W=13, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=14, W=14, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=13, W=15, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=12, W=16, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=11, W=17, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=10, W=18, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=19, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
        MilpConfiguration(baseconf, basedir, 3, sigma=9, W=20, flight_sequence_fpath=flight_sequence_fpath3, time_limit=1800, rescheduling_frequency=85),
    ]

    for conf in confs:
        try:
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.error(f"failed to run configuration: {e}")
