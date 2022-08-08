import logging
import os
import yaml

from experiments.configuration import NaiveConfiguration, MilpConfiguration
from experiments.util_funcs import schedule_charge_from_conf

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath2 = "out/flight_sequences/villalvernia_2/flight_sequences.pkl"  # [237, 241]
    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"  # [163, 160, 155]
    flight_sequence_fpath4 = "out/flight_sequences/villalvernia_4/flight_sequences.pkl"  # [121, 120, 117, 120]
    flight_sequence_fpath5 = "out/flight_sequences/villalvernia_5/flight_sequences.pkl"  # [96, 94, 97, 95, 96]
    flight_sequence_fpath6 = "out/flight_sequences/villalvernia_6/flight_sequences.pkl"  # [82, 81, 79, 77, 82, 77]
    flight_sequence_fpath7 = "out/flight_sequences/villalvernia_7/flight_sequences.pkl"  # [70, 67, 66, 68, 68, 70, 69]

    basedir = "out/villalvernia/6_charging_cycle"
    # increase sigmas
    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, B_min=0.4),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, B_min=0.45),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, B_min=0.5),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, B_min=0.55),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, B_min=0.6),
        MilpConfiguration(baseconf, basedir, 3, W=30, sigma=3, flight_sequence_fpath=flight_sequence_fpath3, time_limit=20, rescheduling_frequency=44, B_min=0.4),
        MilpConfiguration(baseconf, basedir, 3, W=30, sigma=3, flight_sequence_fpath=flight_sequence_fpath3, time_limit=30, rescheduling_frequency=44, B_min=0.45),
        MilpConfiguration(baseconf, basedir, 3, W=30, sigma=3, flight_sequence_fpath=flight_sequence_fpath3, time_limit=40, rescheduling_frequency=44, B_min=0.5),
        MilpConfiguration(baseconf, basedir, 3, W=30, sigma=3, flight_sequence_fpath=flight_sequence_fpath3, time_limit=50, rescheduling_frequency=44, B_min=0.55),
        MilpConfiguration(baseconf, basedir, 3, W=30, sigma=3, flight_sequence_fpath=flight_sequence_fpath3, time_limit=60, rescheduling_frequency=44, B_min=0.6),
    ]

    for conf in confs:
        try:
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            raise e
            logger.error(f"failed to run configuration: {e}")
