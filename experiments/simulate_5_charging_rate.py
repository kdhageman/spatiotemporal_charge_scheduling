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

    basedir = "out/villalvernia/5_charging_rate"
    # increase sigmas
    confs = [
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.0007, r_deplete=0.007),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.0014, r_deplete=0.007),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.0035, r_deplete=0.007),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.007, r_deplete=0.007),
        NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.014, r_deplete=0.007),
        # NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.035, r_deplete=0.007),
        # NaiveConfiguration(baseconf, basedir, 3, flight_sequence_fpath=flight_sequence_fpath3, r_charge=0.07, r_deplete=0.007),
        MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.0007, r_deplete=0.007, int_feas_tol=1e-7),
        MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.0014, r_deplete=0.007, int_feas_tol=1e-7),
        MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.0035, r_deplete=0.007, int_feas_tol=1e-7),
        MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.007, r_deplete=0.007, int_feas_tol=1e-7),
        MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.014, r_deplete=0.007, int_feas_tol=1e-7),
        # MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.035, r_deplete=0.007, int_feas_tol=1e-7),
        # MilpConfiguration(baseconf, basedir, 3, W=8, sigma=24, flight_sequence_fpath=flight_sequence_fpath3, time_limit=10, rescheduling_frequency=84, r_charge=0.07, r_deplete=0.007, int_feas_tol=1e-7),  # TODO: fix bug where problem is not solvable
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
