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
    flight_sequence_fpath4 = "out/flight_sequences/villalvernia_4/flight_sequences.pkl"  # ~120 waypoints per UAV
    flight_sequence_fpath5 = "out/flight_sequences/villalvernia_5/flight_sequences.pkl"  # ~95 waypoints per UAV
    flight_sequence_fpath6 = "out/flight_sequences/villalvernia_6/flight_sequences.pkl"  # ~80 waypoints per UAV

    confs = [
        NaiveConfiguration(baseconf, "out/villalvernia", 3, flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=3, W=4, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=3, W=7, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=6, W=4, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=6, W=7, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=10, W=4, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=10, W=7, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        MilpConfiguration(baseconf, "out/villalvernia", 3, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath3),
        NaiveConfiguration(baseconf, "out/villalvernia", 4, flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=3, W=4, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=3, W=7, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=6, W=4, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=6, W=7, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=10, W=4, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=10, W=7, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath4),
        MilpConfiguration(baseconf, "out/villalvernia", 4, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath4),
        NaiveConfiguration(baseconf, "out/villalvernia", 5, flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=3, W=4, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=3, W=7, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=6, W=4, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=6, W=7, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=10, W=4, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=10, W=7, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath5),
        MilpConfiguration(baseconf, "out/villalvernia", 5, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath5),
        NaiveConfiguration(baseconf, "out/villalvernia", 6, flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=3, W=4, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=3, W=7, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=3, W=10, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=6, W=4, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=6, W=7, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=6, W=10, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=10, W=4, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=10, W=7, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=10, W=10, flight_sequence_fpath=flight_sequence_fpath6),
        MilpConfiguration(baseconf, "out/villalvernia", 6, sigma=16, W=10, flight_sequence_fpath=flight_sequence_fpath6),
    ]

    for conf in confs:
        try:
            if not os.path.exists(conf.outputdir()):
                schedule_charge_from_conf(conf.as_dict())
            else:
                logger.info(f"skipping configuration because it already exists ({conf.outputdir()})")
        except Exception as e:
            logger.error(f"failed to run configuration: {e}")
