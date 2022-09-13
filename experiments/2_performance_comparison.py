import logging

import yaml

from experiments.configuration import MilpConfiguration
from experiments.util_funcs import load_flight_sequences, schedule_charge_from_conf


def main():
    logger = logging.getLogger(__name__)

    with open("config/charge_scheduling/base.yml", 'r') as f:
        baseconf = yaml.load(f, Loader=yaml.Loader)

    flight_sequence_fpath3 = "out/flight_sequences/villalvernia_3/flight_sequences.pkl"  # ~160 waypoints per UAV
    flight_sequences = load_flight_sequences(flight_sequence_fpath3)
    min_nr_waypoints = min(len(seq) for seq in flight_sequences)
    max_nr_waypoints = max(len(seq) for seq in flight_sequences)

    basedir = "out/villalvernia/baseline"
    conf = MilpConfiguration(baseconf, basedir, 3, sigma=10, W=11, flight_sequence_fpath=flight_sequence_fpath3, time_limit=3600, rescheduling_frequency=50)
    schedule_charge_from_conf(conf.as_dict())


if __name__ == "__main__":
    main()
