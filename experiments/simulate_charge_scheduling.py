import logging
import sys
import yaml
from experiments.util_funcs import schedule_charge_from_conf

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must provide path to configuration file as parameter")
        sys.exit(1)
    fpath_conf = sys.argv[1]
    with open(fpath_conf, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    schedule_charge_from_conf(conf)
