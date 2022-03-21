import yaml
from yaml import Loader


class Scenario:
    def __init__(self, fname):
        with open(fname, 'r') as f:
            doc = yaml.load(f, Loader=Loader)

        self.positions_S = []
        for charging_station in doc['charging_stations']:
            x, y = charging_station['x'], charging_station['y']
            self.positions_S.append((x, y))

        self.positions_w = []
        for drone in doc['drones']:
            waypoints = []
            for waypoint in drone['waypoints']:
                x, y = waypoint['x'], waypoint['y']
                waypoints.append((x, y))
            self.positions_w.append(waypoints)
