import numpy as np


class Event:
    def __init__(self, ts_start, duration, name, node, uav=None, battery=1):
        self.ts_start = ts_start
        self.duration = duration
        self.uav = uav
        self.name = name
        self.node = node
        self.battery = battery
        
    @property
    def ts_end(self):
        return self.ts_start + self.duration

    def __repr__(self):
        return f"{self.uav.uav_id}, <{self.ts_start:.2f}-{self.ts_end:.2f}> ({self.duration:.2f}), {self.name}, {self.node}, {np.round(self.battery, 1)}"
