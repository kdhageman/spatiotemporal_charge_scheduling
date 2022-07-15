from enum import Enum

import numpy as np


class EventType(Enum):
    reached = 1
    waited = 2
    charged = 3
    started = 4
    changed_course = 5


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


class ReachedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1):
        super().__init__(ts_start, duration, EventType.reached, node, uav=uav, battery=battery)


class WaitedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1):
        super().__init__(ts_start, duration, EventType.waited, node, uav=uav, battery=battery)


class ChargedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1):
        super().__init__(ts_start, duration, EventType.charged, node, uav=uav, battery=battery)


class StartedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1):
        super().__init__(ts_start, duration, EventType.started, node, uav=uav, battery=battery)


class ChangedCourseEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1):
        super().__init__(ts_start, duration, EventType.changed_course, node, uav=uav, battery=battery)
