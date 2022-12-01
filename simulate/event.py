from enum import Enum


class EventType(Enum):
    reached = 'reached'
    waited = 'waited'
    charged = 'charged'
    started = 'started'
    changed_course = 'changed_course'
    crashed = 'crashed'


class Event:
    def __init__(self, ts_start: float, duration: float, type: str, node, uav=None, battery: float = 1, depletion: float = 0, forced: bool = False):
        self.t_start = ts_start
        self.duration = duration
        self.uav = uav
        self.type = type
        self.node = node
        self.battery = battery
        self.forced = forced
        self.depletion = depletion

    @property
    def t_end(self):
        return self.t_start + self.duration

    @property
    def pre_battery(self):
        return self.battery + self.depletion

    def __repr__(self):
        return f"{self.uav.uav_id}, <{self.t_start:.2f}-{self.t_end:.2f}> ({self.duration:.2f}), {self.type}, {self.node}, {self.battery:.2f}, {self.depletion:.3f}, {self.forced}"


class ReachedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.reached, node, uav=uav, battery=battery, depletion=depletion, forced=forced)


class WaitedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.waited, node, uav=uav, battery=battery, depletion=depletion, forced=forced)


class ChargedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.charged, node, uav=uav, battery=battery, depletion=depletion, forced=forced)


class StartedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.started, node, uav=uav, battery=battery, depletion=depletion, forced=forced)


class ChangedCourseEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.changed_course, node, uav=uav, battery=battery, depletion=depletion, forced=forced)


class CrashedEvent(Event):
    def __init__(self, ts_start, duration, node, uav=None, battery=1, depletion=0, forced=False):
        super().__init__(ts_start, duration, EventType.crashed, node, uav=uav, battery=battery, depletion=depletion, forced=forced)
