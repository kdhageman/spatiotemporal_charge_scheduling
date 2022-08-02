import logging
from datetime import datetime
from typing import Callable, List

import simpy

from simulate.event import Event, EventType
from simulate.node import NodeType


class Strategy:
    def __init__(self):
        self.cb = None
        self.logger = logging.getLogger(__name__)

    def set_cb(self, cb: Callable[[List[Event]], None]):
        self.cb = cb

    def handle_event(self, event):
        """
        Allows the strategy to react to events from the UAVs
        :return:
        """
        pass

    def sim(self, env):
        """
        Allows the strategy to generate simpy events itself for scheduling
        :param env:
        :return:
        """
        while True:
            try:
                yield env.timeout(10)
            except simpy.exceptions.Interrupt:
                break

    def debug(self, env, msg):
        self.logger.debug(f"[{datetime.now()}] [{env.now:.2f}] {msg}")


class IntervalStrategy(Strategy):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.changed_since_last = []

    def handle_event(self, event: simpy.Event):
        uav_id = event.value.uav.uav_id
        forced = event.value.forced
        if uav_id not in self.changed_since_last and not forced:
            self.changed_since_last.append(uav_id)

    def sim(self, env: simpy.Environment):
        while True:
            event = env.timeout(self.interval)
            try:
                yield event
                if len(self.changed_since_last) > 0:
                    self.debug(env, f"recurring rescheduling triggered by changes in UAVs {self.changed_since_last}")
                    self.cb('all')
                    self.changed_since_last = []
                else:
                    self.debug(env, f"recurring rescheduling skipped because no UAV changed its status")
            except simpy.exceptions.Interrupt:
                break


class OnEventStrategy(Strategy):
    def __init__(self, interval: float = 0):
        super().__init__()
        self.interval = interval
        self.last_time = 0

    def handle_event(self, event: simpy.Event):
        uav_id = event.value.uav.uav_id
        if self.last_time is None or event.env.now >= self.last_time + self.interval:
            self.logger.debug(f"[{event.env.now:.2f}] rescheduling triggered by UAV [{event.value.uav.uav_id}] for UAVs: {self._uavs(uav_id)}")
            self.cb(self._uavs(uav_id))
            self.last_time = event.env.now
        # else:
        #     self.debug(event.env, f"skipping scheduling triggered by UAV [{uav_id}] because most recent reschedule was too soon")

    def _uavs(self, uav_id: int) -> List[int]:
        """
        Returns which UAVs must be scheduled
        Must be implemented by subclasses
        :param uav_id: uav of the triggered event
        """
        raise NotImplementedError


class OnEventStrategySingle(OnEventStrategy):
    def _uavs(self, uav_id):
        return [uav_id]


class OnEventStrategyAll(OnEventStrategy):
    def _uavs(self, uav_id):
        return 'all'


class AfterNEventsStrategy(Strategy):
    def __init__(self, n):
        super().__init__()
        self.logger.debug(f"[{datetime.now()}] triggering rescheduling after {n} events")
        self.events_seen = {}
        self.n = n
        self.last_time = 0

    def handle_event(self, event: simpy.Event):
        if event.value.type == EventType.reached and event.value.node.node_type == NodeType.Waypoint:
            uav_id = event.value.uav.uav_id
            events_seen_for_uav = self.events_seen.get(uav_id, 0) + 1
            self.events_seen[uav_id] = events_seen_for_uav

            if events_seen_for_uav >= self.n:
                # reschedule
                self.logger.debug(f"[{datetime.now()}] [{event.env.now:.2f}] rescheduling triggered by UAV [{event.value.uav.uav_id}] for UAVs: {self._uavs(uav_id)}")
                self.cb(self._uavs(uav_id))
                self.last_time = event.env.now

                self.events_seen = {}


class AfterNEventsStrategySingle(AfterNEventsStrategy):
    def _uavs(self, uav_id):
        return [uav_id]


class AfterNEventsStrategyAll(AfterNEventsStrategy):
    def _uavs(self, uav_id):
        return 'all'
