import logging

import simpy


class Strategy:
    def __init__(self):
        self.cb = None
        self.logger = logging.getLogger(__name__)

    def set_cb(self, cb):
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
        self.logger.debug(f"[{env.now:.2f}] {msg}")


class IntervalStrategy(Strategy):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.changed_since_last = []

    def handle_event(self, event):
        uav_id = event.value.uav.uav_id
        forced = event.value.forced
        if uav_id not in self.changed_since_last and not forced:
            self.changed_since_last.append(uav_id)

    def sim(self, env):
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
    def __init__(self, interval=0):
        super().__init__()
        self.interval = interval
        self.last_time = 0

    def handle_event(self, event):
        uav_id = event.value.uav.uav_id
        if self.last_time is None or event.env.now >= self.last_time + self.interval:
            self.logger.debug(f"[{event.env.now:.2f}] rescheduling triggered by UAV [{event.value.uav.uav_id}] for UAVs: {self._uavs(uav_id)}")
            self.cb(self._uavs(uav_id))
            self.last_time = event.env.now
        # else:
        #     self.debug(event.env, f"skipping scheduling triggered by UAV [{uav_id}] because most recent reschedule was too soon")

    def _uavs(self, uav_id):
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
