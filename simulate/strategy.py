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


class OnEventStrategySingle(Strategy):
    def handle_event(self, event):
        self.cb([event.value.uav.uav_id])
