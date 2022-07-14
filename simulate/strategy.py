import simpy


class Strategy:
    def __init__(self):
        self.cb = None

    def set_cb(self, cb):
        self.cb = cb

    def handle_event(self, event):
        """
        Allows the strategy to react to events from the UAVs
        :return:
        """
        raise NotImplementedError

    def sim(self, env):
        """
        Allows the strategy to generate simpy events itself for scheduling
        :param env:
        :return:
        """
        raise NotImplementedError


class IntervalStrategy(Strategy):
    def __init__(self, interval):
        self.interval = interval

    def handle_event(self, event):
        pass

    def sim(self, env):
        while True:
            event = env.timeout(self.interval)
            try:
                yield event
                self.cb('all')
            except simpy.exceptions.Interrupt:
                break
        # TODO: add finish callbacks?


class ArrivalStrategy(Strategy):
    def handle_event(self, event):
        self.cb([event.value.uav.uav_id])

    def sim(self, env):
        while True:
            try:
                yield env.timeout(10)
            except simpy.exceptions.Interrupt:
                break

