class Event:
    def __init__(self, ts, name, node, uav=None):
        self.ts = ts
        self.uav = uav
        self.name = name
        self.node = node