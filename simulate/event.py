class Event:
    def __init__(self, ts, name, node, uav=None):
        self.ts = ts
        self.uav = uav
        self.name = name
        self.node = node

    def __repr__(self):
        return f"{self.ts}, {self.uav}, {self.name}, {self.node}"