class Event:
    def __init__(self, ts, name, node, uav=None, battery=1):
        self.ts = ts
        self.uav = uav
        self.name = name
        self.node = node
        self.battery = battery

    def __repr__(self):
        return f"{self.ts}, {self.uav}, {self.name}, {self.node}"