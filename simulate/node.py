from enum import Enum

import numpy as np

from util.distance import dist3


class NodeType(Enum):
    AuxWaypoint = "auxiliary_waypoint"
    Waypoint = "waypoint"
    ChargingStation = "charging_station"


class Node:
    def __init__(self, x, y, z, wt, ct):
        self.x = x
        self.y = y
        self.z = z
        self.wt = wt
        self.ct = ct

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    def equal_pos(self, other):
        """
        Returns whether this node is at the same positions as the other
        """
        return np.array_equal(self.pos, other.pos)

    def dist(self, other):
        return dist3(self.pos, other.pos)

    def direction(self, other):
        """
        Unit vector in the direction of the other node
        """
        dir_vector = other.pos - self.pos
        if np.linalg.norm(dir_vector) == 0:
            return np.array([0, 0, 0])
        return dir_vector / np.linalg.norm(dir_vector)

    @property
    def node_type(self):
        raise NotImplementedError

    def __repr__(self):
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


class ChargingStation(Node):
    def __init__(self, x, y, z, identifier, wt=0, ct=0):
        super().__init__(x, y, z, wt, ct)
        self.identifier = identifier

    @property
    def node_type(self):
        return NodeType.ChargingStation

    def __repr__(self):
        return f"charging station ({self.identifier}) {super().__repr__()}"


class Waypoint(Node):
    def __init__(self, x, y, z, strided=False):
        super().__init__(x, y, z, 0, 0)
        self.strided = strided

    @property
    def node_type(self):
        return NodeType.Waypoint


class AuxWaypoint(Node):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0, 0)

    @property
    def node_type(self):
        return NodeType.AuxWaypoint
