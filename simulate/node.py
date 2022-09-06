from enum import Enum

import jsons
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

    @property
    def identifier(self):
        raise NotImplementedError

    def dist(self, other):
        return dist3(self.pos, other.pos)

    def __eq__(self, other):
        return type(self) == type(other) and self.x == other.x and self.y == other.y and self.z == other.z and self.wt == other.wt and self.ct == other.ct

    def same_pos(self, other):
        """
        Returns whether the other node occupies the same space
        :param other:
        :return:
        """
        return np.round(self.dist(other), 5) == 0

    def direction(self, other):
        """
        Unit vector in the direction of the other node
        """
        dir_vector = other.pos - self.pos
        if np.linalg.norm(dir_vector) == 0:
            return np.array([0, 0, 0])
        return dir_vector / np.linalg.norm(dir_vector)

    @property
    def node_type(self) -> NodeType:
        raise NotImplementedError

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class ChargingStation(Node):
    def __init__(self, x, y, z, identifier, wt=0, ct=0):
        super().__init__(x, y, z, wt, ct)
        self._identifier = identifier

    @property
    def node_type(self) -> NodeType:
        return NodeType.ChargingStation

    @property
    def identifier(self):
        return self._identifier

    def __repr__(self):
        return f"CS{super().__repr__()} [{self.identifier}]"


class Waypoint(Node):
    def __init__(self, x, y, z, strided=False, identifier=None):
        super().__init__(x, y, z, 0, 0)
        self.strided = strided
        self._identifier = identifier

    @property
    def node_type(self) -> NodeType:
        return NodeType.Waypoint

    @property
    def identifier(self):
        return self._identifier

    def __repr__(self):
        if self.identifier is not None:
            return f"WP{super().__repr__()} [{self.identifier}]"
        else:
            return f"WP{super().__repr__()}"


class AuxWaypoint(Node):
    def __init__(self, x, y, z):
        super().__init__(x, y, z, 0, 0)

    @property
    def node_type(self) -> NodeType:
        return NodeType.AuxWaypoint

    @property
    def identifier(self):
        return None

    def __repr__(self):
        return f"AUX{super().__repr__()}"


def node_serializer(obj: Node, *args, **kwargs):
    return dict(
        x=obj.x,
        y=obj.y,
        z=obj.z,
        wt=obj.wt,
        ct=obj.ct,
        id=int(obj.identifier) if obj.identifier is not None else None,
        type=obj.node_type.value,
    )


jsons.set_serializer(node_serializer, Node)
