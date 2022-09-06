from typing import List, Tuple
from simulate.event import Event
from simulate.node import Node
from simulate.parameters import Parameters


class SimResult:
    """
    Stores the status of the simulation result
    """

    def __init__(self, params: Parameters, events: List[Event], solve_times: List[float], execution_time: float, time_spent: dict, schedules: List[Tuple[int, List[Node]]], nr_visited_waypoints: List[int]):
        self.params = params
        self.events = events
        self.solve_times = solve_times
        self.execution_time = execution_time
        self.time_spent = time_spent
        self.schedules = schedules
        self.nr_visited_waypoints = nr_visited_waypoints
