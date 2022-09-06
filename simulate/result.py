from typing import List
from simulate.event import Event
from simulate.parameters import Parameters


class SimResult:
    """
    Stores the status of the simulation result
    """

    def __init__(self, params: Parameters, events: List[Event], solve_times: List[float], execution_time: float, time_spent: dict):
        self.params = params
        self.events = events
        self.solve_times = solve_times
        self.execution_time = execution_time
        self.time_spent = time_spent
