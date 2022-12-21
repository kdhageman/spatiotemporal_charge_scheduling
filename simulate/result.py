from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Type

import jsons

from simulate.event import Event
from simulate.node import Node
from simulate.parameters import SchedulingParameters
from simulate.scheduling import Scheduler
from util.scenario import Scenario

@dataclass
class SimResult:
    """
    Stores the status of the simulation result
    """
    sched_params: SchedulingParameters
    scenario: Scenario
    events: List[Event]
    solve_times: List[float]
    execution_time: float
    time_spent: dict
    schedules: List[Tuple[int, List[Node], Scenario]]
    nr_visited_waypoints: List[int]
    occupancy: Dict[int, List[Dict[str, float]]]
    scheduler_cls: Type[Scheduler]

    @property
    def scheduler(self):
        return self.scheduler_cls.__class__.__name__.lower()


def simresult_serializer(obj: SimResult, *args, **kwargs):
    res = dict(
        sched_params=jsons.dump(obj.sched_params),
        scenario=jsons.dump(obj.scenario),
        event=jsons.dump(obj.events),
        solve_times=jsons.dump(obj.solve_times),
        execution_time=jsons.dump(obj.execution_time),
        time_spend=jsons.dump(obj.time_spent),
        schedules=jsons.dump(obj.schedules),
        nr_visited_waypoints=jsons.dump(obj.nr_visited_waypoints),
        occupancy=jsons.dump(obj.occupancy),
        scheduler=jsons.dump(obj.scheduler),
    )
    return res


jsons.set_serializer(simresult_serializer, SimResult)
