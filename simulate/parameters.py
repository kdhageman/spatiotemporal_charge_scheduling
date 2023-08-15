import copy
from dataclasses import dataclass, field, asdict
from typing import List

import jsons
import numpy as np


@dataclass
class SchedulingParameters:
    """
    v: velocity
    r_charge: rate of charge
    r_deplete: rate of depletion
    B_start: start battery capacity
    B_min: minimum battery capacity
    epsilon: distance between time charging windows
    omega: minimum time after charging is allowed at a station
    rho: remaining distance after the current scheduling task
    W_hat: waypoint horizon size
    sigma: anchor interval
    pi: rescheduling interval
    epsilon: separation in seconds of two subsequent charging windows
    int_feas_tol: feasibility tolerance for integer problems to solve in Gurobi
    time_limit: time limit for Gurobi to solve a given problem
    """
    v: np.ndarray
    r_charge: np.ndarray
    r_deplete: np.ndarray
    B_start: np.ndarray
    B_min: np.ndarray
    B_max: np.ndarray
    omega: np.ndarray
    rho: np.ndarray
    W_hat: int
    sigma: int
    int_feas_tol: float = 1e-9
    time_limit: float = 60
    epsilon: float = 1

    @classmethod
    def from_raw(cls,
                 v: List[float],
                 r_charge: List[float],
                 r_deplete: List[float],
                 B_start: List[float],
                 B_min: List[float],
                 B_max: List[float],
                 omega: List[List[float]] = None,
                 rho: List[float] = None,
                 W_hat: int = 1,
                 sigma: int = 1,
                 int_feas_tol: float = 1e-9,
                 time_limit: float = 60,
                 epsilon: float = 1,
                 ):
        if rho is None:
            rho = [0] * len(v)
        return SchedulingParameters(
            np.array(v),
            np.array(r_charge),
            np.array(r_deplete),
            np.array(B_start),
            np.array(B_min),
            np.array(B_max),
            np.array(omega),
            np.array(rho),
            W_hat,
            sigma,
            int_feas_tol,
            time_limit,
            epsilon,
        )

    def with_remaining_distances(self, rho):
        self.rho = rho
        return self

    def with_start_battery(self, B_start):
        self.B_start = B_start
        return self

    def with_min_battery(self, B_min):
        self.B_min = B_min
        return self

    def with_omega(self, omega):
        self.omega = omega
        return self


@dataclass
class ScenarioParams:
    """
    dists_to: distance between waypoint and next path node
    dists_from: distance between next path node and hext waypoint
    """
    dists_to: np.ndarray
    dists_from: np.ndarray

    @classmethod
    def from_raw(cls, dists_to: List[List[float]], dists_from: List[List[float]]):
        return ScenarioParams(np.array(dists_to), np.array(dists_from))


@dataclass
class SimulationParameters:
    """
    plot_delta: granularity of simulation time to visualise data as
    delta_t: time interval for simulation
    """
    plot_delta: float = 0
    delta_t: float = 0.1
    draw_scheduling: bool = True
    draw_battery_profile_greyscale: bool = False


def sched_parameters_serializer(obj: SchedulingParameters, *args, **kwargs):
    res = asdict(obj)
    for k, v in res.items():
        if type(v) == np.ndarray:
            if not v.shape:
                res[k] = []
            else:
                res[k] = v.astype(float).tolist()
    return res


def sim_parameters_serializer(obj: SimulationParameters, *args, **kwargs):
    res = asdict(obj)
    for k, v in res.items():
        if type(v) == np.ndarray:
            if not v.shape:
                res[k] = []
            else:
                res[k] = v.astype(float).tolist()
    return res


jsons.set_serializer(sched_parameters_serializer, SchedulingParameters)
jsons.set_serializer(sim_parameters_serializer, SimulationParameters)
# jsons.set_serializer(parameters_serializer, Parameters)
