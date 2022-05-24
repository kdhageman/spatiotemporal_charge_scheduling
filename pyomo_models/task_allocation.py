from itertools import product

import numpy as np
import pyomo.environ as pyo
import yaml
from yaml import Loader

from util.distance import dist3


class Scenario:
    def __init__(self, doc):
        self.positions_s = []
        for pos in doc.get('charging_stations', []):
            x, y, z = [float(v) for v in pos.split(",")]
            self.positions_s.append((x, y, z))

        self.positions_w = []
        for pos in doc.get('waypoints', []):
            x, y, z = [float(v) for v in pos.split(",")]
            self.positions_w.append((x, y, z))

        self.positions_start = []
        for pos in doc.get("start_positions", []):
            x, y, z = [float(v) for v in pos.split(",")]
            self.positions_start.append((x, y, z))

        self.N_s = len(self.positions_s)
        self.N_w = len(self.positions_w)
        self.N_n = self.N_w + self.N_s
        self.N_d = len(self.positions_start)

        # calculate distance matrices
        self.D_start = []
        for pos_start in self.positions_start:
            row = []
            for pos_n in self.positions_w + self.positions_s:
                d = dist3(pos_start, pos_n)
                row.append(d)
            self.D_start.append(row)
        self.D_start = np.array(self.D_start)

        self.D = []
        for pos_n in self.positions_w + self.positions_s:
            row = []
            for pos_n_prime in self.positions_w + self.positions_s:
                d = dist3(pos_n, pos_n_prime)
                row.append(d)
            self.D.append(row)
        self.D = np.array(self.D)

    @classmethod
    def from_file(cls, fname):
        with open(fname, 'r') as f:
            doc = yaml.load(f, Loader=Loader)
        return Scenario(doc)


class Parameters:
    class Parameters:
        def __init__(self, N_t: int, v: np.array, r_charge: float, r_deplete_fly: float, r_deplete_inspect: float,
                     i: np.array, B_start: np.array, B_min: float, B_max: float, R: np.array):
            self.N_t = N_t
            self.v = v
            self.r_charge = r_charge
            self.r_deplete_fly = r_deplete_fly
            self.r_deplete_inspect = r_deplete_inspect
            self.i = i
            self.B_start = B_start
            self.B_min = B_min
            self.B_max = B_max
            self.R = R


class TaskAllocationModel(pyo.ConcreteModel):
    def __init__(self, scenario, parameters):
        super().__init__()

        self.N_d = parameters['N_d']
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w
        self.N_n = scenario.N_n
        self.N_t = parameters['N_t']

        self.B_max = parameters['B_max']
        self.B_min = parameters['B_min']
        self.B_start = parameters['B_start']
        self.r_charge = parameters['r_charge']
        self.r_deplete_fly = parameters['r_deplete_fly']
        self.r_deplete_inspect = parameters['r_deplete_inspect']
        self.v = parameters['v']
        self.i = parameters['i']
        self.R = parameters['R']

        # MODEL DEFINITION
        self.d = pyo.RangeSet(0, self.N_d - 1)
        self.w = pyo.RangeSet(0, self.N_w - 1)
        self.s = pyo.RangeSet(self.N_w, self.N_n - 1)
        self.n = pyo.RangeSet(0, self.N_n - 1)
        self.t = pyo.RangeSet(0, self.N_t - 1)
        self.t_r = pyo.RangeSet(1, self.N_t - 1)

        self.D_start = scenario.D_start
        self.D = scenario.D

        # VARIABLES
        self.P = pyo.Var(self.d, self.n, self.t, domain=pyo.Binary)
        self.I = pyo.Var(self.d, self.t, domain=pyo.PositiveReals)
        self.b = pyo.Var(self.d, self.t)
        self.b_plus = pyo.Var(self.d, self.t_r)
        self.b_min = pyo.Var(self.d, self.t_r)
        self.alpha = pyo.Var(domain=pyo.PositiveReals)  # used for minmax

        # OBJECTIVE
        self.alpha_min = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.alpha >= m.inspection_time(d) + m.charging_time(d) + m.moving_time(d)
        )

        self.execution_time = pyo.Objective(
            expr=self.alpha,
            sense=pyo.minimize,
        )

        # PATH CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.t,
            rule=lambda m, d, t: sum(m.P[d, n, t] for n in m.n) == 1
        )

        # BATTERY CONSTRAINTS
        self.b_first_calc = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b[d, 0] == m.B_start[d] - m.r_deplete_fly[d] / m.v[d] * (
                sum(m.P[d, n, 1] * m.D_start[d, n] for n in m.n)) - m.r_deplete_inspect[d] * m.I[d, 1]
        )

        self.b_plus_calc = pyo.Constraint(
            self.d,
            self.t_r,
            rule=lambda m, d, t_r: m.b_plus[d, t_r] ==
                                   sum(m.P[d, s, t_r - 1] * (m.B_max - m.b[d, t_r - 1]) for s in m.s)
        )

        self.b_min_calc = pyo.Constraint(
            self.d,
            self.t_r,
            rule=lambda m, d, t_r: m.b_min[d, t_r] == m.r_deplete_fly[d] / m.v[d] * (
                sum(m.P[d, n, t_r - 1] * m.P[d, n_prime, t_r] * m.D[n, n_prime] for n, n_prime in product(m.n, m.n))) +
                                   m.r_deplete_inspect[d] * m.I[d, t_r]
        )

        self.b_calc = pyo.Constraint(
            self.d,
            self.t_r,
            rule=lambda m, d, t_r: m.b[d, t_r] == m.b[d, t_r - 1] + m.b_plus[d, t_r] - m.b_min[d, t_r]
        )

        self.b_req_minimum_val = pyo.Constraint(
            self.d,
            self.t,
            rule=lambda m, d, t: m.b[d, t] >= m.B_min
        )

        # INSPECTION CONSTRAINTS
        self.inspection_constraint = pyo.Constraint(
            self.w,
            rule=lambda m, w: sum(m.P[d, w, t] * m.I[d, t] * m.i[d] for d, t in product(m.d, m.t)) >= m.R[w]
        )

    def inspection_time(self, d):
        """
        Returns the inspection time for drone 'd'
        """
        return sum(self.I[d, t] for t in self.t)

    def charging_time(self, d):
        """
        Return the charging time for drone 'd'
        """
        return sum(self.P[d, s, t] * (self.B_max - self.b[d, t]) for s, t in product(self.s, self.t)) / self.r_charge[
            d]

    def moving_time(self, d):
        """
        Return the total moving time of drone 'd'
        """
        start_dist = sum(self.D_start[d, n] * self.P[d, n, 0] for n in self.n)
        remaining_dist = sum(self.P[d, n, t - 1] * self.P[d, n_prime, t] * self.D[n, n_prime] for t, n, n_prime in
                             product(self.t_r, self.n, self.n))
        return (start_dist + remaining_dist) / self.v[d]
