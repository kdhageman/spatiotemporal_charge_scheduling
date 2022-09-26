import logging
from datetime import datetime
from functools import lru_cache
from typing import List

import numpy as np
import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import SumExpression

from simulate.parameters import Parameters
from util.scenario import Scenario


class MultiUavModel(pyo.ConcreteModel):
    def __init__(self, sc: Scenario, params: Parameters, anchor_offsets=List[int]):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # extract from function parameters
        self.N_d = sc.N_d
        self.N_s = sc.N_s
        self.N_w = sc.N_w

        self.epsilon = params.epsilon
        self.B_start = params.B_start
        self.B_min = params.B_min
        self.B_end = params.B_end
        self.B_max = params.B_max
        self.r_charge = params.r_charge
        self.r_deplete = params.r_deplete
        self.v = params.v
        self.W_zero_min = params.W_zero_min
        self.remaining_distances = params.remaining_distances

        self.positions_S = sc.positions_S
        self.positions_w = sc.positions_w
        self.D_N = sc.D_N
        self.D_W = sc.D_W
        self.C_max = (self.B_max - self.B_min) / self.r_charge

        self.W_max = []
        for d in range(self.N_d):
            wmax = 0
            for d_prime in range(self.N_d):
                if d_prime == d:
                    continue
                for w_s in range(self.N_w):
                    wmax += max(self.D_N[d_prime, :-1, w_s]) / self.v[d_prime]
                wmax += self.N_w * (self.C_max[d] + self.epsilon)
            self.W_max.append(wmax)
        self.M = sum(self.W_max)

        ### MODEL DEFINITION
        # INDICES
        self.d = pyo.RangeSet(0, self.N_d - 1)
        self.s = pyo.RangeSet(0, self.N_s - 1)
        self.w = pyo.RangeSet(0, self.N_w)
        self.w_s = pyo.RangeSet(0, self.N_w - 1)
        self.w_d = pyo.RangeSet(1, self.N_w)
        self.n = pyo.RangeSet(0, self.N_s)
        self.info("created control indices")

        # CONTROL VARIABLES
        self.P = pyo.Var(self.d, self.n, self.w_s, domain=pyo.Binary)
        self.C = pyo.Var(self.d, self.w_s, domain=pyo.NonNegativeReals)
        self.W = pyo.Var(self.d, self.w_s, domain=pyo.NonNegativeReals)
        self.Gamma = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        self.info("created control variables")

        for d in self.d:
            anchor_ids = []
            anchor_id = anchor_offsets[d] % params.sigma
            while anchor_id <= self.N_w:
                anchor_ids.append(anchor_id)
                anchor_id += params.sigma
            self.info(f"anchor ids for UAV [{d}]: {anchor_ids}")

            for w_s in self.w_s:
                if w_s not in anchor_ids:
                    for s in self.s:
                        self.P[d, s, w_s].fix(0)
                    self.P[d, self.N_s, w_s].fix(1)
                    self.C[d, w_s].fix(0)
                    self.W[d, w_s].fix(0)
        self.info("fixed control variable values")

        # STATE VARIABLES
        self.theta = pyo.Var(self.d, self.d, self.w_s, self.w_s, self.s, domain=pyo.Binary)  # used for linearazing the Theta state variable
        self.alpha = pyo.Var()  # used for linearizing the minmax objective
        self.info("created state variables")

        # STATE COMPUTATION CONSTRAINTS

        # CONTROL VARIABLE LIMIT CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: sum(m.P[d, n, w_s] for n in self.n) == 1
        )
        self.info(f"finished initializing 'path_constraint' ({len(self.path_constraint):,})")

        self.C_ulim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.C[d, w_s] <= (1 - m.P[d, m.N_s, w_s]) * m.C_max[d]
        )
        self.info(f"finished initializing 'C_ulim' ({len(self.C_ulim):,})")

        def W_ulim_rule(m, d, w_s):
            try:
                return m.W[d, w_s] <= (1 - m.P[d, m.N_s, w_s]) * sum([self.W_max[d_prime] for d_prime in m.d if d_prime != d])
            except Exception as e:
                raise e

        self.W_ulim = pyo.Constraint(self.d, self.w_s, rule=W_ulim_rule)
        self.info(f"finished initializing 'W_ulim' ({len(self.W_ulim):,})")

        self.W_llim = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.W[d, 0] + sum(m.P[d, s, 0] * m.D_N[d, s, 0] for s in m.s) / m.v[d] >= sum(m.P[d, s, 0] * m.W_zero_min[d, s] for s in m.s)
        )
        self.info(f"finished initializing 'W_llim' ({len(self.W_llim):,})")

        def b_star_llim_rule(m, d, w_d):
            if w_d == m.N_w:
                lim = m.B_end[d]
            else:
                lim = m.B_min[d]
            return m.b_star(d, w_d) >= lim

        self.b_star_llim = pyo.Constraint(self.d, self.w_d, rule=b_star_llim_rule)
        self.info(f"finished initializing 'b_star_llim' ({len(self.b_star_llim):,})")

        def b_min_llim_rule(m, d, w_s):
            b_min = m.b_min(d, w_s)
            if type(b_min) != SumExpression:
                return pyo.Constraint.Skip
            return b_min >= m.B_min[d]

        self.b_min_llim = pyo.Constraint(self.d, self.w_s, rule=b_min_llim_rule)
        self.info(f"finished initializing 'b_min_llim' ({len(self.b_min_llim):,})")

        self.b_plus_ulim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus(d, w_s) <= m.B_max[d]
        )
        self.info(f"finished initializing 'b_plus_ulim' ({len(self.b_plus_ulim):,})")

        # THETA LINEARIZATION
        def theta_1_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.theta[d, d_prime, w_s, w_s_prime, s] <= m.P[d, s, w_s]

        self.theta_1 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, self.s, rule=theta_1_rule)
        self.info(f"finished initializing 'theta_1' ({len(self.theta_1):,})")

        def theta_2_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.theta[d, d_prime, w_s, w_s_prime, s] <= m.P[d_prime, s, w_s_prime]

        self.theta_2 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, self.s, rule=theta_2_rule)
        self.info(f"finished initializing 'theta_2' ({len(self.theta_2):,})")

        def theta_3_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.theta[d, d_prime, w_s, w_s_prime, s] >= m.P[d, s, w_s] + m.P[d_prime, s, w_s_prime] - 1

        self.theta_3 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, self.s, rule=theta_3_rule)
        self.info(f"finished initializing 'theta_3' ({len(self.theta_3):,})")

        # WINDOW OVERLAP CONSTRAINTS
        def window_i_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_e(d, w_s) <= m.T_s(d_prime, w_s_prime) - m.epsilon + m.M * (1 + m.Gamma[d, d_prime, w_s, w_s_prime] - m.Theta(d, d_prime, w_s, w_s_prime))

        def window_ii_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_s(d_prime, w_s_prime) <= m.T_s(d, w_s) - m.epsilon + m.M * (2 - m.Gamma[d, d_prime, w_s, w_s_prime] - m.Theta(d, d_prime, w_s, w_s_prime))

        self.window_i = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_i_rule)
        self.info(f"finished initializing 'window_i' ({len(self.window_i):,})")
        self.window_ii = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_ii_rule)
        self.info(f"finished initializing 'window_ii' ({len(self.window_ii):,})")

        # OBJECTIVE
        self.alpha_min = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.alpha >= m.E(d)
        )
        self.info(f"finished initializing 'alpha_min' ({len(self.alpha_min):,})")

        self.execution_time = pyo.Objective(
            expr=self.alpha,
            sense=pyo.minimize,
        )
        self.info("finished initializing 'execution_time'")

    def debug(self, msg):
        self.logger.debug(self._craft_msg(msg))

    def info(self, msg):
        self.logger.debug(self._craft_msg(msg))

    def _craft_msg(self, msg):
        return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

    @lru_cache(maxsize=None)
    def b_star(self, d, w):
        """
        Calculate the battery at arrival at waypoint 'w' for drone 'd'
        """
        if w == 0:
            # base case
            res = self.B_start[d]
        else:
            res = self.b_plus(d, w - 1) - self.r_deplete[d] / self.v[d] * sum(self.P[d, n, w - 1] * self.D_W[d, n, w - 1] for n in self.n)
        return res

    @lru_cache(maxsize=None)
    def b_min(self, d, w_s):
        """
        Calculate the battery of drone 'd' when arriving at the next path node after waypoint 'w_s'
        """
        return self.b_star(d, w_s) - self.r_deplete[d] / self.v[d] * sum(self.P[d, n, w_s] * self.D_N[d, n, w_s] for n in self.n)

    @lru_cache(maxsize=None)
    def b_plus(self, d, w_s):
        """
        Calculate the battery of drone 'd' after charging after waypoint 'w_s'
        """
        return self.b_min(d, w_s) + self.r_charge[d] * self.C[d, w_s]

    @lru_cache(maxsize=None)
    def T_s(self, d, w_s):
        return sum(
            self.C[d, w_p] + self.W[d, w_p] + self.t(d, w_p) for w_p in range(w_s)) + sum(
            self.P[d, n, w_s] * self.D_N[d, n, w_s] for n in self.n) / self.v[d] + self.W[d, w_s]

    @lru_cache(maxsize=None)
    def Theta(self, d, d_prime, w_s, w_s_prime):
        return sum(self.theta[d, d_prime, w_s, w_s_prime, s] for s in self.s)

    @lru_cache(maxsize=None)
    def T_e(self, d, w_s):
        return self.T_s(d, w_s) + self.C[d, w_s]

    @lru_cache(maxsize=None)
    def E(self, d):
        return sum(self.C[d, w_s] + self.W[d, w_s] + self.t(d, w_s) for w_s in self.w_s) + self.lambda_move(d) + self.lambda_charge(d)

    def total_waiting_time(self, d):
        """
        Return the total waiting time for d in the calculated schedule
        """
        return sum(self.W[d, w_s] for w_s in self.w_s)

    def total_charging_time(self, d):
        """
        Return the total charging time for d in the calculated schedule
        """
        return sum(self.C[d, w_s] for w_s in self.w_s)

    def total_moving_time(self, d):
        return sum(self.t(d, w_s) for w_s in self.w_s)

    def remaining_move_time(self, d):
        return self.remaining_distances[d] / self.v[d]

    def remaining_depletion(self, d):
        return self.remaining_move_time(d) * self.r_deplete[d]

    def lambda_move(self, d):
        return self.remaining_distances[d] / self.v[d]

    def lambda_charge(self, d):
        # TODO: add max statement
        return (self.erd(d) - self.oc(d)) / self.r_charge[d]

    def t(self, d, w_s):
        return sum(self.P[d, n, w_s] * (self.D_N[d, n, w_s] + self.D_W[d, n, w_s]) for n in self.n) / self.v[d]

    def oc(self, d):
        """
        Return the overcharge for drone 'd'
        """
        return self.b_star(d, self.N_w) - self.B_end[d]

    def erd(self, d):
        """
        Return the expected remaining depletion for drone 'd'
        """
        return self.remaining_move_time(d) * self.r_deplete[d]

    @property
    def P_np(self):
        """
        Returns the chosen path decision variable (P) as a numpy array
        """
        return np.reshape(self.P[:, :, :](), (self.N_d, self.N_s + 1, self.N_w)).round()

    @property
    def C_np(self):
        """
        Return the charging time decision variable (C) as a numpy array
        """
        return np.reshape(self.C[:, :](), (self.N_d, self.N_w))

    @property
    def W_np(self):
        """
        Return the waiting time decision variable (W) as a numpy array
        """
        return np.reshape(self.W[:, :](), (self.N_d, self.N_w))
