from itertools import product

import pyomo.environ as pyo

from util.second.parameters import get_T_N, get_T_W

_epsilon = 0.001


class Model(pyo.ConcreteModel):
    def __init__(self, indices={}, parameters={}):
        super().__init__()
        # extract from function parameters
        N_d = indices["N_d"]
        N_s = indices["N_s"]
        N_w = indices["N_w"]

        B_start = parameters["B_start"]
        B_min = parameters["B_min"]
        B_max = parameters["B_max"]

        r_charge = parameters['r_charge']
        r_deplete = parameters['r_deplete']

        D_max = (B_max - B_min) / r_charge

        positions_S = parameters['positions_S']
        positions_w = parameters['positions_w']

        T_N = get_T_N(N_d, N_s, N_w - 1, positions_S, positions_w)
        T_W = get_T_W(N_d, N_s, N_w - 1, positions_S, positions_w)

        # MODEL DEFINITION
        self.d = pyo.RangeSet(0, N_d - 1)
        self.s = pyo.RangeSet(0, N_s - 1)
        self.w = pyo.RangeSet(0, N_w - 1)
        self.w_s = pyo.RangeSet(0, N_w - 2)
        self.w_d = pyo.RangeSet(1, N_w - 1)
        self.n = pyo.RangeSet(0, N_s)
        self.T_N = T_N
        self.T_W = T_W

        # VARIABLES
        # control variables
        self.P = pyo.Var(self.d, self.n, self.w_s, domain=pyo.Binary)
        self.D = pyo.Var(self.d, self.w_s, domain=pyo.NonNegativeReals)
        self.F = pyo.Var(self.d, self.w_s, bounds=(0, sum(D_max)))
        self.alpha = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)

        # state variables
        self.b_arr = pyo.Var(self.d, self.w)
        self.b_min = pyo.Var(self.d, self.w_s)
        self.b_plus = pyo.Var(self.d, self.w_s)
        self.Z_s = pyo.Var(self.d, self.w_s)
        self.Z_e = pyo.Var(self.d, self.w_s)
        self.C = pyo.Var(self.d, self.d, self.w_s, self.w_s)
        self.C_prime = pyo.Var(self.d, self.d, self.w_s, self.w_s)

        # CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: sum(m.P[d, n, w_s] for n in self.n) == 1
        )

        # battery constraints
        self.b_arr_start = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b_arr[d, 0] == B_start[d]
        )

        self.b_min_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_min[d, w_s] == m.b_arr[d, w_s] - r_deplete[d] * sum(
                m.P[d, n, w_s] * T_N[d, n, w_s] for n in m.n)
        )

        self.b_plus_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus[d, w_s] == m.b_min[d, w_s] + r_charge[d] * m.D[d, w_s]
        )

        self.b_arr_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_arr[d, w_s + 1] == m.b_plus[d, w_s] - r_deplete[d] * sum(
                m.P[d, n, w_s] * T_W[d, n, w_s] for n in m.n)
        )

        # lower and upper bounds of variables values
        self.b_arr_llim = pyo.Constraint(
            self.d,
            self.w,
            rule=lambda m, d, w: m.b_arr[d, w] >= B_min
        )

        self.b_min_llim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_min[d, w_s] >= B_min
        )
        self.b_plus_ulim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus[d, w_s] <= B_max
        )

        self.D_lim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.D[d, w_s] <= (1 - m.P[d, N_s, w_s]) * D_max[d]
        )

        self.Z_s_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.Z_s[d, w_s] == sum(
                m.D[d, w_p] + m.F[d, w_p] + m.t(d, w_p) for w_p in range(w_s)) + sum(
                m.P[d, n, w_s] * T_W[d, n, w_s] for n in m.n) + m.F[d, w_s]
        )

        self.Z_e_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.Z_e[d, w_s] == m.Z_s[d, w_s] + m.D[d, w_s]
        )

        self.C_calc = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            rule=lambda m, d, d_prime, w_s, w_s_prime: m.C[d, d_prime, w_s, w_s_prime] == m.alpha[
                d, d_prime, w_s, w_s_prime] * (m.Z_s[d_prime, w_s_prime] - m.Z_s[d, w_s]) + (
                                                               1 - m.alpha[d, d_prime, w_s, w_s_prime]) * (
                                                                   m.Z_e[d, w_s] - m.Z_s[d_prime, w_s_prime])
        )

        self.C_prime_calc = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            rule=lambda m, d, d_prime, w_s, w_s_prime: m.C_prime[d, d_prime, w_s, w_s_prime] == m.C[
                d, d_prime, w_s, w_s_prime] * sum(m.P[d, s, w_s] * m.P[d_prime, s, w_s_prime] for s in m.s)
        )

        def C_prime_lim_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.C_prime[d, d_prime, w_s, w_s_prime] <= 0

        self.C_prime_lim = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=C_prime_lim_rule)

        # OBJECTIVE
        def E(d):
            return sum(self.D[d, w_s] + self.F[d, w_s] + self.t(d, w_s) for w_s in self.w_s)

        self.execution_time = pyo.Objective(
            expr=sum(E(d) for d in self.d),
            sense=pyo.minimize,
        )

    def t(self, d, w_s):
        return sum(self.P[d, n, w_s] * (self.T_N[d, n, w_s] + self.T_W[d, n, w_s]) for n in self.n)
