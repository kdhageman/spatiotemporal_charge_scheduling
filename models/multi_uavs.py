import pyomo.environ as pyo

from models.base import BaseModel


class MultiUavModel(BaseModel):
    def __init__(self, indices={}, parameters={}):
        super().__init__(indices, parameters)

        # VARIABLES
        # control variables
        self.F = pyo.Var(self.d, self.w_s, bounds=(0, sum(self.D_max)))
        self.alpha = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)

        # state variables
        self.Z_s = pyo.Var(self.d, self.w_s)
        self.Z_e = pyo.Var(self.d, self.w_s)
        self.C = pyo.Var(self.d, self.d, self.w_s, self.w_s)
        self.C_prime = pyo.Var(self.d, self.d, self.w_s, self.w_s)

        # CONSTRAINTS
        self.Z_s_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.Z_s[d, w_s] == sum(
                m.D[d, w_p] + m.F[d, w_p] + m.t(d, w_p) for w_p in range(w_s)) + sum(
                m.P[d, n, w_s] * self.T_W[d, n, w_s] for n in m.n) + m.F[d, w_s]
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
