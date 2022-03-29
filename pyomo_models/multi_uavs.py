import pyomo.environ as pyo

from pyomo_models.base import BaseModel


class MultiUavModel(BaseModel):
    def __init__(self, scenario, parameters):
        self.epsilon = parameters.get("epsilon", 0.01)
        super().__init__(scenario, parameters)

        # VARIABLES
        # control variables
        self.beta = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary, initialize=0)

        # state variables
        self.Z_s = pyo.Var(self.d, self.w_s)
        self.Z_e = pyo.Var(self.d, self.w_s)
        self.Y = pyo.Var(self.d, self.d, self.w_s, self.w_s)
        self.Y_prime = pyo.Var(self.d, self.d, self.w_s, self.w_s)

        # CONSTRAINTS
        self.Z_s_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.Z_s[d, w_s] == sum(
                m.C[d, w_p] + m.W[d, w_p] + m.t(d, w_p) for w_p in range(w_s)) + sum(
                m.P[d, n, w_s] * self.T_N[d, n, w_s] for n in m.n) / self.v[d] + m.W[d, w_s]
        )

        self.Z_e_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.Z_e[d, w_s] == m.Z_s[d, w_s] + m.C[d, w_s]
        )

        self.Y_calc = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            rule=lambda m, d, d_prime, w_s, w_s_prime: m.Y[d, d_prime, w_s, w_s_prime] == m.beta[
                d, d_prime, w_s, w_s_prime] * (m.Z_s[d_prime, w_s_prime] - m.Z_s[d, w_s] + self.epsilon) + (
                                                               1 - m.beta[d, d_prime, w_s, w_s_prime]) * (
                                                               m.Z_e[d, w_s] - m.Z_s[d_prime, w_s_prime])
        )

        self.Y_prime_calc = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            rule=lambda m, d, d_prime, w_s, w_s_prime: m.Y_prime[d, d_prime, w_s, w_s_prime] == m.Y[
                d, d_prime, w_s, w_s_prime] * sum(m.P[d, s, w_s] * m.P[d_prime, s, w_s_prime] for s in m.s)
        )

        def Y_prime_lim_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.Y_prime[d, d_prime, w_s, w_s_prime] <= 0

        self.Y_prime_lim = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=Y_prime_lim_rule)

    def get_W_max(self):
        return sum([max(self.C_max[d], self.epsilon) for d in self.d])
