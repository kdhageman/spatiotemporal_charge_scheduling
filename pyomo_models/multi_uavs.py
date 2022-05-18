import pyomo.environ as pyo

from pyomo_models.base import BaseModel


class MultiUavModel(BaseModel):
    def __init__(self, scenario, parameters):
        self.epsilon = parameters.get("epsilon", 0.01)
        super().__init__(scenario, parameters)

        self.M_1 = self.W_max * self.N_d * self.N_w_s
        self.M_2 = self.M_1 + self.W_max

        # VARIABLES
        self.o = pyo.Var(self.d, self.d, self.w_s, self.w_s, self.s, domain=pyo.Binary)
        self.y = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)

        # CONSTRAINTS

        def o_1_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] <= m.P[d, s, w_s]

        self.o_1 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=o_1_rule
        )

        def o_2_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] <= m.P[d_prime, s, w_s_prime]

        self.o_2 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=o_2_rule
        )

        def o_3_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] >= m.P[d, s, w_s] + m.P[d_prime, s, w_s_prime] - 1

        self.o_3 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=o_3_rule
        )

        # window constraints
        def window_i_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_e(d, w_s) <= m.T_s(d_prime, w_s_prime) - self.epsilon + self.M_1 * (
                    1 + m.y[d, d_prime, w_s, w_s_prime] - m.O(d, d_prime, w_s, w_s_prime))

        def window_ii_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_s(d_prime, w_s_prime) <= m.T_s(d, w_s) - self.epsilon + self.M_2 * (
                    2 - m.y[d, d_prime, w_s, w_s_prime] - m.O(d, d_prime, w_s, w_s_prime))

        self.window_i = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_i_rule)
        self.window_ii = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_ii_rule)

    @property
    def W_max(self):
        return sum([self.C_max[d] + self.epsilon for d in self.d])

    def T_s(self, d, w_s):
        return sum(
            self.C[d, w_p] + self.W[d, w_p] + self.t(d, w_p) for w_p in range(w_s)) + sum(
            self.P[d, n, w_s] * self.D_N[d, n, w_s] for n in self.n) / self.v[d] + self.W[d, w_s]

    def T_e(self, d, w_s):
        return self.T_s(d, w_s) + self.C[d, w_s]

    def O(self, d, d_prime, w_s, w_s_prime):
        return sum(self.o[d, d_prime, w_s, w_s_prime, s] for s in self.s)
