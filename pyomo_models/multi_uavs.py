import pyomo.environ as pyo

from pyomo_models.base import BaseModel


class MultiUavModel(BaseModel):
    def __init__(self, scenario, parameters):
        self.epsilon = parameters.get("epsilon", 0.01)
        super().__init__(scenario, parameters)

        self.M_1 = (self.W_max + max(self.C_max)) * self.N_w_s
        self.M_2 = (self.W_max + max(self.C_max)) * self.N_w_s

        # VARIABLES
        self.o = pyo.Var(self.d, self.d, self.w_s, self.w_s, self.s, domain=pyo.Binary)
        self.O = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        self.y = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        self.T_s = pyo.Var(self.d, self.w_s)
        self.T_e = pyo.Var(self.d, self.w_s)

        # CONSTRAINTS
        self.Z_s_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.T_s[d, w_s] == sum(
                m.C[d, w_p] + m.W[d, w_p] + m.t(d, w_p) for w_p in range(w_s)) + sum(
                m.P[d, n, w_s] * self.D_N[d, n, w_s] for n in m.n) / self.v[d] + m.W[d, w_s]
        )

        self.Z_e_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.T_e[d, w_s] == m.T_s[d, w_s] + m.C[d, w_s]
        )

        def a_1_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] <= m.P[d, s, w_s]

        self.a_1 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_1_rule
        )

        def a_2_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] <= m.P[d_prime, s, w_s_prime]

        self.a_2 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_2_rule
        )

        def a_3_rule(m, d, d_prime, w_s, w_s_prime, s):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.o[d, d_prime, w_s, w_s_prime, s] >= m.P[d, s, w_s] + m.P[d_prime, s, w_s_prime] - 1

        self.a_3 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_3_rule
        )

        def A_calc_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.O[d, d_prime, w_s, w_s_prime] == sum(m.o[d, d_prime, w_s, w_s_prime, s] for s in m.s)

        self.A_calc = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            rule=A_calc_rule
        )

        # window constraints
        def window_i_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_e[d, w_s] <= m.T_s[d_prime, w_s_prime] - self.epsilon + self.M_1 * (
                        1 + m.y[d, d_prime, w_s, w_s_prime] - m.O[d, d_prime, w_s, w_s_prime])

        def window_ii_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.T_s[d_prime, w_s_prime] <= m.T_s[d, w_s] - self.epsilon + self.M_2 * (
                        2 - m.y[d, d_prime, w_s, w_s_prime] - m.O[d, d_prime, w_s, w_s_prime])

        self.window_i = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_i_rule)
        self.window_ii = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_ii_rule)

    @property
    def W_max(self):
        return sum([max(self.C_max[d], self.epsilon) for d in self.d])
