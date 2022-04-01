import pyomo.environ as pyo

from pyomo_models.base import BaseModel


class MultiUavModel(BaseModel):
    def __init__(self, scenario, parameters):
        self.epsilon = parameters.get("epsilon", 0.01)
        super().__init__(scenario, parameters)

        self.M_1 = int(1e5) # TODO: redefine
        self.M_2 = int(1e5) # TODO: redefine

        # VARIABLES
        self.a = pyo.Var(self.d, self.d, self.w_s, self.w_s, self.s, domain=pyo.Binary)
        self.A = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        self.y = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        # self.N_i = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        # self.N_ii = pyo.Var(self.d, self.d, self.w_s, self.w_s, domain=pyo.Binary)
        self.Z_s = pyo.Var(self.d, self.w_s)
        self.Z_e = pyo.Var(self.d, self.w_s)

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

        def a_1_rule(m, d, d_prime, w_s, w_s_prime, s):
            return m.a[d, d_prime, w_s, w_s_prime, s] <= m.P[d, s, w_s]

        self.a_1 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_1_rule
        )

        def a_2_rule(m, d, d_prime, w_s, w_s_prime, s):
            return m.a[d, d_prime, w_s, w_s_prime, s] <= m.P[d_prime, s, w_s_prime]

        self.a_2 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_2_rule
        )

        def a_3_rule(m, d, d_prime, w_s, w_s_prime, s):
            return m.a[d, d_prime, w_s, w_s_prime, s] >= m.P[d, s, w_s] + m.P[d_prime, s, w_s_prime] - 1

        self.a_3 = pyo.Constraint(
            self.d,
            self.d,
            self.w_s,
            self.w_s,
            self.s,
            rule=a_3_rule
        )

        def A_calc_rule(m, d, d_prime, w_s, w_s_prime):
            return m.A[d, d_prime, w_s, w_s_prime] == sum(m.a[d, d_prime, w_s, w_s_prime, s] for s in m.s)

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
            return m.Z_e[d, w_s] <= m.Z_s[d_prime, w_s_prime] - self.epsilon + self.M_1 * (m.y[d, d_prime, w_s, w_s_prime] + 1 - m.A[d, d_prime, w_s, w_s_prime])

        def window_ii_rule(m, d, d_prime, w_s, w_s_prime):
            if d == d_prime:
                return pyo.Constraint.Skip
            return m.Z_s[d_prime, w_s_prime] <= m.Z_s[d, w_s] - self.epsilon + self.M_2 * (2 - m.y[d, d_prime, w_s, w_s_prime] - m.A[d, d_prime, w_s, w_s_prime])

        self.window_i = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_i_rule)
        self.window_ii = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_ii_rule)


        # -------- BEGIN N_i rules --------
        # def N_i_1_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_i[d, d_prime, w_s, w_s_prime] <= m.y[d, d_prime, w_s, w_s_prime]
        #
        # def N_i_2_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_i[d, d_prime, w_s, w_s_prime] <= 1 - m.A[d, d_prime, w_s, w_s_prime]
        #
        # def N_i_3_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_i[d, d_prime, w_s, w_s_prime] >= m.y[d, d_prime, w_s, w_s_prime] - m.A[
        #         d, d_prime, w_s, w_s_prime]
        #
        # self.N_i_1 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_i_1_rule)
        # self.N_i_2 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_i_2_rule)
        # self.N_i_3 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_i_3_rule)
        #
        # # N_ii rules
        # def N_ii_1_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_ii[d, d_prime, w_s, w_s_prime] <= 1 - m.y[d, d_prime, w_s, w_s_prime]
        #
        # def N_ii_2_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_ii[d, d_prime, w_s, w_s_prime] <= 1 - m.A[d, d_prime, w_s, w_s_prime]
        #
        # def N_ii_3_rule(m, d, d_prime, w_s, w_s_prime):
        #     return m.N_ii[d, d_prime, w_s, w_s_prime] >= 1 - m.y[d, d_prime, w_s, w_s_prime] - m.A[
        #         d, d_prime, w_s, w_s_prime]
        #
        # self.N_ii_1 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_ii_1_rule)
        # self.N_ii_2 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_ii_2_rule)
        # self.N_ii_3 = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=N_ii_3_rule)

        # window constraints
        # def window_i_rule(m, d, d_prime, w_s, w_s_prime):
        #     if d == d_prime:
        #         return pyo.Constraint.Skip
        #     return m.Z_e[d, w_s] <= m.Z_s[d_prime, w_s_prime] + self.M_1 * m.N_i[d, d_prime, w_s, w_s_prime]

        # def window_ii_rule(m, d, d_prime, w_s, w_s_prime):
        #     if d == d_prime:
        #         return pyo.Constraint.Skip
        #     return m.Z_s[d_prime, w_s_prime] <= m.Z_s[d, w_s] - self.epsilon + self.M_2 * m.N_ii[d, d_prime, w_s, w_s_prime]

        # self.window_i = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_i_rule)
        # self.window_ii = pyo.Constraint(self.d, self.d, self.w_s, self.w_s, rule=window_ii_rule)
        # -------- END N_i rules --------

    @property
    def W_max(self):
        return sum([max(self.C_max[d], self.epsilon) for d in self.d])
