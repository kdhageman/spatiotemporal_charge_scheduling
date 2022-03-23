import numpy as np
import pyomo.environ as pyo
from matplotlib import pyplot as plt

from util import constants
from util.distance import dist


class BaseModel(pyo.ConcreteModel):
    def __init__(self, scenario, parameters):
        super().__init__()

        # extract from function parameters
        self.N_d = scenario.N_d
        self.N_s = scenario.N_s
        self.N_w = scenario.N_w
        self.N_w_s = scenario.N_w - 1

        self.B_start = parameters["B_start"]
        self.B_min = parameters["B_min"]
        self.B_max = parameters["B_max"]
        self.r_charge = parameters['r_charge']
        self.r_deplete = parameters['r_deplete']
        self.v = parameters['v']

        self.C_max = (self.B_max - self.B_min) / self.r_charge

        self.positions_S = scenario.positions_S
        self.positions_w = scenario.positions_w

        self.T_N = self._get_T_N()
        self.T_W = self._get_T_W()

        # MODEL DEFINITION
        self.d = pyo.RangeSet(0, self.N_d - 1)
        self.s = pyo.RangeSet(0, self.N_s - 1)
        self.w = pyo.RangeSet(0, self.N_w - 1)
        self.w_s = pyo.RangeSet(0, self.N_w - 2)
        self.w_d = pyo.RangeSet(1, self.N_w - 1)
        self.n = pyo.RangeSet(0, self.N_s)

        # VARIABLES
        self.P = pyo.Var(self.d, self.n, self.w_s, domain=pyo.Binary)
        self.C = pyo.Var(self.d, self.w_s, domain=pyo.NonNegativeReals)
        self.W = pyo.Var(self.d, self.w_s, bounds=(0, sum(self.C_max)))
        self.b_arr = pyo.Var(self.d, self.w)
        self.b_min = pyo.Var(self.d, self.w_s)
        self.b_plus = pyo.Var(self.d, self.w_s)

        # CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: sum(m.P[d, n, w_s] for n in self.n) == 1
        )

        # battery constraints
        self.b_arr_start = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b_arr[d, 0] == self.B_start[d]
        )

        self.b_min_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_min[d, w_s] == m.b_arr[d, w_s] - self.r_deplete[d] / self.v[d] * sum(
                m.P[d, n, w_s] * self.T_N[d, n, w_s] for n in m.n)
        )

        self.b_plus_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus[d, w_s] == m.b_min[d, w_s] + self.r_charge[d] * m.C[d, w_s]
        )

        self.b_arr_calc = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_arr[d, w_s + 1] == m.b_plus[d, w_s] - self.r_deplete[d] / self.v[d] * sum(
                m.P[d, n, w_s] * self.T_W[d, n, w_s] for n in m.n)
        )

        # lower and upper bounds of variables values
        self.b_arr_llim = pyo.Constraint(
            self.d,
            self.w,
            rule=lambda m, d, w: m.b_arr[d, w] >= self.B_min
        )

        self.b_min_llim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_min[d, w_s] >= self.B_min
        )
        self.b_plus_ulim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus[d, w_s] <= self.B_max
        )

        self.D_lim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.C[d, w_s] <= (1 - m.P[d, self.N_s, w_s]) * self.C_max[d]
        )

        # OBJECTIVE
        def E(d):
            return sum(self.C[d, w_s] + self.W[d, w_s] + self.t(d, w_s) for w_s in self.w_s)

        self.execution_time = pyo.Objective(
            expr=sum(E(d) for d in self.d),
            sense=pyo.minimize,
        )

    def plot(self, ax=None):
        if not ax:
            _, ax = plt.subplots()

        # draw charge station(s)
        for s in range(self.N_s):
            x_s = self.positions_S[s][0]
            y_s = self.positions_S[s][1]
            ax.scatter(x_s, y_s, marker='s', color='k', s=100)

            # label station
            x_text = x_s + 0.1
            y_text = y_s
            ax.text(x_text, y_text, f"$s_{{{s + 1}}}$", fontsize=15)

        # draw waypoints and lines for each drone
        for d in range(self.N_d):
            waypoints = self.positions_w[d]
            x = [w[0] for w in waypoints[:self.N_w]]
            y = [w[1] for w in waypoints[:self.N_w]]
            ax.scatter(x, y, marker='o', color=constants.W_COLORS[d], facecolor='white', s=70)

            # label waypoints
            for w in range(self.N_w):
                x_text = waypoints[w][0]
                y_text = waypoints[w][1] + 0.05
                ax.text(x_text, y_text, f"$w^{{{d + 1}}}_{{{w + 1}}}$", color=constants.W_COLORS[d], fontsize=15)

        # draw lines
        for d in self.d:
            for w_s in self.w_s:
                cur_waypoint = self.positions_w[d][w_s]
                next_waypoint = self.positions_w[d][w_s + 1]
                for n in self.n:
                    if self.P[d, n, w_s]():
                        if n == self.N_s:
                            # directly to next waypoint
                            x = [cur_waypoint[0], next_waypoint[0]]
                            y = [cur_waypoint[1], next_waypoint[1]]
                        else:
                            # via charging station S
                            pos_S = self.positions_S[n]
                            x = [cur_waypoint[0], pos_S[0], next_waypoint[0]]
                            y = [cur_waypoint[1], pos_S[1], next_waypoint[1]]
                        ax.plot(x, y, constants.W_COLORS[d], zorder=-1)

    def t(self, d, w_s):
        return sum(self.P[d, n, w_s] * (self.T_N[d, n, w_s] + self.T_W[d, n, w_s]) for n in self.n) / self.v[d]

    def schedule(self, d):
        """
        Return the schedule for this model
        :param d: drone identifier
        """
        path = np.reshape(self.P[d, :, :](), (self.N_s + 1, self.N_w_s))
        charging_times = np.array(self.C[d, :]())
        return path, charging_times

    def _get_T_N(self):
        T_n = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.positions_w[d]
            for w_s in range(self.N_w_s):
                row = []
                cur_waypoint = waypoints[w_s]

                # distance to charging points
                for s in range(self.N_s):
                    pos_S = self.positions_S[s]
                    d = dist(cur_waypoint, pos_S)
                    row.append(d)

                # distance to next waypoint
                next_waypoint = waypoints[w_s + 1]
                d = dist(cur_waypoint, next_waypoint)
                row.append(d)
                matr.append(row)
            T_n.append(matr)
        T_n = np.array(T_n).transpose(0, 2, 1)
        return T_n

    def _get_T_W(self):
        T_w = []
        for d in range(self.N_d):
            matr = []
            waypoints = self.positions_w[d]
            for w_s in range(self.N_w_s):
                row = []
                next_waypoint = waypoints[w_s + 1]

                # distance to charging points
                for s in range(self.N_s):
                    pos_S = self.positions_S[s]
                    d = dist(next_waypoint, pos_S)
                    row.append(d)

                row.append(0)
                matr.append(row)
            T_w.append(matr)
        T_w = np.array(T_w).transpose(0, 2, 1)
        return T_w
