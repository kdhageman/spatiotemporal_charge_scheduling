from itertools import product

import numpy as np
import pyomo.environ as pyo
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from util import constants
from util.distance import dist3


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

        self.D_N = self._get_D_N()
        self.D_W = self._get_D_W()

        # MODEL DEFINITION
        self.d = pyo.RangeSet(0, self.N_d - 1)
        self.s = pyo.RangeSet(0, self.N_s - 1)
        self.w = pyo.RangeSet(0, self.N_w - 1)
        self.w_s = pyo.RangeSet(0, self.N_w_s - 1)
        self.w_d = pyo.RangeSet(1, self.N_w - 1)
        self.n = pyo.RangeSet(0, self.N_s)

        # VARIABLES
        # control
        self.P = pyo.Var(self.d, self.n, self.w_s, domain=pyo.Binary)
        self.C = pyo.Var(self.d, self.w_s, domain=pyo.NonNegativeReals)
        self.W = pyo.Var(self.d, self.w_s, bounds=(0, self.W_max))
        self.alpha = pyo.Var()  # used for minmax

        # CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: sum(m.P[d, n, w_s] for n in self.n) == 1
        )

        # lower and upper bounds of variables values
        self.b_arr_llim = pyo.Constraint(
            self.d,
            self.w_d,
            rule=lambda m, d, w: m.b_arr(d, w) >= self.B_min[d]
        )

        self.b_min_llim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_min(d, w_s) >= self.B_min[d]
        )
        self.b_plus_ulim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.b_plus(d, w_s) <= self.B_max[d]
        )

        self.C_lim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.C[d, w_s] <= (1 - m.P[d, self.N_s, w_s]) * self.C_max[d]
        )

        self.W_lim = pyo.Constraint(
            self.d,
            self.w_s,
            rule=lambda m, d, w_s: m.W[d, w_s] <= (1 - m.P[d, self.N_s, w_s]) * sum(self.C_max)
        )

        self.alpha_min = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.alpha >= m.E(d)
        )

        self.execution_time = pyo.Objective(
            expr=self.alpha,
            sense=pyo.minimize,
        )

    def initialize_variables(self, seed=None):
        np.random.seed(seed=seed)

        # randomly intitialize P
        for d, w_s in product(self.d, self.w_s):
            # pick path node
            n_selected = np.random.randint(self.N_s + 1)
            for n in range(self.N_s + 1):
                if n == n_selected:
                    self.P[d, n, w_s] = 1
                else:
                    self.P[d, n, w_s] = 0

        # randomize W
        for d, w_s in product(self.d, self.w_s):
            self.W[d, w_s] = np.random.rand() * self.W_max

        # randomize C
        for d, w_s in product(self.d, self.w_s):
            self.C[d, w_s] = np.random.rand() * self.C_max[d]

    def b_arr(self, d, w):
        """
        Calculate the battery at arrival at waypoint 'w' for drone 'd'
        """
        if w == 0:
            # base case
            res = self.B_start[d]
        else:
            res = self.b_plus(d, w - 1) - self.r_deplete[d] / self.v[d] * sum(
                self.P[d, n, w - 1] * self.D_W[d, n, w - 1] for n in self.n)
        return res

    def b_min(self, d, w_s):
        """
        Calculate the battery of drone 'd' when arriving at the next path node after waypoint 'w_s'
        """
        return self.b_arr(d, w_s) - self.r_deplete[d] / self.v[d] * sum(
            self.P[d, n, w_s] * self.D_N[d, n, w_s] for n in self.n)

    def b_plus(self, d, w_s):
        """
        Calculate the battery of drone 'd' after charging after waypoint 'w_s'
        """
        return self.b_min(d, w_s) + self.r_charge[d] * self.C[d, w_s]

    @property
    def W_max(self):
        return sum(self.C_max)

    # OBJECTIVE
    def E(self, d):
        return sum(self.C[d, w_s] + self.W[d, w_s] + self.t(d, w_s) for w_s in self.w_s)

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
        return sum(self.P[d, n, w_s] * (self.D_N[d, n, w_s] + self.D_W[d, n, w_s]) for n in self.n) / self.v[d]

    def schedules(self):
        schedules = []
        for d in self.d:
            schedules.append(self.schedule(d))
        return schedules

    def schedule(self, d):
        """
        Return the schedule for this model
        :param d: drone identifier
        """
        path = np.reshape(self.P[d, :, :](), (self.N_s + 1, self.N_w_s))
        charging_times = np.array(self.C[d, :]())
        waiting_times = np.array(self.W[d, :]())
        return path, charging_times, waiting_times

    def _get_D_N(self):
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
                    d = dist3(cur_waypoint, pos_S)
                    row.append(d)

                # distance to next waypoint
                next_waypoint = waypoints[w_s + 1]
                d = dist3(cur_waypoint, next_waypoint)
                row.append(d)
                matr.append(row)
            T_n.append(matr)
        T_n = np.array(T_n).transpose(0, 2, 1)
        return T_n

    def _get_D_W(self):
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
                    d = dist3(next_waypoint, pos_S)
                    row.append(d)

                row.append(0)
                matr.append(row)
            T_w.append(matr)
        T_w = np.array(T_w).transpose(0, 2, 1)
        return T_w

    @property
    def P_np(self):
        """
        Returns the chosen path decision variable (P) as a numpy array
        """
        return np.reshape(self.P[:, :, :](), (self.N_d, self.N_s + 1, self.N_w_s))

    @property
    def C_np(self):
        """
        Return the charging time decision variable (C) as a numpy array
        """
        return np.reshape(self.C[:, :](), (self.N_d, self.N_w_s))

    @property
    def W_np(self):
        """
        Return the waiting time decision variable (W) as a numpy array
        """
        return np.reshape(self.W[:, :](), (self.N_d, self.N_w_s))

    def plot_charge(self, d: int, ax=None, **kwargs):
        """
        Plot the optimized battery charge for drone 'd'
        """
        if ax is None:
            _, ax = plt.subplots()

        P = np.reshape(self.P[:, :, :](), (self.N_d, self.N_s + 1, self.N_w_s))
        C = np.reshape(self.C[:, :](), (self.N_d, self.N_w_s)).round(7)
        W = np.reshape(self.W[:, :](), (self.N_d, self.N_w_s)).round(7)
        b_arr = np.reshape(self.b_arr[:, :](), (self.N_d, self.N_w))
        b_min = np.reshape(self.b_min[:, :](), (self.N_d, self.N_w_s))
        b_plus = np.reshape(self.b_plus[:, :](), (self.N_d, self.N_w_s))

        T_N = (self.D_N * P).sum(axis=1) / np.reshape(self.v, (self.N_d, 1))
        T_W = (self.D_W * P).sum(axis=1) / np.reshape(self.v, (self.N_d, 1))

        C_cum = np.cumsum(C, axis=1)
        W_cum = np.cumsum(W, axis=1)
        T_N_cum = np.cumsum(T_N, axis=1)
        T_W_cum = np.cumsum(T_W, axis=1)

        t_cur = 0
        b_cur = self.B_start[d]

        X = [t_cur]
        Y = [b_cur]

        for w_s in self.w_s:
            t_n = T_N[d, w_s]
            t_w = T_W[d, w_s]
            w = W[d, w_s]
            c = C[d, w_s]

            # update timestamps
            for t in [t_n, w, c, t_w]:
                t_cur += t
                X.append(t_cur)
            # update batteries
            Y.append(b_min[d, w_s])  # after arriving at node
            Y.append(b_min[d, w_s])  # after waiting at node
            Y.append(b_plus[d, w_s])  # after charging at node
            Y.append(b_arr[d, w_s + 1])  # after arriving at next waypoint

            # charging windows
            t_charging_window = T_N_cum[d, w_s] + W_cum[d, w_s]
            if w_s > 1:
                t_charging_window += C_cum[d, w_s - 1] + T_W_cum[d, w_s - 1]
            s = np.where(P[d, :, w_s] == 1)[0][0]
            if s != self.N_s:
                rect = Rectangle((t_charging_window, 0), C[d, w_s], 1, color=constants.W_COLORS[s], ec=None, alpha=0.2,
                                 zorder=-1)
                ax.add_patch(rect)

        ax.plot(X, Y, marker='o', **kwargs)
