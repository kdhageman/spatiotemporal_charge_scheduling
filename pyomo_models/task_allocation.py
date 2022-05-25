from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import yaml
from matplotlib.patches import Rectangle
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

        self.positions_n = self.positions_w + self.positions_s

        self.positions_start = []
        for pos in doc.get("start_positions", []):
            x, y, z = [float(v) for v in pos.split(",")]
            self.positions_start.append((x, y, z))

        self.N_s = len(self.positions_s)
        self.N_w = len(self.positions_w)
        self.N_n = len(self.positions_n)
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

    def plot(self, ax=None):
        if not ax:
            _, ax = plt.subplots()

        x_start = [x for x, _, _ in self.positions_start]
        y_start = [y for _, y, _ in self.positions_start]
        x_wp = [x for x, _, _ in self.positions_w]
        y_wp = [y for _, y, _ in self.positions_w]
        x_s = [x for x, _, _ in self.positions_s]
        y_s = [y for _, y, _ in self.positions_s]

        ax.scatter(x_start, y_start, marker='o', color='black', label='start')
        ax.scatter(x_wp, y_wp, marker='s', color='red', label='waypoint')
        for i, (x, y) in enumerate(zip(x_wp, y_wp)):
            label = f"{i}"
            ax.text(x + 0.5, y + 0.5, label, color='red')
        ax.scatter(x_s, y_s, marker='x', color='blue', label='charging station')
        # ax.legend()


class Schedule:
    def __init__(self, model):

        self.P = np.reshape(model.P[:, :, :](), (model.N_d, model.N_n, model.N_t))
        self.I = np.reshape(model.I[:, :](), (model.N_d, model.N_t))
        self.b = np.reshape(model.b[:, :](), (model.N_d, model.N_t + 1))
        self.b_min_fly = np.reshape(model.b_min_fly[:, :](), (model.N_d, model.N_t))
        self.b_min_inspect = np.reshape(model.b_min_inspect[:, :](), (model.N_d, model.N_t))
        self.b_plus = np.reshape(model.b_plus[:, :](), (model.N_d, model.N_t))

        self.model = model

    def plot_battery(self, d, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        t_cur = 0
        b_cur = self.b[d, 0]
        timestamps = [t_cur]
        battery_charges = [b_cur]

        charge_windows = []
        inspect_windows = []

        for t in range(self.model.N_t):
            # charge
            t_charge = self.model.charging_time_at_t(d, t)()
            b_charge = self.b_plus[d, t]
            charge_windows.append((t_cur, t_charge))
            t_cur += t_charge
            b_cur += b_charge
            timestamps.append(t_cur)
            battery_charges.append(b_cur)

            # move
            t_move = self.model.moving_time_at_t(d, t)()
            b_move = self.b_min_fly[d, t]
            t_cur += t_move
            b_cur -= b_move
            timestamps.append(t_cur)
            battery_charges.append(b_cur)

            # inspect
            t_inspect = self.model.inspection_time_at_t(d, t)()
            b_inspect = self.b_min_inspect[d, t]
            inspect_windows.append((t_cur, t_inspect))
            t_cur += t_inspect
            b_cur -= b_inspect
            timestamps.append(t_cur)
            battery_charges.append(b_cur)

        ax.plot(timestamps, battery_charges, marker='o')
        # plot charging windows
        for x_rect, width_rect in charge_windows:
            rect = Rectangle((x_rect, 0), width_rect, 1, color='green', ec=None, alpha=0.2, zorder=-1)
            ax.add_patch(rect)

        # plot inspect windows
        for x_rect, width_rect in inspect_windows:
            rect = Rectangle((x_rect, 0), width_rect, 1, color='red', ec=None, alpha=0.2, zorder=-1)
            ax.add_patch(rect)

    def plot_path(self, d, sc, ax=None, **kwargs):
        if not ax:
            _, ax = plt.subplots()

        path = [sc.positions_start[d]]
        for t in self.model.t:
            n = self.P[d, :, t].tolist().index(1)
            path.append(sc.positions_n[n])

        x_path = [x for x, _, _ in path]
        y_path = [y for _, y, _ in path]
        ax.plot(x_path, y_path, **kwargs)


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

        self.N_d = scenario.N_d
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
        self.t_b = pyo.RangeSet(-1, self.N_t - 1)  # used to express the battery state before starting

        self.D_start = scenario.D_start
        self.D = scenario.D

        # VARIABLES
        self.P = pyo.Var(self.d, self.n, self.t, domain=pyo.Binary)
        self.I = pyo.Var(self.d, self.t, domain=pyo.PositiveReals)
        self.b = pyo.Var(self.d, self.t_b)
        self.b_plus = pyo.Var(self.d, self.t)
        self.b_min_fly = pyo.Var(self.d, self.t)
        self.b_min_inspect = pyo.Var(self.d, self.t)
        self.alpha = pyo.Var(domain=pyo.PositiveReals)  # used for minmax
        self.y = pyo.Var(self.d, self.n, self.n, self.t_r, domain=pyo.Binary)

        # OBJECTIVE
        self.alpha_min = pyo.Constraint(
            self.d,
            # rule=lambda m, d: m.alpha >= m.inspection_time(d) + m.charging_time(d) + m.moving_time(d)
            rule=lambda m, d: m.alpha >= m.inspection_time(d) + m.charging_time(d)
        )

        self.execution_time = pyo.Objective(
            expr=self.alpha,
            sense=pyo.minimize,
        )

        # LINEAR CONSTRAINTS
        self.y_constr_1 = pyo.Constraint(
            self.d,
            self.n,
            self.n,
            self.t_r,
            rule=lambda m, d, n, n_prime, t_r: m.y[d, n, n_prime, t_r] <= m.P[d, n, t_r - 1]
        )

        self.y_constr_2 = pyo.Constraint(
            self.d,
            self.n,
            self.n,
            self.t_r,
            rule=lambda m, d, n, n_prime, t_r: m.y[d, n, n_prime, t_r] <= m.P[d, n_prime, t_r]
        )

        self.y_constr_3 = pyo.Constraint(
            self.d,
            self.n,
            self.n,
            self.t_r,
            rule=lambda m, d, n, n_prime, t_r: m.y[d, n, n_prime, t_r] >= m.P[d, n, t_r - 1] + m.P[d, n_prime, t_r] - 1
        )

        # PATH CONSTRAINTS
        self.path_constraint = pyo.Constraint(
            self.d,
            self.t,
            rule=lambda m, d, t: sum(m.P[d, n, t] for n in m.n) == 1
        )

        # BATTERY CONSTRAINTS

        self.b_plus_calc_first = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b_plus[d, 0] == 0
        )

        self.b_plus_calc = pyo.Constraint(
            self.d,
            self.t_r,
            rule=lambda m, d, t_r: m.b_plus[d, t_r] == sum(
                m.P[d, s, t_r - 1] * (m.B_max - m.b[d, t_r - 1]) for s in m.s)
        )

        self.b_min_fly_calc_first = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b_min_fly[d, 0] == m.r_deplete_fly[d] / m.v[d] * sum(
                m.P[d, n, 1] * m.D_start[d, n] for n in m.n)
        )

        self.b_min_fly_calc = pyo.Constraint(
            self.d,
            self.t_r,
            rule=lambda m, d, t_r: m.b_min_fly[d, t_r] == m.r_deplete_fly[d] / m.v[d] * (
                sum(m.y[d, n, n_prime, t_r] * m.D[n, n_prime] for n, n_prime in product(m.n, m.n)))
        )

        self.b_min_inspect_calc = pyo.Constraint(
            self.d,
            self.t,
            rule=lambda m, d, t: m.b_min_inspect[d, t] == m.I[d, t] * m.r_deplete_inspect[d]
        )

        self.b_calc_start = pyo.Constraint(
            self.d,
            rule=lambda m, d: m.b[d, -1] == m.B_start[d]
        )

        self.b_calc = pyo.Constraint(
            self.d,
            self.t,
            rule=lambda m, d, t: m.b[d, t] == m.b[d, t - 1] + m.b_plus[d, t] - m.b_min_fly[d, t] - m.b_min_inspect[d, t]
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

    def inspection_time_at_t(self, d, t):
        return self.I[d, t]

    def inspection_time(self, d):
        """
        Returns the inspection time for drone 'd'
        """
        return sum(self.inspection_time_at_t(d, t) for t in self.t)

    def charging_time_at_t(self, d, t):
        return sum(self.P[d, s, t] * (self.B_max - self.b[d, t]) for s in self.s) / self.r_charge[d]

    def charging_time(self, d):
        """
        Return the charging time for drone 'd'
        """
        return sum(self.charging_time_at_t(d, t) for t in self.t)

    def moving_time_at_t(self, d, t):
        if t == 0:
            return sum(self.D_start[d, n] * self.P[d, n, 0] for n in self.n) / self.v[d]
        else:
            return sum(self.y[d, n, n_prime, t] * self.D[n, n_prime] for n, n_prime in
                       product(self.n, self.n)) / self.v[d]

    def moving_time(self, d):
        """
        Return the total moving time of drone 'd'
        """
        return sum(self.moving_time_at_t(d, t) for t in self.t)
