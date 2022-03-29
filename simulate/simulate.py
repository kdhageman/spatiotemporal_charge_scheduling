import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from util import constants


class Schedule:
    def __init__(self, decisions: np.ndarray, charging_times: np.ndarray, waiting_times: np.ndarray):
        assert (decisions.ndim == 2)
        assert (charging_times.ndim == 1)
        assert (waiting_times.ndim == 1)

        self.decisions = decisions
        self.charging_times = charging_times
        self.waiting_times = waiting_times


class Parameters:
    def __init__(self, v: float, r_charge: float, r_deplete: float, B_start: float, T_N: np.ndarray, T_W: np.ndarray):
        assert (T_N.ndim == 2)
        assert (T_W.ndim == 2)

        self.v = v
        self.r_charge = r_charge
        self.r_deplete = r_deplete
        self.B_start = B_start
        self.T_N = T_N
        self.T_W = T_W


class Environment:
    def distance(self, x):
        raise NotImplementedError

    def velocity(self, x):
        raise NotImplementedError

    def depletion(self, x):
        raise NotImplementedError


class DeterministicEnvironment(Environment):

    def distance(self, x):
        return x

    def velocity(self, x):
        return x

    def depletion(self, x):
        return x


def nonnegative(func):
    def f(*args, **kwargs):
        res = -1
        while res < 0:
            res = func(*args, **kwargs)
        return res

    return f


class WhiteNoiseEnvironment(Environment):
    def __init__(self, d_scale=0.1, v_scale=0.1, dep_scale=0.1):
        self.d_scale = d_scale
        self.v_scale = v_scale
        self.dep_scale = dep_scale

    @nonnegative
    def distance(self, x):
        return x + x * np.random.normal(0, self.d_scale)

    @nonnegative
    def velocity(self, x):
        return x + x * np.random.normal(0, self.v_scale)

    @nonnegative
    def depletion(self, x):
        return x + x * np.random.normal(0, self.dep_scale)


class Simulation:
    def __init__(self, schedule: Schedule, params: Parameters, env: Environment):
        self.schedule = schedule
        self.params = params
        self.env = env

        self.N_s, self.N_w_s = schedule.decisions.shape

    @classmethod
    def from_base_model(cls, model, d, env=None):
        schedule = Schedule(*model.schedule(d))
        params = Parameters(model.v[d], model.r_charge[d], model.r_deplete[d], model.B_start[d], model.T_N[d],
                            model.T_W[0])
        if not env:
            env = DeterministicEnvironment()

        return Simulation(schedule, params, env)

    def simulate(self):
        # Potential plot fixing
        cur_charge = self.params.B_start
        cur_time = 0
        charges = [cur_charge]
        charge_timestamps = [cur_time]
        charging_windows = []

        for w_s in range(self.N_w_s):
            # time to node
            dist_to_node = (self.params.T_N[:, w_s] * self.schedule.decisions[:, w_s]).sum()
            t_to_node = np.round(dist_to_node / self.params.v, 3)
            depletion = t_to_node * self.params.r_deplete
            cur_charge -= depletion
            cur_time += t_to_node
            charges.append(cur_charge)
            charge_timestamps.append(cur_time)
            if cur_charge < 0:
                return charges, charge_timestamps, charging_windows, False

            # waiting time
            t_wait = np.round(self.schedule.waiting_times[w_s], 3)
            cur_time += t_wait
            charges.append(cur_charge)
            charge_timestamps.append(cur_time)

            # charge time
            ts_charge = cur_time
            t_charge = np.round(self.schedule.charging_times[w_s], 3)
            charged = t_charge * self.params.r_charge
            cur_charge += charged
            cur_time += t_charge
            charges.append(cur_charge)
            charge_timestamps.append(cur_time)

            station = np.where(np.round(self.schedule.decisions[:, w_s]) == 1)[0][0]
            if station != self.N_s:
                charging_windows.append((ts_charge, t_charge, station))

            # to next waypoint
            dist_to_waypoint = (self.params.T_W[:, w_s] * self.schedule.decisions[:, w_s]).sum()
            t_to_waypoint = np.round(dist_to_waypoint / self.params.v, 3)
            depletion = t_to_waypoint * self.params.r_deplete
            cur_charge -= depletion
            cur_time += t_to_waypoint
            charges.append(cur_charge)
            charge_timestamps.append(cur_time)
            if cur_charge < 0:
                return charges, charge_timestamps, charging_windows, False

        return charges, charge_timestamps, charging_windows, True

    def plot_charge(self, ax=None, **kwargs):
        if not ax:
            _, ax = plt.subplots()
        charges, charge_timestamps, charging_windows, ok = self.simulate()
        ax.plot(charge_timestamps, charges, marker='o', **kwargs)
        for x_rect, width_rect, s in charging_windows:
            rect = Rectangle((x_rect, 0), width_rect, 1, color=constants.W_COLORS[s], ec=None, alpha=0.2, zorder=-1)
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Charge")
        ax.set_xlabel("Arrival time at node")
        ax.grid(axis='y')
        return ok
