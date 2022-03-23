import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class Schedule:
    def __init__(self, decisions: np.ndarray, charging_times: np.ndarray):
        assert (decisions.ndim == 2)
        assert (charging_times.ndim == 1)

        self.decisions = decisions
        self.charging_times = charging_times


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


class Simulation:
    def __init__(self, schedule: Schedule, params: Parameters, env: Environment):
        self.schedule = schedule
        self.params = params
        self.env = env

    @classmethod
    def from_base_model(cls, model, d):
        schedule = Schedule(*model.schedule(d))
        params = Parameters(model.v[d], model.r_charge[d], model.r_deplete[d], model.B_start[d], model.T_N[d], model.T_W[0])
        env = DeterministicEnvironment()

        return Simulation(schedule, params, env)

    def simulate(self):
        cur_charge = self.params.B_start
        charges = [cur_charge]
        timestamps = [0]
        path = ['$w_{1}$']
        path_arrivals = [0]
        charging_windows = []

        for w_s in range(self.schedule.decisions.shape[1]):
            p = self.schedule.decisions[:, w_s]
            for n in range(len(p)):
                if p[n] == 1:
                    # to path node
                    dist = self.env.distance(self.params.T_N[n, w_s])
                    velocity = self.env.velocity(self.params.v)
                    time = dist / velocity
                    arrival_timestamp = timestamps[-1] + time
                    depletion_rate = self.env.depletion(self.params.r_deplete)
                    depleted = depletion_rate * time
                    cur_charge -= depleted
                    charges.append(cur_charge)
                    timestamps.append(arrival_timestamp)

                    # charge at station
                    charging_time = self.schedule.charging_times[w_s]
                    charged = self.params.r_charge * charging_time
                    if charged > 0:
                        cur_charge += charged
                        charges.append(cur_charge)
                        charging_finished_timestamp = timestamps[-1] + charging_time
                        timestamps.append(charging_finished_timestamp)
                        path_arrivals.append(charging_finished_timestamp)
                        path.append(f"$s_{{{n + 1}}}$")
                        charging_windows.append((arrival_timestamp, charging_time))

                    # move to next waypoint
                    dist = self.env.distance(self.params.T_W[n, w_s])
                    velocity = self.env.velocity(self.params.v)
                    time = dist / velocity
                    next_waypoint_timestamp = timestamps[-1] + time
                    depletion_rate = self.env.depletion(self.params.r_deplete)
                    depleted = depletion_rate * time
                    if depleted > 0:
                        cur_charge -= depleted
                        charges.append(cur_charge)
                        timestamps.append(next_waypoint_timestamp)
                    path_arrivals.append(next_waypoint_timestamp)
                    path.append(f"$w_{{{w_s + 2}}}$")
        return charges, timestamps, path, path_arrivals, charging_windows

    def plot_charge(self, ax=None):
        if not ax:
            _, ax = plt.subplots()
        charges, timestamps, path, path_arrivals, charging_windows = self.simulate()

        ax.plot(timestamps, charges, marker='o')
        for x_rect, width_rect in charging_windows:
            rect = Rectangle((x_rect, 0), width_rect, 1, color='g', alpha=0.2, zorder=-1)
            ax.add_patch(rect)
        ax.set_xticks(path_arrivals, path, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Charge")
        ax.set_xlabel("Arrival time at node")
        ax.grid(axis='y')
