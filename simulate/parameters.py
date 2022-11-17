import copy

import jsons
import numpy as np


class Parameters:
    def __init__(self, v: float, r_charge: float, r_deplete: float, B_start: float, B_min: float, B_max: float, W_zero_min: np.array, epsilon: float = 0.1, B_end=[], remaining_distances=[], schedule_delta=1, plot_delta=0, W: int = 0, sigma=1,
                 time_limit=60,
                 int_feas_tol=1e-9, rescheduling_frequency=1, delta_t=0.1):
        N_d = len(v)

        self.v = np.array(v)
        self.r_charge = np.array(r_charge)
        self.r_deplete = np.array(r_deplete)
        self.B_start = np.array(B_start)
        self.B_min = np.array(B_min)
        self.B_max = np.array(B_max)
        self.B_end = np.array(B_end)
        if not B_end:
            self.B_end = self.B_min
        self.epsilon = epsilon
        self.remaining_distances = remaining_distances
        if not remaining_distances:
            self.remaining_distances = [0] * N_d
        self.schedule_delta = schedule_delta
        self.plot_delta = plot_delta
        self.W = W
        self.sigma = sigma
        self.W_zero_min = W_zero_min
        self.time_limit = time_limit
        self.int_feas_tol = int_feas_tol
        self.rescheduling_frequency = rescheduling_frequency
        self.delta_t = delta_t

    def as_dict(self):
        return dict(
            v=self.v,
            r_charge=self.r_charge,
            r_deplete=self.r_deplete,
            B_start=self.B_start,
            B_min=self.B_min,
            B_max=self.B_max,
            B_end=self.B_end,
            W_zero_min=self.W_zero_min,
            epsilon=self.epsilon,
            schedule_delta=self.schedule_delta,
            plot_delta=self.plot_delta,
            W=self.W,
            sigma=self.sigma,
            remaining_distances=self.remaining_distances,
            time_limit=self.time_limit,
            int_feas_tol=self.int_feas_tol,
            rescheduling_frequency=self.rescheduling_frequency,
            delta_t=self.delta_t,
        )

    def copy(self):
        return copy.deepcopy(self)


def parameters_serializer(obj: Parameters, *args, **kwargs):
    res = obj.as_dict()
    for k, v in res.items():
        if type(v) == np.ndarray:
            res[k] = v.astype(float).tolist()
    return res


jsons.set_serializer(parameters_serializer, Parameters)
