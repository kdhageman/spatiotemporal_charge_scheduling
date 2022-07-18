import copy

import numpy as np


class Parameters:
    def __init__(self, v: float, r_charge: float, r_deplete: float, B_start: float, B_min: float, B_max: float,
                 epsilon: float = 0.1, B_end=[], schedule_delta=1, plot_delta=0, W: int = 0, sigma=1):
        self.v = np.array(v)
        self.r_charge = np.array(r_charge)
        self.r_deplete = np.array(r_deplete)
        self.B_start = np.array(B_start)
        self.B_min = np.array(B_min)
        self.B_max = np.array(B_max)
        self.B_end = np.array(B_end)
        self.epsilon = epsilon
        self.schedule_delta = schedule_delta
        self.plot_delta = plot_delta
        self.W = W
        self.sigma = sigma

    def as_dict(self):
        return dict(
            v=self.v,
            r_charge=self.r_charge,
            r_deplete=self.r_deplete,
            B_start=self.B_start,
            B_min=self.B_min,
            B_max=self.B_max,
            B_end=self.B_end,
            epsilon=self.epsilon,
            schedule_delta=self.schedule_delta,
            plot_delta=self.plot_delta,
            W=self.W,
            sigma=self.sigma,
        )

    def copy(self):
        return copy.deepcopy(self)
