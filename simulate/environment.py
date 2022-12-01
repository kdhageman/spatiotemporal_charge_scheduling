from typing import Tuple

from numpy.random import MT19937, SeedSequence, RandomState


class Environment:
    def move(self, delta_t, dist_to_dest, v, r_deplete, b_cur, b_min=0) -> Tuple[float, float, float, bool]:
        """
        Returns the elapsed time, distance and depletion that a drone experienced during "delta_t" while moving.
        Also returns whether the drone reached its destination during this timeslice.
        :param delta_t: time slice
        :param dist_to_dest: remaining distance to the destination node
        :param v: expected velocity
        :param r_deplete: expected depletion rate
        :param b_cur: current battery
        :param b_min: minimum battery
        """
        raise NotImplementedError

    def charge(self, delta_t, remaining_charge, r_charge, b_cur, b_max=1) -> Tuple[float, float, bool]:
        """
        Returns the elapsed time and depletion that the drone experienced during "delta_t" while charging.
        Also returns whether the drone reached it maximum charge during this timeslice.
        :param delta_t: time slice
        :param remaining_charge: target remaining charge
        :param r_charge: expected charging rate
        :param b_cur: current battery
        :param b_max: maximum battery
        """
        raise NotImplementedError


class NormalDistributedEnvironment(Environment):
    seed = 1

    def __init__(self, scale, seed=None):
        self.scale = scale
        self.rs = RandomState(MT19937(SeedSequence(seed)))

    @classmethod
    def from_seed(cls, stddev, seed=None):
        # ensures that multiple distributions called with the same seed will return a predictable, yet different environment
        if seed is None:
            env = NormalDistributedEnvironment(stddev, seed)
        else:
            env = NormalDistributedEnvironment(stddev, cls.seed + seed)
            cls.seed += 1
        return env

    def _sim(self, x):
        """
        Returns a simulated value around the given 'x' with standard deviation of the scale of the environment
        """
        return x * max(0.1, self.rs.normal(1, scale=self.scale))

    def move(self, delta_t, dist_to_dest, v, r_deplete, b_cur, b_min=0) -> Tuple[float, float, float, bool]:
        real_v = self._sim(v)
        real_r_deplete = self._sim(r_deplete)

        # calculate the distance and time the UAV will take to reach the destination (or get as close as possible)
        if delta_t * real_v <= dist_to_dest:
            max_dist = delta_t * real_v
        else:
            max_dist = dist_to_dest
        max_travel_delta_t = max_dist / real_v

        # calculate the maximum depletion and the time it will take to deplete that much
        max_deplete_delta_t = (b_cur - b_min) / real_r_deplete

        real_delta_t = min(delta_t, max_travel_delta_t, max_deplete_delta_t)
        real_depletion = real_delta_t * real_r_deplete
        real_dist = real_delta_t * real_v

        # only when max_travel_delta_t is being used, the UAV reached the destination
        if real_delta_t == delta_t or real_delta_t == max_deplete_delta_t:
            reached_destination = False
        else:
            reached_destination = True

        return real_delta_t, real_dist, real_depletion, reached_destination

    def charge(self, delta_t, remaining_charge, r_charge, b_cur, b_max=1) -> Tuple[float, float, bool]:
        real_r_charge = self._sim(r_charge)
        maximum_charge_amount = min(remaining_charge, b_max - b_cur)
        if delta_t * real_r_charge <= maximum_charge_amount:
            real_charge = delta_t * real_r_charge
            finished_charging = False
        else:
            real_charge = maximum_charge_amount
            finished_charging = True
        real_delta_t = real_charge / real_r_charge

        return real_delta_t, real_charge, finished_charging


class DeterministicEnvironment(NormalDistributedEnvironment):
    def __init__(self):
        super().__init__(0)
