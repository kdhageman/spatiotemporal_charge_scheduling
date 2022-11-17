from typing import Tuple

from numpy.random import MT19937, SeedSequence, RandomState


class Environment:
    def move(self, delta_t, dist_to_dest, v, r_deplete) -> Tuple[float, float, float, bool]:
        """
        Returns the elapsed time, distance and depletion that a drone experienced during "delta_t" while moving.
        Also returns whether the drone reached its destination during this timeslice.
        :param delta_t: time slice
        :param dist_to_dest: remaining distance to the destination node
        :param v: expected velocity
        :param r_deplete: expected depletion rate
        """
        raise NotImplementedError

    def charge(self, delta_t, remaining_charge, r_charge) -> Tuple[float, float, bool]:
        """
        Returns the elapsed time and depletion that the drone experienced during "delta_t" while charging.
        Also returns whether the drone reached it maximum charge during this timeslice.
        :param delta_t: time slice
        :param remaining_charge: target remaining charge
        :param r_charge: expected charging rate
        """
        raise NotImplementedError


class DeterministicEnvironment(Environment):

    def move(self, delta_t, dist_to_dest, v, r_deplete) -> Tuple[float, float, float, bool]:
        if delta_t * v <= dist_to_dest:
            real_dist = delta_t * v
            reached_destination = False
        else:
            real_dist = dist_to_dest
            reached_destination = True
        real_delta_t = real_dist / v
        real_depletion = real_delta_t * r_deplete
        return real_delta_t, real_dist, real_depletion, reached_destination

    def charge(self, delta_t, remaining_charge, r_charge) -> Tuple[float, float, bool]:
        if delta_t * r_charge <= remaining_charge:
            real_charge = delta_t * r_charge
            finished_charging = False
        else:
            real_charge = remaining_charge
            finished_charging = True
        real_delta_t = real_charge / r_charge
        return real_delta_t, real_charge, finished_charging


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

    def move(self, delta_t, dist_to_dest, v, r_deplete) -> Tuple[float, float, float, bool]:
        real_v = self._sim(v)
        if delta_t * real_v <= dist_to_dest:
            real_dist = delta_t * real_v
            reached_destination = False
        else:
            real_dist = dist_to_dest
            reached_destination = True
        real_delta_t = real_dist / real_v
        real_depletion = real_delta_t * self._sim(r_deplete)

        return real_delta_t, real_dist, real_depletion, reached_destination

    def charge(self, delta_t, remaining_charge, r_charge) -> Tuple[float, float, bool]:
        real_r_charge = self._sim(r_charge)
        if delta_t * real_r_charge <= remaining_charge:
            real_charge = delta_t * real_r_charge
            finished_charging = False
        else:
            real_charge = remaining_charge
            finished_charging = True
        real_delta_t = real_charge / real_r_charge
        return real_delta_t, real_charge, finished_charging