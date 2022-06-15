import logging
from unittest import TestCase

import numpy as np
import simpy
from simpy import Resource, Timeout

from simulate.node import Waypoint, ChargingStation
from simulate.simulate import TimeStepper
from simulate.uav import UAV


class TestUAV(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_uav(self):
        env = simpy.Environment()

        nodes = np.array([
            # Waypoint(0, 0, 0),
            ChargingStation(1, 0, 0, 0, ct=1, wt=3),
            Waypoint(2, 0, 0),
            ChargingStation(3, 0, 0, 1, ct=2, wt=2),
            Waypoint(4, 0, 0),
        ])
        v = 1

        charging_stations = []
        for i in range(2):
            resource = simpy.Resource(env, capacity=1)
            charging_stations.append(resource)

        def uav_cb(event):
            print(f"{event.env.now} {event.value.name}")

        uav = UAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
        uav.set_schedule(env, (0, 0, 0), nodes)
        uav_proc = env.process(uav.sim(env, callbacks=[uav_cb]))

        def ts_cb(event):
            uav_state = uav.get_state(env)
            print(f"{event.env.now}: {uav_state.state_type} {uav_state.battery} {uav_state.pos}")

        ts = TimeStepper(interval=0.5)
        env.process(ts.sim(env, callbacks=[ts_cb]))

        env.run(until=uav_proc)

        self.assertEqual(env.now, 12)

    def test_set_schedule(self):
        env = simpy.Environment()
        nodes = np.array([
            Waypoint(10, 0, 0),
        ])
        v = 1
        self.event_list = []

        def uav_cb(ev):
            self.event_list.append(ev)

        charging_stations = [simpy.Resource(env)]

        self.uav = UAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
        self.uav.set_schedule(env, (0, 0, 0), nodes)
        uav_proc = env.process(self.uav.sim(env, callbacks=[uav_cb]))

        def timeout(cbs=[]):
            ev = Timeout(env, 1)
            for cb in cbs:
                ev.callbacks.append(cb)
            yield ev

        def timeout_cb(ev):
            nodes = [
                Waypoint(2, 0, 0),
                Waypoint(8, 0, 0),
            ]
            self.uav.set_schedule(env, (1, 0, 0), nodes)

        env.process(timeout(cbs=[timeout_cb]))
        env.run(until=uav_proc)
        pass

    def test_get_state(self):
        """
        Moving a UAV in the following pattern, with the 'o' representing the time stepper evaluation
        2 --o------ 3
        |           |
        |           o
        |           |
        |           |
        1 --o------ 4
        """
        env = simpy.Environment()
        charging_stations = [
            Resource(env)
        ]
        uav = UAV(0, charging_stations, 1, 0.1, 0.1, 1)
        nodes = [
            # Waypoint(0, 0, 0),
            Waypoint(0, 2, 0),
            ChargingStation(0, 5, 0, identifier=0, wt=0, ct=0),
            Waypoint(1, 5, 0),
            Waypoint(5, 5, 0),
            ChargingStation(5, 3, 0, identifier=0, wt=0, ct=0),
            ChargingStation(5, 0, 0, identifier=0, wt=0, ct=0),
            Waypoint(4, 0, 0),
            Waypoint(1, 0, 0),
            Waypoint(0, 0, 0),
        ]
        uav.set_schedule(env, (0, 0, 0), nodes)
        proc = env.process(uav.sim(env))

        self.counter = 0
        expected_arrs = [
            np.array([1, 5, 0]),
            np.array([5, 3, 0]),
            np.array([2, 0, 0]),
        ]

        timestepper = TimeStepper(interval=6)

        def ts_cb(_):
            state = uav.get_state(env)
            actual = state.pos
            expected = expected_arrs[self.counter]
            self.assertTrue(np.array_equal(actual, expected))
            self.counter += 1

        env.process(timestepper.sim(env, callbacks=[ts_cb]))

        env.run(until=proc)
