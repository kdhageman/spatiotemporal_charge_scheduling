import logging
from unittest import TestCase

import numpy as np
import simpy
from simpy import Resource, Timeout

from simulate.node import Waypoint, ChargingStation
from simulate.simulate import TimeStepper, Parameters
from simulate.uav import UAV, _EventGenerator


class TestEventGenerator(TestCase):
    def test_event_generator(self):
        env = simpy.Environment()

        pos = (0, 0, 0)
        nodes = [
            Waypoint(1, 0, 0),
            Waypoint(2, 0, 0),
            Waypoint(3, 0, 0),
        ]
        v = 1
        battery = 1
        r_deplete = 0.1
        r_charge = 0.1
        eg = _EventGenerator(pos, nodes, v, battery, r_deplete, r_charge, [])

        self.sim_cb_count = 0
        self.finish_cb_count = 0

        def sim_cb(event):
            self.sim_cb_count += 1

        def finish_cb():
            self.finish_cb_count += 1

        eg.add_arrival_cb(sim_cb)
        eg.add_finish_cb(finish_cb)
        eg.add_finish_cb(finish_cb)

        proc = env.process(eg.sim(env))
        env.run(until=proc)

        self.assertEqual(self.sim_cb_count, 3)
        self.assertEqual(self.finish_cb_count, 2)


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
        uav.add_arrival_cb(uav_cb)
        uav.set_schedule(env, (0, 0, 0), nodes)
        uav_proc = env.process(uav.sim(env))

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
        self.events = []
        charging_stations = [simpy.Resource(env)]

        def arrival_cb(ev):
            self.events.append(ev)

        self.uav = UAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
        self.uav.add_arrival_cb(arrival_cb)
        self.uav.set_schedule(env, (0, 0, 0), nodes)
        uav_proc = env.process(self.uav.sim(env))

        def timeout(delay, cbs=[]):
            ev = Timeout(env, delay)
            for cb in cbs:
                ev.callbacks.append(cb)
            yield ev

        def timeout_cb(ev):
            nodes = [
                Waypoint(2, 0, 0),
                Waypoint(8, 0, 0),
            ]
            start_pos = (1, 0, 0)
            self.uav.set_schedule(env, start_pos, nodes)

        env.process(timeout(1, cbs=[timeout_cb]))
        env.run(until=uav_proc)
        self.assertEqual(len(self.events), 2)


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
