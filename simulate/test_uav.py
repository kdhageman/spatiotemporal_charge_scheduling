import logging
from unittest import TestCase

import numpy as np
import simpy
from simpy import Resource, Timeout

from simulate.node import Waypoint, ChargingStation
from simulate.simulate import TimeStepper
from simulate.uav import MilpUAV, _EventGenerator, UavStateType, NaiveUAV
from util.scenario import Scenario


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
        eg = _EventGenerator(pos, nodes, v, r_deplete, r_charge, [])

        proc = env.process(eg.sim(env))
        env.run(until=proc)


class TestMilpUAV(TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("pyomo").setLevel(logging.INFO)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("gurobi").setLevel(logging.ERROR)

    def test_sim(self):
        env = simpy.Environment()

        nodes = [
            Waypoint(1, 0, 0),
            ChargingStation(2, 0, 0, identifier=0, ct=2, wt=2),
            ChargingStation(3, 0, 0, identifier=0, ct=1, wt=0),
            Waypoint(4, 0, 0),
        ]

        charging_stations = [simpy.Resource(env)]

        self.n_arrivals = 0
        self.n_waited = 0
        self.n_charged = 0

        def inc_arrivals(_):
            self.n_arrivals += 1

        def inc_waited(_):
            self.n_waited += 1

        def inc_charges(_):
            self.n_charged += 1

        uav = MilpUAV(0, charging_stations, v=1, r_charge=0.1, r_deplete=0.1)
        uav.add_arrival_cb(inc_arrivals)
        uav.add_waited_cb(inc_waited)
        uav.add_charged_cb(inc_charges)

        uav.set_schedule(env, (0, 0, 0), nodes)
        env.process(uav.sim(env))
        env.run()
        self.assertEqual(len(uav.events), 7)
        self.assertEqual(self.n_arrivals, 4)
        self.assertEqual(self.n_waited, 1)
        self.assertEqual(self.n_charged, 2)

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

        uav = MilpUAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
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

        self.uav = MilpUAV(0, charging_stations, v, r_charge=0.1, r_deplete=0.1)
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
        env = simpy.Environment()

        uav = MilpUAV(0, [simpy.Resource(env)], 1, 0.02, 0.1, 1)
        nodes = [
            ChargingStation(1, 0, 0, identifier=0, wt=1, ct=1)
        ]
        uav.set_schedule(env, (0, 0, 0), nodes)
        env.process(uav.sim(env))

        def timeout(delay, cbs):
            timeout = env.timeout(delay)
            for cb in cbs:
                timeout.callbacks.append(cb)
            yield timeout

        expected_vals = [
            (UavStateType.Moving, np.array([0.5, 0, 0]), 0.95),
            (UavStateType.Waiting, np.array([1, 0, 0]), 0.9),
            (UavStateType.Charging, np.array([1, 0, 0]), 0.91),
        ]
        self.expected_offset = 0

        def cb(_):
            expected_state_type, expected_pos, expected_battery = expected_vals[self.expected_offset]
            state = uav.get_state(env)
            self.assertEqual(state.state_type, expected_state_type)
            self.assertTrue(np.array_equal(expected_pos, state.pos))
            self.assertEqual(state.battery, expected_battery)
            self.expected_offset += 1

        env.process(timeout(0.5, cbs=[cb]))
        env.process(timeout(1.5, cbs=[cb]))
        env.process(timeout(2.5, cbs=[cb]))

        env.run()

    def test_integration(self):
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
        uav = MilpUAV(0, charging_stations, 1, 0.1, 0.1, 1)
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


class TestNaiveUAV(TestCase):
    def test_sim(self):
        env = simpy.Environment()
        charging_stations = [
            Resource(env)
        ]
        positions_S = [
            (0, 0),
        ]
        positions_w = [
            [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
            ]
        ]
        sc = Scenario(positions_S=positions_S, positions_w=positions_w)
        self.arrivals = 0
        self.waitings = 0
        self.chargings = 0

        self.events = []
        for _ in range(sc.N_d):
            self.events.append([])

        def arrival_cb(ev):
            self.arrivals += 1
            self.events[ev.value.uav.uav_id].append(ev)

        def waited_cb(ev):
            self.waitings += 1
            self.events[ev.value.uav.uav_id].append(ev)

        def charged_cb(ev):
            self.chargings += 1
            self.events[ev.value.uav.uav_id].append(ev)

        uav = NaiveUAV(0, sc, charging_stations, 1, 0.05, 0.3, 0.3)
        uav.add_arrival_cb(arrival_cb)
        uav.add_waited_cb(waited_cb)
        uav.add_charged_cb(charged_cb)
        env.process(uav.sim(env))

        env.run()

        print(self.arrivals)
        print(self.waitings)
        print(self.chargings)