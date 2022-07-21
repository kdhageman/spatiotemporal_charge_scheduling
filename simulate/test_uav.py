import logging
from unittest import TestCase

import simpy

from simulate.node import Waypoint, ChargingStation
from simulate.uav import UAV


class TestUav(TestCase):
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

        charging_stations = [simpy.PriorityResource(env)]

        self.n_arrivals = 0
        self.n_waited = 0
        self.n_charged = 0

        def inc_arrivals(_):
            self.n_arrivals += 1

        def inc_waited(_):
            self.n_waited += 1

        def inc_charges(_):
            self.n_charged += 1

        uav = UAV(0, charging_stations, v=1, r_charge=0.1, r_deplete=0.1, initial_pos=(0, 0, 0))
        uav.add_arrival_cb(inc_arrivals)
        uav.add_waited_cb(inc_waited)
        uav.add_charged_cb(inc_charges)

        uav.set_schedule(env, nodes)
        env.process(uav.sim(env))
        env.run()
        self.assertEqual(len(uav.events), 8)
        self.assertEqual(self.n_arrivals, 4)
        self.assertEqual(self.n_waited, 1)
        self.assertEqual(self.n_charged, 2)
