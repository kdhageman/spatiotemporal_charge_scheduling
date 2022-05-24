from unittest import TestCase

from pyomo_models.task_allocation import Scenario


class TestScenario(TestCase):
    def test_from_file(self):
        Scenario.from_file("../scenarios/task_allocation/test.yaml")
