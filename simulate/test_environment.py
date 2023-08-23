from unittest import TestCase
from simulate.environment import NormalDistributedEnvironment


class TestNormalDistributedEnvironment(TestCase):
    def test_from_constructor(self):
        env1 = NormalDistributedEnvironment(scale=0.1, seed=1)
        env2 = NormalDistributedEnvironment(scale=0.1, seed=1)
        first = env1.rs.normal(0.1)
        second = env2.rs.normal(0.1)
        self.assertEqual(first, second)

    def test_from_seed(self):
        env1 = NormalDistributedEnvironment.from_seed(scale=0.1, seed=1)
        env2 = NormalDistributedEnvironment.from_seed(scale=0.1, seed=1)
        first = env1.rs.normal(0.1)
        second = env2.rs.normal(0.1)
        self.assertNotEqual(first, second)