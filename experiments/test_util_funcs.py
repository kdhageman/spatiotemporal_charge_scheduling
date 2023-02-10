from unittest import TestCase

import networkx as nx
import open3d as o3d

from experiments.util_funcs import plan_path


class TestUtils(TestCase):
    def test_plan_path(self):
        points = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (-1, 2, 0),
            (-1, 3, 0),
            (1, 1, 0),
            (1, 2, 0),
        ]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        g_local = nx.Graph()
        edges = [
            (0, 1),
            (1, 2),
            (3, 4),
            (5, 6),
        ]
        g_local.add_edges_from(edges)

        g_global = nx.Graph()
        edges += [
            (1, 3),
            (1, 6),
        ]
        g_global.add_edges_from(edges)

        path = plan_path(pcd, g_local, g_global, z_penalty=1)
        print(path)
