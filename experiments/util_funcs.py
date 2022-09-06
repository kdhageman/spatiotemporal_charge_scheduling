import json
import logging
import os
import pickle
import time
from datetime import datetime
from enum import Enum
from typing import List

import cvxpy
import jsons
import networkx as nx
import numpy as np
import nxmetis
import open3d as o3d
import scipy
from matplotlib import pyplot as plt
from open3d.cpu.pybind.geometry import Geometry3D
from pyomo.opt import SolverFactory
from scipy.spatial import KDTree

from simulate.event import EventType
from simulate.parameters import Parameters
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import gen_colors, Simulator
from simulate.strategy import OnEventStrategySingle, AfterNEventsStrategyAll
from util.decorators import timed
from util.distance import dist3
from util.scenario import Scenario

logging.basicConfig(
    level=logging.DEBUG
)
logging.getLogger("pyomo.core").setLevel(logging.WARN)
logging.getLogger("pyomo.opt").setLevel(logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARN)
logger = logging.getLogger(__name__)


@timed
def downsample(pcd: o3d.geometry.PointCloud, voxel_size: float, normalize=False):
    """
    Downsamples the point cloud with the given voxel size granularity.
    Optionally also normalizes the normals
    """
    downsampled = pcd.voxel_down_sample(voxel_size)
    if normalize:
        downsampled.normalize_normals()
    return downsampled


@timed
def estimate_normal(pcd: o3d.geometry.PointCloud):
    """
    Estimate the normal vectors for the given point cloud
    Also orients the normals with respect to consistent tangent planes.
    """
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(9)
    return pcd


def draw_geometries(geometries: List[Geometry3D], xrot: float, yrot: float, xtrans: float = 0, ytrans: float = 0, fname: str = None, **kwargs):
    vis = o3d.visualization.Visualizer()

    vis.create_window(**kwargs)
    for g in geometries:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.translate(xtrans, ytrans)
    ctr.rotate(xrot, yrot)
    if fname:
        dat = np.asarray(vis.capture_screen_float_buffer(True))
        dat = np.minimum(dat, 1)
        plt.imsave(fname, dat)
    else:
        vis.run()
        vis.destroy_window()


def pcd_to_graph(pcd: o3d.geometry.PointCloud, nb_count: int = 5, z: int = 1):
    """
    Extracts a graph fro m the given point cloud Each node in the graph is connected to at least 'nb_count' neighbors
    Edges in the graph are annotated with the distance between the nodes, where 'z' denotes an additional punishment
    for vertical distances
    """
    # create KD tree
    points = np.asarray(pcd.points)
    tree = KDTree(points)

    # create neighbor list
    edge_list = []
    for idx, p in enumerate(points):
        _, idx_nbs = tree.query(p, nb_count + 1)
        for idx_nb in idx_nbs[1:]:
            dist = np.linalg.norm([points[idx] * [1, 1, z], points[idx_nb] * [1, 1, z]])
            edge_list.append([idx, idx_nb, {"weight": int(dist)}])

    # create graph
    G = nx.Graph()
    G.add_edges_from(edge_list)

    return G


def pcd_to_subgraphs(pcd: o3d.geometry.PointCloud, n_drones: int, nb_count: int = 5, z: int = 1):
    G = pcd_to_graph(pcd, nb_count=nb_count, z=z)
    objval, partitioning = nxmetis.partition(G, n_drones)
    logger.debug(f"objective value: {objval:.2f}")

    subgraphs = []
    for d in range(n_drones):
        sg = nx.subgraph(G, partitioning[d])
        subgraphs.append(sg)
    return G, subgraphs


def graphs_to_geometries(pcd: o3d.geometry.PointCloud, graphs: list):
    colors = gen_colors(len(graphs))

    geos = []
    for i, g in enumerate(graphs):
        ls = o3d.geometry.LineSet()
        ls.points = pcd.points
        ls.lines = o3d.utility.Vector2iVector([e for e in g.edges()])
        ls.colors = o3d.utility.Vector3dVector([colors[i]] * g.number_of_edges())
        geos.append(ls)
    return geos


def paths_to_geometries(pcd: o3d.geometry.PointCloud, paths: list):
    colors = gen_colors(len(paths))

    geos = []
    for i, path in enumerate(paths):
        ls = o3d.geometry.LineSet()
        ls.points = pcd.points
        ls.lines = o3d.utility.Vector2iVector(path)
        ls.colors = o3d.utility.Vector3dVector([colors[i]] * len(path))
        geos.append(ls)

        start_end_points = [
            pcd.points[path[0][0]],
            pcd.points[path[-1][1]],
        ]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(start_end_points)
        pc.colors = o3d.utility.Vector3dVector([
            [1, 0, 0],
            [0, 1, 0]
        ])
        geos.append(pc)
    return geos


def flight_sequences_to_geometries(flight_sequences: np.array):
    colors = gen_colors(len(flight_sequences))

    geos = []
    for i, seq in enumerate(flight_sequences):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(seq)
        flight_path = [(i, i + 1) for i in range(len(seq) - 1)]
        ls.lines = o3d.utility.Vector2iVector(flight_path)
        ls.colors = o3d.utility.Vector3dVector([colors[i]] * len(flight_path))
        geos.append(ls)
    return geos


def charging_positions_to_geometries(charging_station_positions, scale: float = 1.0):
    res = []
    for pos in charging_station_positions:
        mesh_box = o3d.geometry.TriangleMesh.create_box(scale, scale, scale * 0.5)
        mesh_box.translate(pos)
        translation = [-scale / 2, -scale / 2, 0]
        mesh_box.translate(translation)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        res.append(mesh_box)
    return res


def closest_point(points: np.array, g: nx.Graph, x: list = [0, 0, 0]):
    """
    Returns the start node of the path for graph 'g' as index in the list of points
    The start node is the closest point to the given starting location
    """
    points = np.vstack([x, points])
    tree = KDTree(points)
    indices = tree.query(x, points.shape[0])[1][1:] - 1
    for idx in indices:
        if idx in g:
            return idx
    raise Exception("could not find closest point")


def get_next_node(points: np.array, cur_node: int, g: nx.Graph, sn_idx: int, z_penalty=1):
    pos_sn = points[sn_idx]

    # calculate cost to neighbors
    costs = []
    for nb in g.neighbors(cur_node):
        if not g.nodes[nb].get('visited', False):
            pos_src = points[cur_node]
            pos_dst = points[nb]
            cost = np.linalg.norm([pos_src * [1, 1, z_penalty], pos_dst * [1, 1, z_penalty]]) + np.linalg.norm(
                [pos_sn, pos_dst])
            costs.append((nb, cost))

    res = None

    if len(costs) == 0:
        # no remaining neighbors to visit, use BFS
        for cand in nx.bfs_tree(g, cur_node):
            if not g.nodes[cand].get('visited', False):
                res = cand
                break
    else:
        # closest neighbor
        res = sorted(costs, key=lambda v: v[1])[0][0]

    if res is None:
        # find node from different connected component
        _ = 1
        for n, dat in g.nodes(data=True):
            if not dat.get('visited', False):
                res = n
                break

    return res


def plan_path(pcd: o3d.geometry.PointCloud, g: nx.Graph, z_penalty=1):
    """
    Plans a path for the given subgraph 'sg', using the information from pointcloud 'pcd'
    """
    points = np.asarray(pcd.points)
    sn_idx = closest_point(points, g, [0, 0, 0])

    cur_node = sn_idx
    g.nodes[sn_idx]['visited'] = True
    seq = [cur_node]

    while len(seq) < g.number_of_nodes():
        next_node = get_next_node(points, cur_node, g, sn_idx, z_penalty=z_penalty)
        if next_node is not None:
            g.nodes[next_node]['visited'] = True
            seq.append(next_node)
            cur_node = next_node
        else:
            raise Exception("failed to find next node")
    path = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    return seq, path


def axis_angle_to_rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * theta))


def get_normal_vector(vertices, instance_angle, norm_vector_normed):
    normals = np.zeros((4, 3))
    x_product = np.cross((vertices[1] - vertices[0]), (vertices[3] - vertices[0]))
    direct_check = np.dot(x_product, norm_vector_normed)  # positive value refers to the correct side (direction)
    for i in range(4):
        if direct_check > 0:
            axis = vertices[(i + 1) % 4] - vertices[i]
        else:
            axis = vertices[i] - vertices[(i + 1) % 4]
        rot_m = axis_angle_to_rotation_matrix(axis, instance_angle)
        normals[i] = np.matmul(rot_m, norm_vector_normed)
    return normals


def optimize_path(seq: list, pcd: o3d.geometry.PointCloud, p_start, voxel_size, d_min, d_max, instance_angle, verbose=True):
    dimension = 3
    vertices_num = 4
    J = len(seq)
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)

    n = np.zeros((J, vertices_num, dimension))
    g = cvxpy.Variable((J, dimension))
    delta = cvxpy.Variable((J, dimension))

    # describe constraints
    #    initial constrain
    constrains_first = [
        g[0] == p_start + delta[0]
    ]
    # sequential update constrains
    constrains_seq = [
        g[j + 1] == g[j] + delta[j + 1] for j in range(J - 1)
    ]

    # distance constrains, min, max
    constrains_3 = [
        cvxpy.sum(cvxpy.multiply((g[j] - pcd_points[seq[j]]), pcd_normals[seq[j]])) - d_min >= 0 for j in range(J)
    ]

    constrains_4 = [
        d_max - cvxpy.sum(cvxpy.multiply((g[j] - pcd_points[seq[j]]), pcd_normals[seq[j]])) >= 0 for j in range(J)
    ]
    # instance angle constrains
    constrains_5 = []
    for j in range(J):
        normal_vec = pcd.normals[seq[j]]
        tangent_x = np.cross(normal_vec,
                             [normal_vec[0] * 3 + np.random.random(), normal_vec[1] * 5 + np.random.random(),
                              normal_vec[2] * 7 + np.random.random()])
        tangent_x = tangent_x / np.linalg.norm(tangent_x) * voxel_size / 2
        tangent_y = np.cross(normal_vec, tangent_x)
        tangent_y = tangent_y / np.linalg.norm(tangent_y) * voxel_size / 2

        rot_mat = axis_angle_to_rotation_matrix(normal_vec, np.pi)
        tangent_x_opposit = np.matmul(rot_mat, tangent_x)
        tangent_y_opposit = np.matmul(rot_mat, tangent_y)

        tangent_plane_vertex = []
        tangent_plane_vertex.append(pcd.points[seq[j]] + tangent_y + tangent_x)
        tangent_plane_vertex.append(pcd.points[seq[j]] + tangent_y + tangent_x_opposit)
        tangent_plane_vertex.append(pcd.points[seq[j]] + tangent_y_opposit + tangent_x_opposit)
        tangent_plane_vertex.append(pcd.points[seq[j]] + tangent_y_opposit + tangent_x)
        n[j] = get_normal_vector(tangent_plane_vertex, instance_angle,
                                 pcd.normals[seq[j]])
        constrains_5 += [
            cvxpy.sum(cvxpy.multiply((g[j] - tangent_plane_vertex[i]), n[j][i])) >= 0 for i in range(vertices_num)
        ]

    sum_norm2 = cvxpy.sum([cvxpy.norm(delta[j]) for j in range(J)])
    sum_diff_norm2 = cvxpy.sum([cvxpy.norm(delta[j + 1][:2] - delta[j][:2]) for j in range(J - 1)])
    sum_vert_diff_norm2 = 10 * cvxpy.sum([cvxpy.abs(delta[j + 1][2] - delta[j][2]) for j in range(J - 1)])
    prob = cvxpy.Problem(cvxpy.Minimize(sum_norm2 + sum_diff_norm2 + sum_vert_diff_norm2),
                         constrains_first + constrains_seq + constrains_3 + constrains_4 + constrains_5)
    # set solver option: https://www.cvxpy.org/tutorial/advanced/index.html
    prob.solve(solver="ECOS", verbose=verbose)
    return g.value


def get_scenario(seqs: list, charging_station_positions: list):
    return Scenario(seqs, charging_station_positions)


class ChargingStrategy(Enum):
    Milp = 0
    Naive = 1

    @classmethod
    def parse(cls, s):
        if s == "milp":
            return ChargingStrategy.Milp
        elif s == "naive":
            return ChargingStrategy.Naive
        return NotImplementedError()


def schedule_charge(seqs: list, charging_station_positions: list, params: Parameters, directory: str = None, strategy: ChargingStrategy = ChargingStrategy.Milp):
    """
    Schedules the charging for a sequence of flight waypoints and number of charging station positions
    """
    if directory:
        os.makedirs(directory, exist_ok=True)

    sc = Scenario(charging_station_positions, [seq.tolist() for seq in seqs])
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] # drones:               {sc.N_d}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] # stations:             {sc.N_s}")
    for d in range(sc.N_d):
        logger.debug(f" [{datetime.now().strftime('%H:%M:%S')}] # waypoints for UAV[{d}]: {len(seqs[d])}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] sigma:                  {params.sigma}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] W:                      {params.W}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] epsilon:                {params.epsilon}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] v:                      {params.v}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] r_charge:               {params.r_charge}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] r_deplete:              {params.r_deplete}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] B_start:                {params.B_start}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] B_min:                  {params.B_min}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Time limit :            {params.time_limit}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] IntFeasTol :            {params.int_feas_tol}")
    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] Rescheduling freq.:     {params.rescheduling_frequency}")

    if strategy == ChargingStrategy.Milp:
        strat = AfterNEventsStrategyAll(params.rescheduling_frequency)
        solver = SolverFactory("gurobi")
        solver.options['IntFeasTol'] = params.int_feas_tol
        solver.options['TimeLimit'] = params.time_limit
        scheduler = MilpScheduler(params, sc, solver=solver)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] prepared MILP simulator")
    elif strategy == ChargingStrategy.Naive:
        strat = OnEventStrategySingle()
        scheduler = NaiveScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        logger.debug("[{datetime.now().strftime('%H:%M:%S')}] prepared naive simulator")
    result = simulator.sim()

    if directory:
        with open(os.path.join(directory, "result.json"), 'w') as f:
            json.dump(jsons.dump(result), f)

    return result


def schedule_charge_from_conf(conf):
    co = conf["charging_optimization"]
    n_drones = conf['n_drones']
    B_min = [co["B_min"]] * n_drones
    B_max = [co["B_max"]] * n_drones
    B_start = [co["B_start"]] * n_drones
    v = [co["v"]] * n_drones
    r_charge = [co["r_charge"]] * n_drones
    r_deplete = [co["r_deplete"]] * n_drones
    epsilon = co.get("epsilon", None)
    plot_delta = co['plot_delta']
    W = co.get('W', None)
    sigma = co.get('sigma', None)
    charging_station_positions = co['charging_positions']
    time_limit = co['time_limit']
    int_feas_tol = co['int_feas_tol']
    rescheduling_frequency = co['rescheduling_frequency']
    output_dir = conf['output_directory']

    flight_sequence_fpath = conf['flight_sequence_fpath']
    with open(flight_sequence_fpath, 'rb') as f:
        flight_sequences = pickle.load(f)

    # add intermediate positions
    dist_cuttoffs = []
    for seq in flight_sequences:
        distances = []
        for i in range(len(seq)-1):
            pos_i = seq[i]
            pos_j = seq[i + 1]
            distance = dist3(pos_i, pos_j)
            distances.append(distance)
        dist_cuttoffs.append(np.mean(distances) * 3)

    flight_sequences_segmented = []
    for d, seq in enumerate(flight_sequences):
        seq_padded = []
        for i in range(len(seq)-1):
            pos_i = seq[i]
            pos_j = seq[i + 1]
            distance = dist3(pos_i, pos_j)
            if distance > dist_cuttoffs[d]:
                # segmentation needed
                nr_pads = int(np.ceil(distance / dist_cuttoffs[d]))
                logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] for UAV [{d}] splitting segment in {nr_pads} parts")
                for i_pad in range(nr_pads):
                    frac = (i_pad + 1) / nr_pads
                    pos_padded = pos_i + frac*(pos_j - pos_i)
                    seq_padded.append(pos_padded)
            else:
                # no padding needed
                seq_padded.append(pos_i)
        seq_padded.append(pos_j)  # add last waypoint
        flight_sequences_segmented.append(np.array(seq_padded))

    logger.debug(f"[{datetime.now().strftime('%H:%M:%S')}] starting charge scheduling..")
    params = Parameters(
        v=v,
        r_charge=r_charge,
        r_deplete=r_deplete,
        B_start=B_start,
        B_min=B_min,
        B_max=B_max,
        epsilon=epsilon,
        plot_delta=plot_delta,
        W=W,
        sigma=sigma,
        time_limit=time_limit,
        int_feas_tol=int_feas_tol,
        rescheduling_frequency=rescheduling_frequency
    )
    strategy = ChargingStrategy.parse(conf['charging_strategy'])
    t_start = time.perf_counter()
    _ = schedule_charge(flight_sequences, charging_station_positions, params, directory=output_dir, strategy=strategy)
    elapsed = time.perf_counter() - t_start
    logger.debug(f"finished charge schedule simulation in {elapsed:.1f}s")


def convert_schedule_to_flight_sequence(schedule):
    return np.array([e.node.pos for e in schedule if e.type in [EventType.reached, EventType.started, EventType.changed_course]])


def remove_downward_normals(pcd: o3d.geometry.PointCloud, threshold=-0.5):
    """
    Removes the points from the point cloud that point downwards.
    The threshold parameter defines the value of the z-direction of a normal under which a point is to be removed.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    ids = normals[:, 2] > threshold

    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(points[ids])
    res.normals = o3d.utility.Vector3dVector(normals[ids])
    return res
