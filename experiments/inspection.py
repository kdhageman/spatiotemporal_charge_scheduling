import glob
import os
import sys
from enum import Enum

from simulate.event import EventType
from simulate.scheduling import MilpScheduler, NaiveScheduler
from simulate.simulate import Simulator
from simulate.strategy import IntervalStrategy, ArrivalStrategy

sys.path.append('drone_charging')

from simulate.parameters import Parameters
import open3d as o3d
import logging
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import nxmetis
import cvxpy as cp
from numpy import cross, eye, random
from scipy.linalg import expm, norm
from scipy.spatial import KDTree
from tqdm import tqdm
from util.scenario import Scenario

logging.basicConfig(
    level=logging.DEBUG
)
logging.getLogger("pyomo.core").setLevel(logging.WARN)
logging.getLogger("pyomo.opt").setLevel(logging.WARN)
logging.getLogger("matplotlib").setLevel(logging.WARN)
logger = logging.getLogger(__name__)


# for davinci
# camera_x = 0
# camera_y = -300

def timed(func):
    def wrapper(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - t_start
        return elapsed, res

    return wrapper


def gen_colors(n):
    np.random.seed(0)
    res = []
    for d in range(n):
        c = np.random.rand(3).tolist()
        res.append(c)
    return res


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


def draw_geometries(geometries: list, x: float, y: float, fname: str = None, **kwargs):
    vis = o3d.visualization.Visualizer()
    vis.create_window(**kwargs)
    for g in geometries:
        vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.rotate(x, y)
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
    raise Exception("Could not find closests point")


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
    return expm(cross(eye(3), axis / norm(axis) * theta))


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


def optimize_path(seq: list, pcd: o3d.geometry.PointCloud, p_start, voxel_size, d_min, d_max, instance_angle,
                  verbose=True):
    dimension = 3
    vertices_num = 4
    J = len(seq)
    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)

    n = np.zeros((J, vertices_num, dimension))
    g = cp.Variable((J, dimension))
    delta = cp.Variable((J, dimension))

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
        cp.sum(cp.multiply((g[j] - pcd_points[seq[j]]), pcd_normals[seq[j]])) - d_min >= 0 for j in range(J)
    ]

    constrains_4 = [
        d_max - cp.sum(cp.multiply((g[j] - pcd_points[seq[j]]), pcd_normals[seq[j]])) >= 0 for j in range(J)
    ]
    # instance angle constrains
    constrains_5 = []
    for j in range(J):
        normal_vec = pcd.normals[seq[j]]
        tangent_x = np.cross(normal_vec,
                             [normal_vec[0] * 3 + random.random(), normal_vec[1] * 5 + random.random(),
                              normal_vec[2] * 7 + random.random()])
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
            cp.sum(cp.multiply((g[j] - tangent_plane_vertex[i]), n[j][i])) >= 0 for i in range(vertices_num)
        ]

    sum_norm2 = cp.sum([cp.norm(delta[j]) for j in range(J)])
    sum_diff_norm2 = cp.sum([cp.norm(delta[j + 1][:2] - delta[j][:2]) for j in range(J - 1)])
    sum_vert_diff_norm2 = 10 * cp.sum([cp.abs(delta[j + 1][2] - delta[j][2]) for j in range(J - 1)])
    prob = cp.Problem(cp.Minimize(sum_norm2 + sum_diff_norm2 + sum_vert_diff_norm2),
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


def schedule_charge(seqs: list, charging_station_positions: list, params: Parameters,
                    directory: str = None, strategy: ChargingStrategy = ChargingStrategy.Milp):
    """
    Schedules the charging for a sequence of flight waypoints and number of charging station positions
    """
    sc = Scenario(charging_station_positions, [seq.tolist() for seq in seqs])
    logger.debug(f"# drones:    {sc.N_d}")
    logger.debug(f"# stations:  {sc.N_s}")
    logger.debug(f"# waypoints: {sc.N_w}")
    logger.debug(f"W:           {params.W}")
    logger.debug(f"sigma:       {params.sigma}")
    if strategy == ChargingStrategy.Milp:
        strat = IntervalStrategy(params.schedule_delta)
        scheduler = MilpScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        logger.debug("prepared MILP simulator")
    elif strategy == ChargingStrategy.Naive:
        strat = ArrivalStrategy()
        scheduler = NaiveScheduler(params, sc)
        simulator = Simulator(scheduler, strat, params, sc, directory=directory)
        logger.debug("prepared naive simulator")
    solve_times, env, events = simulator.sim()

    # write solve times to disk
    if directory:
        with open(os.path.join(directory, 'solve_times.csv'), 'w') as f:
            f.write("iteration, solve_time\n")
            for i, t in enumerate(solve_times):
                f.write(f"{i}, {t}\n")

        # write mission execution time to disk
        with open(os.path.join(directory, "execution_time.txt"), 'w') as f:
            f.write(f"{env.now}")

    return env, events


def convert_schedule_to_flight_sequence(schedule):
    return np.array([e.node.pos for e in schedule if e.name in [EventType.reached, EventType.started, EventType.changed_course]])


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


def run_from_conf(conf):
    # general parameters
    g = conf['general']
    n_drones = g["n_drones"]
    nb_count = g["nb_count"]
    voxel_size = g["voxel_size"]

    # path planning
    pp = conf["path_planning"]
    p_start = pp["p_start"]
    d_min = pp["d_min"]
    d_max = pp["d_max"]
    instance_angle = np.pi / 6

    # charging optimization
    co = conf["charging_optimization"]
    B_min = [co["B_min"]] * n_drones
    B_max = [co["B_max"]] * n_drones
    B_start = [co["B_start"]] * n_drones
    v = [co["v"]] * n_drones
    r_charge = [co["r_charge"]] * n_drones
    r_deplete = [co["r_deplete"]] * n_drones
    epsilon = co.get("epsilon", None)
    schedule_delta = co.get('schedule_delta', None)
    plot_delta = co['plot_delta']
    W = co.get('W', None)
    sigma = co.get('sigma', None)
    charging_station_positions = co['charging_positions']

    # open3d config
    o = conf['open3d']
    camera_x = o['camera_x']
    camera_y = o['camera_y']
    open3d_width = o['width']
    open3d_height = o['height']

    # options to change
    visualize = conf['visualize']
    do_remove_downward_normals = conf['do_remove_downward_normals']

    # prepare output directory
    output_dir = conf['output_directory']
    os.makedirs(output_dir, exist_ok=True)
    for fname in glob.glob(os.path.join(output_dir, "?_*.png")):
        os.remove(fname)
    os.makedirs(os.path.join(output_dir, "simulation"), exist_ok=True)
    filecounter = 1

    # starting here
    fpath_pcd = conf['pcd_path']
    fpath_mesh = conf['mesh_path']

    pcd = o3d.io.read_point_cloud(fpath_pcd)
    logger.debug(f"finished loading pcd from file ({len(pcd.points):,} points)")

    if fpath_mesh:
        mesh = o3d.io.read_triangle_mesh(fpath_mesh)
        logger.debug("finished loading mesh from file")
    else:
        mesh = None

    # move pointcloud to align with (0,0,0)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [0, 0, 0]
    translation_offset = -aabb.min_bound
    pcd.translate(translation_offset)
    if mesh:
        mesh.translate(translation_offset)
    logger.debug("finished translating pcd")

    aabb.color = [0, 0, 0]
    aabb = pcd.get_axis_aligned_bounding_box()
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    fname = os.path.join(output_dir, f"{filecounter}_pcd_translation.png") if not visualize else None
    draw_geometries([pcd, aabb, cf], camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    if mesh:
        geos = [aabb, cf]
        geos.append(mesh)
        fname = os.path.join(output_dir, f"{filecounter}_inner_mesh.png") if not visualize else None
        draw_geometries(geos, camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
        filecounter += 1

    # remove any point too close to the ground
    points = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(points[points[:, 2] > 0.3])
    logger.debug(f"finished reducing pcd to be sufficiently above ground ({len(pcd.points):,} points)")
    geos = [pcd]
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd_above_ground.png") if not visualize else None
    draw_geometries(geos, camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    t_downsample, pcd = downsample(pcd, 0.1, False)
    t_estimate_normals, pcd = estimate_normal(pcd)
    logger.debug("finished estimating normals")
    geos = [pcd]
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd.png") if not visualize else None
    draw_geometries(geos, camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    t_subsampled, pcd = downsample(pcd, voxel_size, True)
    geos = [pcd]
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_downsampled.png") if not visualize else None
    draw_geometries(geos, camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1
    logger.debug(f"finished downsampling (got {len(pcd.points):,} remaining points)")

    if do_remove_downward_normals:
        pcd = remove_downward_normals(pcd, threshold=-0.7)
        geos = [pcd]
        if mesh:
            geos.append(mesh)
        fname = os.path.join(output_dir, f"{filecounter}_no_downward.png") if not visualize else None
        draw_geometries(geos, camera_x, camera_y, fname=fname, width=open3d_width, height=open3d_height)
        filecounter += 1
        logger.debug("finished downward normal removal")
    else:
        logger.debug("skipped downward normal removal")

    G, subgraphs = pcd_to_subgraphs(pcd, n_drones, nb_count=nb_count, z=3)
    logger.debug("finished graph extraction")

    geos = graphs_to_geometries(pcd, [G])
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_graph.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1
    logger.debug("finished geometry extraction from graphs")

    geos = graphs_to_geometries(pcd, subgraphs)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_subgraphs.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1
    logger.debug("finished geometry extraction from subgraphs")

    z_penalty = 10
    seqs = []
    paths = []
    for i, sg in enumerate(subgraphs):
        seq, path = plan_path(pcd, sg, z_penalty=z_penalty)
        seqs.append(seq)
        paths.append(path)
        logger.debug(f"finished path planning for subgraph [{i}]")
    logger.debug("finished path finding")

    geos = paths_to_geometries(pcd, paths)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_paths.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    flight_sequences = []
    for i, seq in enumerate(tqdm(seqs, desc='optimize paths')):
        flight_sequence = optimize_path(seq, pcd, p_start, voxel_size, d_min, d_max, instance_angle, verbose=False)
        flight_sequences.append(flight_sequence)
    logger.debug("finished path optimization")

    geos = flight_sequences_to_geometries(flight_sequences)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_flight_paths.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    geos += charging_positions_to_geometries(charging_station_positions)
    fname = os.path.join(output_dir,
                         f"{filecounter}_flight_paths_incl_charging_stations.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
    filecounter += 1

    logger.debug("starting charge scheduling..")
    params = Parameters(
        v=v,
        r_charge=r_charge,
        r_deplete=r_deplete,
        B_start=B_start,
        B_min=B_min,
        B_max=B_max,
        epsilon=epsilon,
        plot_delta=plot_delta,
        schedule_delta=schedule_delta,
        W=W,
        sigma=sigma,
    )
    strategy = ChargingStrategy.parse(conf['charging_strategy'])
    _, schedules = schedule_charge(flight_sequences, charging_station_positions, params,
                                   directory=os.path.join(output_dir, "simulation"), strategy=strategy)
    logger.debug("finished charge scheduling")

    # convert schedules to flight sequence for plotting
    flight_sequences = [convert_schedule_to_flight_sequence(s) for s in schedules]
    geos = flight_sequences_to_geometries(flight_sequences)
    geos += charging_positions_to_geometries(charging_station_positions, 0.3)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_flight_paths_incl_charging.png") if not visualize else None
    draw_geometries(geos, x=camera_x, y=camera_y, fname=fname, width=open3d_width, height=open3d_height)
