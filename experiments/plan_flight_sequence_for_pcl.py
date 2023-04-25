import copy
import glob
import json
import logging
import os
import sys

import networkx as nx
import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

from util_funcs import draw_geometries, downsample, estimate_normal, remove_downward_normals, pcd_to_subgraphs, graphs_to_geometries, plan_path, paths_to_geometries, optimize_path, flight_sequences_to_geometries, charging_positions_to_geometries, \
    extract_navigation_graphs, graph_to_pcd

logger = logging.getLogger(__name__)


def main(conf):
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

    # open3d config
    o = conf['open3d']
    open3d_width = o['width']
    open3d_height = o['height']
    open3d_viewpoint_file = o.get('viewpoint_file', None)
    if open3d_viewpoint_file:
        viewpoint = o3d.io.read_pinhole_camera_parameters(open3d_viewpoint_file)
    else:
        viewpoint = None

    # options to change
    visualize = conf['visualize']
    do_remove_downward_normals = conf['do_remove_downward_normals']
    charging_stations_positions = conf.get('charging_stations', [])
    if not charging_stations_positions:
        charging_stations_positions = []

    # prepare output directory
    output_dir = conf['output_directory']
    os.makedirs(output_dir, exist_ok=True)
    for fname in glob.glob(os.path.join(output_dir, "?_*.png")):
        os.remove(fname)
    filecounter = 1

    # starting here
    fpath_pcd = conf['pcd_path']
    fpath_mesh = conf['mesh_path']

    pcd = o3d.io.read_point_cloud(fpath_pcd)
    original_pcd = pcd
    logger.debug(f"finished loading pcd from file ({len(pcd.points):,} points)")

    if fpath_mesh:
        mesh = o3d.io.read_triangle_mesh(fpath_mesh)
        logger.debug("finished loading mesh from file")
    else:
        mesh = None

    # cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    # cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0, origin=[0, 0, 0])
    cf = None
    geos = [pcd]
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd_pretranslation.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='pre translation')
    filecounter += 1

    offset = conf['pcd_offset']
    pcd.translate(offset)
    if mesh:
        mesh.translate(offset)

    geos = [pcd]
    pcd_posttrans = copy.deepcopy(pcd)
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd_posttranslation.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='post translation')
    filecounter += 1

    # remove any point too close to the ground
    points = np.asarray(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(points[points[:, 2] > 2])
    logger.debug(f"finished reducing pcd to be sufficiently above ground ({len(pcd.points):,} points)")
    geos = [pcd]
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd_above_ground.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='pcd above ground')
    filecounter += 1

    t_estimate_normals, pcd = estimate_normal(pcd)
    logger.debug("finished estimating normals")
    geos = [pcd]
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(cf)
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_pcd.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='normal estimation')
    filecounter += 1

    t_subsampled, pcd = downsample(pcd, voxel_size, True)
    geos = [pcd]
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_downsampled.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='down sampled')
    filecounter += 1
    logger.debug(f"finished downsampling (got {len(pcd.points):,} remaining points)")

    if do_remove_downward_normals:
        pcd = remove_downward_normals(pcd, threshold=-0.7)
        geos = [pcd]
        if cf:
            geos.append(cf)
        if mesh:
            geos.append(mesh)
        fname = os.path.join(output_dir, f"{filecounter}_no_downward.png") if not visualize else None
        draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='without downward normals')
        filecounter += 1
        logger.debug("finished downward normal removal")
    else:
        logger.debug("skipped downward normal removal")

    # log how large the inspected structure would be
    obb = pcd.get_oriented_bounding_box()
    logger.debug(f"bounding box size: {obb.extent}")

    G, subgraphs = pcd_to_subgraphs(pcd, n_drones, nb_count=nb_count, z=1.2)
    logger.debug("finished graph extraction")

    geos = graphs_to_geometries(pcd, [G])
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_graph.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='graph')
    filecounter += 1
    logger.debug("finished geometry extraction from graphs")

    geos = graphs_to_geometries(pcd, subgraphs)
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_subgraphs.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='subgraphs')
    filecounter += 1
    logger.debug("finished geometry extraction from subgraphs")

    # extract navigation graphs
    navigation_graphs = extract_navigation_graphs(G, charging_stations_positions, 2)

    for i, navigation_graph in enumerate(navigation_graphs):
        geos = charging_positions_to_geometries(charging_stations_positions, scale=1)
        geos.append(pcd_posttrans)

        pcd_graph = graph_to_pcd(navigation_graph)
        geos += graphs_to_geometries(pcd_graph, [navigation_graph], color_offset=i)
        if cf:
            geos.append(cf)
        if mesh:
            geos.append(mesh)

        fname = os.path.join(output_dir, f"{filecounter}_navigationgraphs_{i}.png") if not visualize else None
        draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name=f'navigation graphs ({i})')
        filecounter += 1
    logger.debug("finished navigation graph extraction from graphs")

    z_penalty = 10
    seqs = []
    paths = []
    for i, sg in enumerate(subgraphs):
        seq, path = plan_path(pcd, sg, G, z_penalty=z_penalty, start_position=p_start[i])
        seqs.append(seq)
        paths.append(path)
        logger.debug(f"finished path planning for subgraph [{i}]")
    logger.debug("finished path finding")

    geos = paths_to_geometries(pcd, paths)
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_paths.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='paths')
    filecounter += 1

    flight_sequences = []
    for i, seq in enumerate(tqdm(seqs, desc='optimize paths')):
        flight_sequence = optimize_path(seq, pcd, p_start[i], voxel_size, d_min, d_max, instance_angle, verbose=False)
        flight_sequences.append(flight_sequence)
    logger.debug("finished path optimization")

    for i, seq in enumerate(flight_sequences):
        if seq is None:
            logger.fatal(f"failed to optimize path for UAV [{i}]")
        logger.info(f"optimized path for UAV [{i}] with {len(flight_sequences[i])} waypoints")

    geos = flight_sequences_to_geometries(flight_sequences)
    if cf:
        geos.append(cf)
    if mesh:
        geos.append(mesh)
    fname = os.path.join(output_dir, f"{filecounter}_flight_paths.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='flight paths')
    filecounter += 1

    # draw flight paths with charging stations + point cloud
    charging_station_geos = charging_positions_to_geometries(charging_stations_positions, scale=1)
    geos.append(pcd_posttrans)
    geos += charging_station_geos
    fname = os.path.join(output_dir, f"{filecounter}_flight_paths_incl_cs.png") if not visualize else None
    draw_geometries(original_pcd, geos, viewpoint=viewpoint, fname=fname, width=open3d_width, height=open3d_height, window_name='flight paths incl charging stations')
    filecounter += 1

    fname = os.path.join(output_dir, "flight_sequences.json")
    with open(fname, 'w') as f:
        json.dump([x.tolist() for x in flight_sequences], f)

    navigation_graphs_as_json = []
    for navigation_graph in navigation_graphs:
        graph_as_json = nx.adjacency_data(navigation_graph)
        navigation_graphs_as_json.append(graph_as_json)

    fname = os.path.join(output_dir, "navigation_graphs.json")
    with open(fname, 'w') as f:
        json.dump(navigation_graphs_as_json, f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must provide path to configuration file as parameter")
        sys.exit(1)
    fpath_conf = sys.argv[1]
    with open(fpath_conf, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    main(conf)
