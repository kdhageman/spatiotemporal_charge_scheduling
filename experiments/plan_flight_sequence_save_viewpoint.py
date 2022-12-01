import sys
import open3d as o3d
import yaml


def main(conf):
    fpath_pcd = conf['pcd_path']
    fpath_out = conf['open3d']['viewpoint_file']
    pcd_offset = conf['pcd_offset']
    opend3d_height = conf['open3d']['height']
    opend3d_width = conf['open3d']['width']

    pcd = o3d.io.read_point_cloud(fpath_pcd)
    pcd.translate(pcd_offset)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=opend3d_width, height=opend3d_height)
    vis.add_geometry(pcd)
    vis.run()

    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(fpath_out, params)
    vis.destroy_window()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Must provide path to configuration file as parameter")
        sys.exit(1)
    fpath_conf = sys.argv[1]
    with open(fpath_conf, 'r') as f:
        conf = yaml.load(f, Loader=yaml.Loader)

    main(conf)
