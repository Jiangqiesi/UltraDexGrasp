import numpy as np


def add_gaussian_noise(pc, sigma=0.02):
    # pc should be (n, 3)
    num_points = pc.shape[0]
    multiplicative_noise = (1 + np.random.randn(num_points)[:, None] * sigma)  # (n, 1)
    return pc * multiplicative_noise

from plyfile import PlyData, PlyElement
def save_pc_as_ply(pc, path):
    have_rgb = True if pc.shape[1] == 6 else False
    xyz = pc[:, :3]
    if have_rgb:
        rgb = (pc[:, 3:] * 255).astype(np.uint8)
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    else:
        vertex_data = np.empty(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    if have_rgb:
        vertex_data['r'] = rgb[:, 0]
        vertex_data['g'] = rgb[:, 1]
        vertex_data['b'] = rgb[:, 2]
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def crop_point_cloud(point_cloud, boundary):
    cropped_point_cloud = point_cloud[
        (point_cloud[:, 0] >= boundary[0, 0]) & (point_cloud[:, 0] <= boundary[0, 1]) &
        (point_cloud[:, 1] >= boundary[1, 0]) & (point_cloud[:, 1] <= boundary[1, 1]) &
        (point_cloud[:, 2] >= boundary[2, 0]) & (point_cloud[:, 2] <= boundary[2, 1])
    ]

    return cropped_point_cloud
