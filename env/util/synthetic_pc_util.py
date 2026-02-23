import os
import sapien
import numpy as np
import torch
import pytorch3d.ops as pytorch3d_ops

from plyfile import PlyData, PlyElement
def save_pc_as_ply(pc, path):
    have_rgb = True if pc.shape[1] == 6 else False
    xyz = pc[:, :3]
    if have_rgb:
        rgb = (pc[:, 3:] * 255).astype(np.uint8)  # RGB 转换为整数
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

class SyntheticPC:
    def __init__(self, urdf_path, image_size=[256, 256]):
        self.setup_scene(image_size=image_size, timestep=1/180)
        self.load_robot(urdf_path)
        self.scene.update_render()

        self.synthetic_table_pc = self.get_synthetic_table_pc()

    def get_synthetic_table_pc(self):
        nx = 40 * 2
        ny = 120 * 2
        
        x = np.linspace(0.5 - 0.01, 0.7 + 0.01, nx)
        y = np.linspace(-0.4 - 0.01, 0.4 + 0.01, ny)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        crop_mask = (points[:, 0] > 0.5 + 0.01) & (points[:, 0] < 0.7 - 0.01) & (points[:, 1] > -0.4 + 0.01) & (points[:, 1] < 0.4 - 0.01) & ((points[:, 1] > 0.01) | (points[:, 1] < -0.01))
        table_pc = points[~crop_mask]
        table_pc = np.concatenate([table_pc, np.zeros((table_pc.shape[0], 3))], axis=-1)

        return table_pc

    def setup_scene(self, image_size, timestep, ray_tracing=True):
        self.scene = sapien.Scene()
        self.scene.set_timestep(timestep)
        self.scene.default_physical_material = self.scene.create_physical_material(1, 1, 0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.setup_camera(image_size)
        if ray_tracing:
            sapien.render.set_camera_shader_dir('rt')
            sapien.render.set_viewer_shader_dir('rt')
            sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser('optix') # change to 'optix' or 'oidn'

    def load_robot(self, urdf_path):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.active_joints = self.robot.get_active_joints()
        self.joint_name_list = [joint.get_name() for joint in self.active_joints]
        # print(self.joint_name_list)
        # print([link.get_name() for link in self.robot.get_links()])
        for link in self.robot.get_links():
            if link.get_name() == 'base':
                self.end_effector = link
                break
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=100, force_limit=100, mode='force')
            joint.set_friction(0.0)
        self.init_qpos = [0.0] * len(self.active_joints)
        self.robot.set_qpos(self.init_qpos)
        for link in self.robot.links:
            link.disable_gravity = True

    def setup_camera(self, image_size, fov=40, near=0.1, far=10.0):
        width = image_size[0]
        height = image_size[1]
        self.cameras = []

        cam_pos = np.array([1.4, 1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_0 = self.scene.add_camera(
            name='camera_0',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_0.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_0)

        cam_pos = np.array([1.4, -1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_1 = self.scene.add_camera(
            name='camera_1',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_1.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_1)

        cam_pos = np.array([-0.6, -1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_2 = self.scene.add_camera(
            name='camera_2',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_2.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_2)

        cam_pos = np.array([-0.6, 1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_3 = self.scene.add_camera(
            name='camera_3',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_3.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_3)

        cam_pos = np.array([1.4, 1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_4 = self.scene.add_camera(
            name='camera_4',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_4.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_4)

        cam_pos = np.array([1.4, -1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_5 = self.scene.add_camera(
            name='camera_5',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_5.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_5)

        cam_pos = np.array([-0.6, -1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_6 = self.scene.add_camera(
            name='camera_6',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_6.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_6)

        cam_pos = np.array([-0.6, 1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_7 = self.scene.add_camera(
            name='camera_7',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_7.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_7)

    def get_ee_pose(self):
        eef_pose = self.end_effector.get_pose()
        return np.concatenate([eef_pose.p, eef_pose.q])

    def get_qpos(self):
        return self.robot.get_qpos()

    def get_pc(self, num_point=None):
        pc_list = []
        for camera in self.cameras:
            camera.take_picture()
            rgb = camera.get_picture('Color')[:,:,:3]
            position = camera.get_picture('Position')

            # segment robot
            seg_labels = camera.get_picture('Segmentation')  # [H, W, 4]
            label_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            points_opengl = position[..., :3][(position[..., 3] < 1) & (label_image > 0)]
            points_color = rgb[(position[..., 3] < 1) & (label_image > 0)]

            model_matrix = camera.get_model_matrix()
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            points_color = np.clip(points_color, 0, 1)
            pc_list.append(np.concatenate([points_world, points_color], axis=-1))

        pc = np.concatenate(pc_list, axis=0)

        if num_point is not None:
            _, fps_idx = pytorch3d_ops.sample_farthest_points(points=torch.from_numpy(pc[:,:3]).cuda().unsqueeze(0), K=num_point)
            pc = pc[fps_idx[0].cpu().numpy()]

        return pc

    def get_pc_at_qpos(self, qpos, num_point=None):
        self.robot.set_qpos(qpos)
        self.scene.update_render()
        pc = self.get_pc(num_point)

        return pc
