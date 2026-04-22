import os
import sapien
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
from pytorch3d.ops import sample_farthest_points
from curobo.types.robot import RobotConfig
from curobo.util_file import get_assets_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.types import WorldConfig

from env.util.synthetic_pc_util import SyntheticPC
from env.util.point_cloud_util import save_pc_as_ply, crop_point_cloud
from env.util.util import calculate_angle_between_quat


UR5E_LEFT_HOME_JOINT = np.array(
    [
        -np.pi / 2,
        -np.pi * 9 / 16,
        -np.pi * 7 / 16,
        np.pi * 7 / 8,
        -np.pi / 2,
        -np.pi / 4,
    ]
)
UR5E_LEFT_JOINT_LIMIT = [
    [UR5E_LEFT_HOME_JOINT[0] - np.pi / 4, UR5E_LEFT_HOME_JOINT[0] + np.pi / 4],
    [UR5E_LEFT_HOME_JOINT[1] - np.pi / 4, UR5E_LEFT_HOME_JOINT[1] + np.pi / 8],
    [UR5E_LEFT_HOME_JOINT[2] - np.pi / 2, UR5E_LEFT_HOME_JOINT[2] + np.pi / 2],
    [UR5E_LEFT_HOME_JOINT[3] - np.pi / 4, UR5E_LEFT_HOME_JOINT[3] + np.pi * 3 / 4],
    [UR5E_LEFT_HOME_JOINT[4] - np.pi / 2, UR5E_LEFT_HOME_JOINT[4] + np.pi / 2],
    [-np.pi, np.pi],
]
UR5E_RIGHT_HOME_JOINT = np.array(
    [np.pi / 2, -np.pi * 7 / 16, np.pi * 7 / 16, np.pi / 8, np.pi / 2, np.pi / 4]
)
UR5E_RIGHT_JOINT_LIMIT = [
    [UR5E_RIGHT_HOME_JOINT[0] - np.pi / 4, UR5E_RIGHT_HOME_JOINT[0] + np.pi / 4],
    [UR5E_RIGHT_HOME_JOINT[1] - np.pi / 8, UR5E_RIGHT_HOME_JOINT[1] + np.pi / 4],
    [UR5E_RIGHT_HOME_JOINT[2] - np.pi / 2, UR5E_RIGHT_HOME_JOINT[2] + np.pi / 2],
    [UR5E_RIGHT_HOME_JOINT[3] - np.pi * 3 / 4, UR5E_RIGHT_HOME_JOINT[3] + np.pi / 4],
    [UR5E_RIGHT_HOME_JOINT[4] - np.pi / 2, UR5E_RIGHT_HOME_JOINT[4] + np.pi / 2],
    [-np.pi, np.pi],
]

XHAND_LEFT_HOME_JOINT = (
    np.array([20, 20, 20, 5, 20, 20, 20, 20, 20, 20, 20, 20]) * np.pi / 180
)
XHAND_RIGHT_HOME_JOINT = (
    np.array([20, 20, 20, 5, 20, 20, 20, 20, 20, 20, 20, 20]) * np.pi / 180
)
XHAND_LEFT_DEFAULT_JOINT_ORDER = [
    "left_hand_thumb_bend_joint",
    "left_hand_thumb_rota_joint1",
    "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint",
    "left_hand_index_joint1",
    "left_hand_index_joint2",
    "left_hand_mid_joint1",
    "left_hand_mid_joint2",
    "left_hand_ring_joint1",
    "left_hand_ring_joint2",
    "left_hand_pinky_joint1",
    "left_hand_pinky_joint2",
]
XHAND_RIGHT_DEFAULT_JOINT_ORDER = [
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
]


class BaseEnv:
    def __init__(
        self,
        config,
        with_object=True,
        control_hz=20,
        timestep=1 / 240,
        ray_tracing=False,
    ):
        self.control_hz = control_hz
        self.timestep = timestep
        self.frame_skip = int(1 / self.timestep / self.control_hz)

        self.config = config
        self.obs_type = config["obs_type"]
        self.with_object = with_object

        self.table_height = 0.714
        self.set_up_physics_and_render(ray_tracing)
        self.set_up_scene()
        self.grasp_qpos = None

        self.scene.update_render()

    def set_up_physics_and_render(self, ray_tracing):
        if ray_tracing:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(
                16
            )  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser(
                "optix"
            )  # change to 'optix' or 'oidn'

        sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
        sapien.physx.set_body_config(
            solver_position_iterations=25,
            solver_velocity_iterations=1,
            sleep_threshold=0.005,
        )
        sapien.physx.set_scene_config(
            gravity=np.array([0.0, 0.0, -9.81]),
            bounce_threshold=2.0,
            enable_pcm=True,
            enable_tgs=True,
            enable_ccd=False,
            enable_enhanced_determinism=False,
            enable_friction_every_iteration=True,
            cpu_workers=0,
        )
        sapien.physx.set_default_material(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        )
        self.scene = sapien.Scene(
            [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()]
        )
        self.scene.set_timestep(self.timestep)
        # self.scene.add_ground(altitude=0.0)

    def set_up_scene(self):
        self.set_up_light()
        self.set_up_table()
        self.set_up_robot()
        self.set_up_object()
        self.set_up_visual_boundary()
        self.set_up_camera()

    def set_up_light(self):
        self.scene.set_ambient_light([1.0, 1.0, 1.0])
        # self.scene.add_directional_light([-1.5, 1.5, -3.5], [0.7, 0.7, 0.7], shadow=False)
        self.scene.add_point_light(
            [2, 2, 4 + self.table_height], [1.0, 1.0, 1.0], shadow=True
        )
        self.scene.add_point_light(
            [2, -2, 4 + self.table_height], [1.0, 1.0, 1.0], shadow=True
        )
        # self.scene.add_point_light([3, 0, 3.5 + self.table_height], [1, 1, 1], shadow=True)

    def set_up_table(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[0.6, 0.8, 0.03],
            material=sapien.physx.PhysxMaterial(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
        )
        builder.add_box_visual(
            half_size=[0.6, 0.8, 0.03], material=[180 / 255, 170 / 255, 160 / 255]
        )
        table = builder.build_kinematic(name="table")
        table.set_pose(sapien.Pose([0.3, 0.0, self.table_height - 0.03]))
        # table.set_pose(sapien.Pose([0.5, 0.0, self.table_height - 0.03 - 0.01]))

        # # create table from mesh
        # builder = self.scene.create_actor_builder()
        # file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'asset/scene/dining_table.glb')
        # scale = [0.9, 1.0, 1.2]
        # builder.add_nonconvex_collision_from_file(
        #     filename=file_path,
        #     scale=scale,
        #     pose=sapien.Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        # )
        # builder.add_visual_from_file(
        #     filename=file_path,
        #     scale=scale,
        #     pose=sapien.Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        # )
        # table = builder.build_kinematic(name='table')
        # table.set_pose(
        #     sapien.Pose(
        #         p=[0.5, 0, -0.64],
        #         q=euler2quat(np.pi/2, 0.0, np.pi/2)
        #     )
        # )

    def set_up_robot(self):
        self.hand_dof = self.config["robot"]["ur5e_with_left_hand"]["hand_dof"]

        # robot left
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot_left = loader.load(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_left_hand']['urdf_path']}"
        )
        self.robot_left.set_name("robot_left")
        self.active_joints_left = self.robot_left.get_active_joints()
        self.LEFT_HAND_SIM_JOINT_ORDER = np.array(
            [joint.get_name() for joint in self.active_joints_left]
        )[-self.hand_dof :]
        for link in self.robot_left.get_links():
            if link.get_name() == "base":
                self.end_effector_left = link
                break
        for joint in self.active_joints_left:
            joint.set_drive_property(
                stiffness=1000, damping=100, force_limit=1e10, mode="force"
            )
            joint.set_friction(0.0)
        for link in self.robot_left.links:
            link.disable_gravity = True
        self.UR5E_LEFT_HOME_JOINT = UR5E_LEFT_HOME_JOINT
        if self.config["robot"]["ur5e_with_left_hand"]["hand_type"] == "xhand":
            self.LEFT_HAND_DEFAULT_JOINT_ORDER = XHAND_LEFT_DEFAULT_JOINT_ORDER
            self.LEFT_HAND_DEFAULT_2_SIM_INDEX = [
                np.where(np.array(self.LEFT_HAND_DEFAULT_JOINT_ORDER) == item)[0][0]
                for item in self.LEFT_HAND_SIM_JOINT_ORDER
            ]
            self.LEFT_HAND_HOME_JOINT = XHAND_LEFT_HOME_JOINT[
                self.LEFT_HAND_DEFAULT_2_SIM_INDEX
            ]

        # robot right
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot_right = loader.load(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_right_hand']['urdf_path']}"
        )
        self.robot_right.set_name("robot_right")
        self.active_joints_right = self.robot_right.get_active_joints()
        self.RIGHT_HAND_SIM_JOINT_ORDER = np.array(
            [joint.get_name() for joint in self.active_joints_right]
        )[-self.hand_dof :]
        for link in self.robot_right.get_links():
            if link.get_name() == "base":
                self.end_effector_right = link
                break
        for joint in self.active_joints_right:
            joint.set_drive_property(
                stiffness=1000, damping=100, force_limit=1e10, mode="force"
            )
            joint.set_friction(0.0)
        for link in self.robot_right.links:
            link.disable_gravity = True
        self.UR5E_RIGHT_HOME_JOINT = UR5E_RIGHT_HOME_JOINT
        if self.config["robot"]["ur5e_with_right_hand"]["hand_type"] == "xhand":
            self.RIGHT_HAND_DEFAULT_JOINT_ORDER = XHAND_RIGHT_DEFAULT_JOINT_ORDER
            self.RIGHT_HAND_DEFAULT_2_SIM_INDEX = [
                np.where(np.array(self.RIGHT_HAND_DEFAULT_JOINT_ORDER) == item)[0][0]
                for item in self.RIGHT_HAND_SIM_JOINT_ORDER
            ]
            self.RIGHT_HAND_HOME_JOINT = XHAND_RIGHT_HOME_JOINT[
                self.RIGHT_HAND_DEFAULT_2_SIM_INDEX
            ]

        self.init_qpos = [
            np.concatenate(
                [self.UR5E_LEFT_HOME_JOINT, self.LEFT_HAND_HOME_JOINT]
            ).astype(np.float32),
            np.concatenate(
                [self.UR5E_RIGHT_HOME_JOINT, self.RIGHT_HAND_HOME_JOINT]
            ).astype(np.float32),
        ]

        # set up curobo robot world
        self.init_robot_world()
        self.LEFT_HAND_ROBOT_WORLD_JOINT_ORDER = self.robot_world[
            0
        ].kinematics.joint_names[-self.hand_dof :]
        self.RIGHT_HAND_ROBOT_WORLD_JOINT_ORDER = self.robot_world[
            1
        ].kinematics.joint_names[-self.hand_dof :]
        self.LEFT_HAND_SIM_2_ROBOT_WORLD_INDEX = [
            np.where(np.array(self.LEFT_HAND_SIM_JOINT_ORDER) == item)[0][0]
            for item in self.LEFT_HAND_ROBOT_WORLD_JOINT_ORDER
        ]
        self.RIGHT_HAND_SIM_2_ROBOT_WORLD_INDEX = [
            np.where(np.array(self.RIGHT_HAND_SIM_JOINT_ORDER) == item)[0][0]
            for item in self.RIGHT_HAND_ROBOT_WORLD_JOINT_ORDER
        ]

        # set up synthetic pc util
        self.synthetic_pc_left = SyntheticPC(
            urdf_path=f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_left_hand']['urdf_path']}"
        )
        self.synthetic_pc_right = SyntheticPC(
            urdf_path=f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_right_hand']['urdf_path']}"
        )

    def init_robot(self):
        self.robot_left_transformation = np.eye(4)
        self.robot_left_transformation[:3, :3] = R.from_euler(
            "zyx", [0.0, 0.0, 0.0]
        ).as_matrix()
        self.robot_left_transformation[:3, 3] = np.array([0.0, 0.45, self.table_height])
        self.robot_left.set_root_pose(sapien.Pose(self.robot_left_transformation))
        self.robot_left.set_qpos(self.init_qpos[0])

        self.robot_right_transformation = np.eye(4)
        self.robot_right_transformation[:3, :3] = R.from_euler(
            "zyx", [0.0, 0.0, 0.0]
        ).as_matrix()
        self.robot_right_transformation[:3, 3] = np.array(
            [0.0, -0.45, self.table_height]
        )
        self.robot_right.set_root_pose(sapien.Pose(self.robot_right_transformation))
        self.robot_right.set_qpos(self.init_qpos[1])

        robot_pos_offset = self.robot_left_transformation[:3, 3]
        robot_rot_offset = R.from_matrix(
            self.robot_left_transformation[:3, :3]
        ).as_euler("XYZ", degrees=False)
        base_offset_joints_dict = {
            "pos_x_joint": robot_pos_offset[0],
            "pos_y_joint": robot_pos_offset[1],
            "pos_z_joint": robot_pos_offset[2],
            "rot_x_joint": robot_rot_offset[0],
            "rot_y_joint": robot_rot_offset[1],
            "rot_z_joint": robot_rot_offset[2],
        }
        robot_config_dict = load_yaml(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_left_hand']['robot_world_config_path']}"
        )["robot_cfg"]
        robot_config_dict["kinematics"]["urdf_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
        )
        robot_config_dict["kinematics"]["asset_root_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
        )
        robot_config_dict["kinematics"]["lock_joints"] = base_offset_joints_dict
        robot_cfg = RobotConfig.from_dict(
            robot_config_dict, self.robot_world[0].tensor_args
        )
        self.robot_world[0].kinematics.update_kinematics_config(
            robot_cfg.kinematics.kinematics_config
        )

        robot_pos_offset = self.robot_right_transformation[:3, 3]
        robot_rot_offset = R.from_matrix(
            self.robot_right_transformation[:3, :3]
        ).as_euler("XYZ", degrees=False)
        base_offset_joints_dict = {
            "pos_x_joint": robot_pos_offset[0],
            "pos_y_joint": robot_pos_offset[1],
            "pos_z_joint": robot_pos_offset[2],
            "rot_x_joint": robot_rot_offset[0],
            "rot_y_joint": robot_rot_offset[1],
            "rot_z_joint": robot_rot_offset[2],
        }
        robot_config_dict = load_yaml(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_right_hand']['robot_world_config_path']}"
        )["robot_cfg"]
        robot_config_dict["kinematics"]["urdf_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
        )
        robot_config_dict["kinematics"]["asset_root_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
        )
        robot_config_dict["kinematics"]["lock_joints"] = base_offset_joints_dict
        robot_cfg = RobotConfig.from_dict(
            robot_config_dict, self.robot_world[1].tensor_args
        )
        self.robot_world[1].kinematics.update_kinematics_config(
            robot_cfg.kinematics.kinematics_config
        )

    def set_up_object(self):
        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
        # builder.add_box_visual(half_size=[0.02, 0.02, 0.02], material=[1.0, 0.0, 0.0])
        # self.object = builder.build(name='object')
        # self.object.set_pose(sapien.Pose([0.5, 0, self.table_height + 0.03]))
        self.object = None

    def set_up_visual_boundary(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.005, 0.4, 0.0005], material=[1.0, 0.0, 0.0])
        visual_boundary_front = builder.build(name="visual_boundary_front")
        visual_boundary_front.set_pose(sapien.Pose([0.7, 0.0, self.table_height]))

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.005, 0.4, 0.0005], material=[1.0, 0.0, 0.0])
        visual_boundary_rear = builder.build(name="visual_boundary_front")
        visual_boundary_rear.set_pose(sapien.Pose([0.5, 0.0, self.table_height]))

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.1, 0.005, 0.0005], material=[1.0, 0.0, 0.0])
        visual_boundary_left = builder.build(name="visual_boundary_front")
        visual_boundary_left.set_pose(sapien.Pose([0.6, 0.4, self.table_height]))

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.1, 0.005, 0.0005], material=[1.0, 0.0, 0.0])
        visual_boundary_right = builder.build(name="visual_boundary_front")
        visual_boundary_right.set_pose(sapien.Pose([0.6, -0.4, self.table_height]))

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.1, 0.005, 0.0005], material=[1.0, 0.0, 0.0])
        visual_boundary_middle = builder.build(name="visual_boundary_front")
        visual_boundary_middle.set_pose(sapien.Pose([0.6, 0.0, self.table_height]))

    def init_robot_world(self):
        robot_config_dict = load_yaml(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_left_hand']['robot_world_config_path']}"
        )["robot_cfg"]
        robot_config_dict["kinematics"]["urdf_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
        )
        robot_config_dict["kinematics"]["asset_root_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
        )
        config_0 = RobotWorldConfig.load_from_config(
            robot_config_dict,
            {
                "mesh": {
                    "object": {
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "file_path": os.path.join(
                            get_assets_path(), "scene/nvblox/srl_ur10_bins.obj"
                        ),
                    }
                },
                "cuboid": {
                    "table": {
                        "dims": [3.0, 3.0, 0.2],
                        "pose": [0.0, 0.0, -0.1 + self.table_height, 1, 0, 0, 0.0],
                    }
                },
                # "cuboid": {"table": {"dims": [3.0, 3.0, 0.2], "pose": [0.0, 0.0, -0.1 + self.table_height - 0.01, 1, 0, 0, 0.0]}}
            },  # for collision cache init
            collision_activation_distance=0.005,
        )
        robot_world_0 = RobotWorld(config_0)
        robot_world_0.clear_world_cache()
        robot_config_dict = load_yaml(
            f"{self.config['asset_path']}/{self.config['robot']['ur5e_with_right_hand']['robot_world_config_path']}"
        )["robot_cfg"]
        robot_config_dict["kinematics"]["urdf_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
        )
        robot_config_dict["kinematics"]["asset_root_path"] = (
            f"{self.config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
        )
        config_1 = RobotWorldConfig.load_from_config(
            robot_config_dict,
            {
                "mesh": {
                    "object": {
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "file_path": os.path.join(
                            get_assets_path(), "scene/nvblox/srl_ur10_bins.obj"
                        ),
                    }
                },
                "cuboid": {
                    "table": {
                        "dims": [3.0, 3.0, 0.2],
                        "pose": [0.0, 0.0, -0.1 + self.table_height, 1, 0, 0, 0.0],
                    }
                },
                # "cuboid": {"table": {"dims": [3.0, 3.0, 0.2], "pose": [0.0, 0.0, -0.1 + self.table_height - 0.01, 1, 0, 0, 0.0]}}
            },  # for collision cache init
            collision_activation_distance=0.005,
        )
        robot_world_1 = RobotWorld(config_1)
        robot_world_1.clear_world_cache()
        self.robot_world = [robot_world_0, robot_world_1]

    def get_actor(self, name):
        all_actors = self.scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f"Not a unique name for actor: {name}")
        elif len(actor) == 0:
            raise RuntimeError(f"Actor not found: {name}")
        return actor[0]

    def get_articulation(self, name):
        all_articulations = self.scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f"Not a unique name for articulation: {name}")
        elif len(articulation) == 0:
            raise RuntimeError(f"Articulation not found: {name}")
        return articulation[0]

    def set_up_camera(self, near=0.01, far=10.0):
        self.camera_list = []
        camera_0 = self.scene.add_camera(
            name="Primary_0",
            width=640,  # 1200
            height=480,  # 1200
            fovy=np.pi / 3,
            near=near,
            far=far,
        )
        self.camera_list.append(camera_0)
        camera_1 = self.scene.add_camera(
            name="Primary_1",
            width=640,
            height=480,
            fovy=np.pi / 3,
            near=near,
            far=far,
        )
        self.camera_list.append(camera_1)
        self.init_camera()

    def init_camera(self):
        pos = np.array([1.2, 1.2, self.table_height + 0.6])
        rot = (
            R.from_euler("XYZ", [-np.pi * 5 / 16, np.pi * 8 / 32, np.pi * 10 / 12])
            * R.from_euler("ZYX", [np.pi / 2, np.pi / 2, 0])
        ).as_quat(scalar_first=True)
        self.camera_list[0].entity.set_pose(sapien.Pose(pos, rot))
        pos = np.array([1.2, -1.2, self.table_height + 0.6])
        rot = (
            R.from_euler("XYZ", [np.pi * 5 / 16, np.pi * 8 / 32, np.pi * 2 / 12])
            * R.from_euler("ZYX", [np.pi / 2, np.pi / 2, 0])
        ).as_quat(scalar_first=True)
        self.camera_list[1].entity.set_pose(sapien.Pose(pos, rot))

    def apply_action(self, action):
        # position control
        for i in range(len(self.active_joints_left)):
            self.active_joints_left[i].set_drive_target(action[i])
        for i in range(
            len(self.active_joints_left),
            len(self.active_joints_left) + len(self.active_joints_right),
        ):
            self.active_joints_right[i - len(self.active_joints_left)].set_drive_target(
                action[i]
            )

    def reset(self, episode_idx=None):
        self.init_robot()
        if self.with_object:
            self.init_object(episode_idx)
        self.init_camera()

        self.warm_up()

        self.object_init_pose = self.get_object_pose()

        self.init_object_pc = None

        return self.get_obs()

    def set_object_path_and_scale_and_hand(self, path, scale, hand, xy_step_str=None):
        assert (
            os.path.exists(path)
            and os.path.isfile(path)
            and path.endswith(".obj")
            and hand in [0, 1, 2]
        )
        self.object_mesh_path = path
        self.object_scale = scale
        self.hand = hand

        if self.hand == 0:
            self.x_range = [0.5, 0.7]
            self.y_range = [0.0, 0.4]
        elif self.hand == 1:
            self.x_range = [0.5, 0.7]
            self.y_range = [-0.4, 0.0]
        elif self.hand == 2:
            self.x_range = [0.55, 0.7]
            self.y_range = [-0.08, 0.08]

        if xy_step_str is not None:
            x_range = np.linspace(
                self.x_range[0], self.x_range[1], eval(xy_step_str)[0]
            )
            y_range = np.linspace(
                self.y_range[0], self.y_range[1], eval(xy_step_str)[1]
            )
            x, y = np.meshgrid(x_range, y_range)
            self.xy_displacement = np.stack([x.flatten(), y.flatten()], axis=-1)

    def init_object(self, episode_idx):
        if self.object is not None:
            self.scene.remove_entity(self.object)

        if self.hand == 0 or self.hand == 1:
            if episode_idx is not None and episode_idx % 4 == 0:
                orientation = R.from_euler(
                    "xyz", np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
                ).as_quat(scalar_first=True)
            else:
                orientation = R.random().as_quat(scalar_first=True)
            # orientation = R.from_euler('xyz', np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])).as_quat(scalar_first=True)
        elif self.hand == 2:
            if episode_idx is not None and episode_idx % 4 == 0:
                orientation = R.from_euler(
                    "xyz", np.array([0.0, 0.0, np.random.uniform(-np.pi, np.pi)])
                ).as_quat(scalar_first=True)
            else:
                orientation = R.from_euler(
                    "xyz",
                    np.array(
                        [
                            np.random.choice([0.0, np.pi / 2, np.pi, np.pi * 3 / 2]),
                            np.random.choice([0.0, np.pi / 2, np.pi, np.pi * 3 / 2]),
                            np.random.choice([0.0, np.pi / 2, np.pi, np.pi * 3 / 2])
                            + np.random.uniform(-np.pi / 12, np.pi / 12),
                        ]
                    ),
                ).as_quat(scalar_first=True)
            # orientation = R.from_euler('xyz', np.array([0.0, 0.0, np.random.uniform(-np.pi / 12, np.pi / 12)])).as_quat(scalar_first=True)
        # orientation = [1.0, 0.0, 0.0, 0.0]

        object_mesh = trimesh.load(self.object_mesh_path)
        object_pose_tmp = np.eye(4)
        object_pose_tmp[:3, :3] = R.from_quat(
            orientation, scalar_first=True
        ).as_matrix()
        object_mesh.apply_transform(object_pose_tmp)
        object_mesh.apply_scale(self.object_scale)
        z_min = object_mesh.bounds[0][2]

        if episode_idx is not None:
            x, y = (
                self.xy_displacement[episode_idx][0],
                self.xy_displacement[episode_idx][1],
            )
        else:
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
        z = -z_min + self.table_height + 0.02
        position = np.array([x, y, z])

        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename=self.object_mesh_path, scale=[self.object_scale] * 3, density=100
        )
        builder.add_visual_from_file(
            filename=self.object_mesh_path, scale=[self.object_scale] * 3
        )
        self.object = builder.build(name="object")
        self.object.set_pose(sapien.Pose(position, orientation))

        object_rigid_component = self.object.find_component_by_type(
            sapien.physx.PhysxRigidBodyComponent
        )
        object_rigid_component.set_linear_damping(20)
        object_rigid_component.set_angular_damping(20)
        if object_mesh.is_watertight:
            mass = min(0.1, object_mesh.volume * 100)
        else:
            print(f"Warning: Object mesh is not watertight, using default mass 0.1")
            mass = 0.1
        object_rigid_component.set_mass(mass)

    def is_object_in_boundary(self, object_pose):
        return (
            object_pose[0] > self.x_range[0] - 0.01
            and object_pose[0] < self.x_range[1] + 0.01
            and object_pose[1] > self.y_range[0] - 0.01
            and object_pose[1] < self.y_range[1] + 0.01
        )

    def warm_up(self):
        for _ in range(20):
            self.apply_action(np.concatenate(self.init_qpos, axis=-1))
            for _ in range(self.frame_skip):
                self.scene.step()
            self.scene.update_render()

    def step(self, action, get_obs=True):
        self.apply_action(action)
        for _ in range(self.frame_skip):
            self.scene.step()
        self.scene.update_render()
        if get_obs:
            obs = self.get_obs()
        else:
            obs = {}
        obs["success"] = self.get_object_pose()[2] - self.object_init_pose[2] > 0.1

        return obs

    def get_obs(self):
        qpos_0 = self.robot_left.get_qpos()
        qvel_0 = self.robot_left.get_qvel()
        qpos_1 = self.robot_right.get_qpos()
        qvel_1 = self.robot_right.get_qvel()
        q_0 = torch.from_numpy(qpos_0[:6]).unsqueeze(dim=0).to(device="cuda")
        ee_pose_0 = self.robot_world[0].get_kinematics(q_0)
        ee_pose_0 = np.concatenate(
            [
                ee_pose_0.ee_position.squeeze(0).cpu().numpy(),
                ee_pose_0.ee_quaternion.squeeze(0).cpu().numpy(),
            ]
        )
        q_1 = torch.from_numpy(qpos_1[:6]).unsqueeze(dim=0).to(device="cuda")
        ee_pose_1 = self.robot_world[1].get_kinematics(q_1)
        ee_pose_1 = np.concatenate(
            [
                ee_pose_1.ee_position.squeeze(0).cpu().numpy(),
                ee_pose_1.ee_quaternion.squeeze(0).cpu().numpy(),
            ]
        )
        obs = {
            "robot_0": {"qpos": qpos_0, "qvel": qvel_0, "ee_pose": ee_pose_0},
            "robot_1": {"qpos": qpos_1, "qvel": qvel_1, "ee_pose": ee_pose_1},
        }

        for camera_idx in range(len(self.camera_list)):
            camera = self.camera_list[camera_idx]
            camera_name = camera.get_name()

            camera.take_picture()
            obs[camera_name] = {}

            if "rgb" in self.obs_type:
                color_image = camera.get_picture("Color")[:, :, :3]
                obs[camera_name]["color_image"] = (
                    (color_image * 255).clip(0, 255).astype(np.uint8)
                )

            if "depth" in self.obs_type:
                position = camera.get_picture("Position")
                obs[camera_name]["depth_image"] = -position[..., 2][
                    ..., None
                ]  # is this distance_to_image_plane?

            if "point_cloud" in self.obs_type:
                color_image = camera.get_picture("Color")[:, :, :3]
                color_image = (color_image * 255).clip(0, 255).astype(np.uint8)

                position = camera.get_picture("Position")
                pc_opengl = position[..., :3][position[..., 3] < 1]
                points_color = color_image[position[..., 3] < 1]
                camera_extrinsic_matrix = camera.get_model_matrix()
                pc_world = (
                    pc_opengl @ camera_extrinsic_matrix[:3, :3].T
                    + camera_extrinsic_matrix[:3, 3]
                )

                seg_label = camera.get_picture("Segmentation")[..., 0].astype(np.uint8)[
                    position[..., 3] < 1
                ]
                object_mask = (seg_label == seg_label.max()).astype(np.uint8)

                obs[camera_name]["point_cloud"] = np.concatenate(
                    [
                        pc_world.astype(np.float32),
                        object_mask.reshape(-1, 1).astype(np.float32),
                    ],
                    axis=-1,
                )

        if "point_cloud" in self.obs_type:
            point_cloud = np.concatenate(
                [obs[camera.get_name()]["point_cloud"] for camera in self.camera_list],
                axis=0,
            )
            point_offset = np.array([0.0, 0.0, -self.table_height]).astype(np.float32)
            point_cloud[:, :3] += point_offset

            x_min = 0.4
            z_min = 0.01
            real_pc = crop_point_cloud(
                point_cloud,
                np.array([[x_min, 1.0], [-0.6, 0.6], [z_min, 0.8]]).astype(np.float32),
            )
            sampled_idx = np.random.permutation(real_pc.shape[0])[
                : min(2000, real_pc.shape[0])
            ]
            real_pc = real_pc[sampled_idx]

            robot_left_pc = self.synthetic_pc_left.get_pc_at_qpos(
                obs["robot_0"]["qpos"]
            )
            robot_left_pc[:, :3] = (
                robot_left_pc[:, :3] @ self.robot_left_transformation[:3, :3].T
                + self.robot_left_transformation[:3, 3]
            )
            robot_left_pc[:, :3] += point_offset
            sampled_idx = np.random.permutation(robot_left_pc.shape[0])[:4000]
            robot_left_pc = robot_left_pc[sampled_idx]
            robot_right_pc = self.synthetic_pc_right.get_pc_at_qpos(
                obs["robot_1"]["qpos"]
            )
            robot_right_pc[:, :3] = (
                robot_right_pc[:, :3] @ self.robot_right_transformation[:3, :3].T
                + self.robot_right_transformation[:3, 3]
            )
            robot_right_pc[:, :3] += point_offset
            sampled_idx = np.random.permutation(robot_right_pc.shape[0])[:4000]
            robot_right_pc = robot_right_pc[sampled_idx]
            table_pc = self.synthetic_pc_left.synthetic_table_pc
            sampled_idx = np.random.permutation(table_pc.shape[0])[:1000]
            table_pc = table_pc[sampled_idx]
            synthetic_pc = np.concatenate(
                [robot_left_pc, robot_right_pc], axis=0
            )  # table_pc,
            synthetic_pc = np.concatenate(
                [synthetic_pc[:, :3], np.zeros((synthetic_pc.shape[0], 1))], axis=-1
            )

            point_cloud = np.concatenate([real_pc, synthetic_pc], axis=0)
            point_cloud = crop_point_cloud(
                point_cloud,
                np.array([[-0.3, 1.0], [-0.6, 0.6], [0.0, 0.8]]).astype(np.float32),
            )

            sampled_point_cloud, fps_idx = sample_farthest_points(
                points=torch.from_numpy(point_cloud[:, :3]).cuda()[None],
                K=1200 * 2,
                random_start_point=True,
            )
            sampled = point_cloud[fps_idx[0].cpu().numpy()]
            # save_pc_as_ply(sampled[:, :3], f'point_cloud.ply')
            obs["point_cloud"] = sampled[:, :3].astype(np.float32)
            obs["point_cloud_mask"] = sampled[:, 3:4].astype(np.uint8)

        return obs

    def get_object_pose(self):
        return np.concatenate(
            [self.object.get_pose().p, self.object.get_pose().q], axis=-1
        )

    def check_object_moved(self):
        object_pose = self.get_object_pose()

        return (
            np.linalg.norm(object_pose[:3] - self.object_init_pose[:3]) > 0.01
            or calculate_angle_between_quat(object_pose[3:], self.object_init_pose[3:])
            > np.pi / 18
        )
