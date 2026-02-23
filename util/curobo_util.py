import os
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import get_assets_path, load_yaml


def setup_curobo_utils(config_path, interpolation_dt=0.05, is_bimanual=False, left_motion_gen_config_path='ur5e_with_leap_urdf/ur5e_with_leap_left.yaml', right_motion_gen_config_path='ur5e_with_leap_urdf/ur5e_with_leap_right.yaml', left_ik_solver_config_path='ur5e_with_leap_urdf/ur5e_with_base_left.yaml', right_ik_solver_config_path='ur5e_with_leap_urdf/ur5e_with_base_right.yaml', device='cuda:0'):
    tensor_args = TensorDeviceType(device)
    # ik solver
    robot_cfg = load_yaml(f'{config_path}/{left_ik_solver_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    ik_config_left = IKSolverConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        None,
        tensor_args=tensor_args,
        position_threshold=0.001,
        rotation_threshold=0.01,
        use_cuda_graph=False,
    )
    ik_solver_left = IKSolver(ik_config_left)
    robot_cfg = load_yaml(f'{config_path}/{right_ik_solver_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    ik_config_right = IKSolverConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        None,
        tensor_args=tensor_args,
        position_threshold=0.001,
        rotation_threshold=0.01,
        use_cuda_graph=False,
    )
    ik_solver_right = IKSolver(ik_config_right)
    ik_solver = [ik_solver_left, ik_solver_right]

    # fk
    robot_cfg = load_yaml(f'{config_path}/{left_ik_solver_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    robot_cfg_left = RobotConfig.from_dict(robot_cfg, tensor_args)
    kin_model_left = CudaRobotModel(robot_cfg_left.kinematics)
    robot_cfg = load_yaml(f'{config_path}/{right_ik_solver_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    robot_cfg_right = RobotConfig.from_dict(robot_cfg, tensor_args)
    kin_model_right = CudaRobotModel(robot_cfg_right.kinematics)
    kin_model = [kin_model_left, kin_model_right]

    # motion gen
    robot_cfg = load_yaml(f'{config_path}/{left_motion_gen_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    motion_gen_common_config_left = MotionGenConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        {"mesh": {"object": {"pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "file_path": f'{get_assets_path()}/scene/nvblox/srl_ur10_bins.obj'}}, "cuboid": {"table": {"dims": [3.0, 3.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0]}}},  # for collision cache init
        tensor_args=tensor_args,
        interpolation_dt=interpolation_dt,
        collision_activation_distance=0.01,
        trajopt_tsteps=32,
        self_collision_check=True,
        position_threshold=0.001,
        rotation_threshold=0.01,
        collision_cache={"obb": 10, "mesh": 10},
        project_pose_to_goal_frame=False
    )
    motion_gen_common_left = MotionGen(motion_gen_common_config_left)
    motion_gen_common_left.warmup()
    robot_cfg = load_yaml(f'{config_path}/{right_motion_gen_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    motion_gen_common_config_right = MotionGenConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        {"mesh": {"object": {"pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "file_path": f'{get_assets_path()}/scene/nvblox/srl_ur10_bins.obj'}}, "cuboid": {"table": {"dims": [3.0, 3.0, 0.2], "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0]}}},  # for collision cache init
        tensor_args=tensor_args,
        interpolation_dt=interpolation_dt,
        collision_activation_distance=0.01,
        trajopt_tsteps=32,
        self_collision_check=True,
        position_threshold=0.001,
        rotation_threshold=0.01,
        collision_cache={"obb": 10, "mesh": 10},
        project_pose_to_goal_frame=False
    )
    motion_gen_common_right = MotionGen(motion_gen_common_config_right)
    motion_gen_common_right.warmup()
    motion_gen_common = [motion_gen_common_left, motion_gen_common_right]

    robot_cfg = load_yaml(f'{config_path}/{left_motion_gen_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    motion_gen_lift_config_left = MotionGenConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        tensor_args=tensor_args,
        interpolation_dt=interpolation_dt,
        collision_activation_distance=0.01,
        trajopt_tsteps=32,
        self_collision_check=True,
        position_threshold=0.001 if is_bimanual else 0.03,
        rotation_threshold=0.01 if is_bimanual else 0.03,
        project_pose_to_goal_frame=False
    )
    motion_gen_lift_left = MotionGen(motion_gen_lift_config_left)
    motion_gen_lift_left.warmup()
    robot_cfg = load_yaml(f'{config_path}/{right_motion_gen_config_path}')["robot_cfg"]
    robot_cfg['kinematics']['urdf_path'] = f"{config_path}/{robot_cfg['kinematics']['urdf_path']}"
    robot_cfg['kinematics']['asset_root_path'] = f"{config_path}/{robot_cfg['kinematics']['asset_root_path']}"
    motion_gen_lift_config_right = MotionGenConfig.load_from_robot_config(
        RobotConfig.from_dict(robot_cfg),
        tensor_args=tensor_args,
        interpolation_dt=interpolation_dt,
        collision_activation_distance=0.01,
        trajopt_tsteps=32,
        self_collision_check=True,
        position_threshold=0.001 if is_bimanual else 0.03,
        rotation_threshold=0.01 if is_bimanual else 0.03,
        project_pose_to_goal_frame=False
    )
    motion_gen_lift_right = MotionGen(motion_gen_lift_config_right)
    motion_gen_lift_right.warmup()
    motion_gen_lift = [motion_gen_lift_left, motion_gen_lift_right]

    return kin_model, ik_solver, motion_gen_common, motion_gen_lift
