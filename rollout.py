import os
import yaml
with open('env/config/env.yaml', 'r') as file:
    config = yaml.safe_load(file)
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config['asset_path'] = os.path.join(cur_path, config['asset_path'])
    config['object_mesh_path'] = os.path.join(cur_path, config['object_mesh_path'])
from util.bodex_util import GraspSynthesizer
import sys; hand = eval(sys.argv[2])
num_grasp = 100
grasp_synthesizer = GraspSynthesizer(hand=hand, num_grasp=num_grasp, hand_type=config['robot']['ur5e_with_left_hand']['hand_type'], dof=config['robot']['ur5e_with_left_hand']['hand_dof'])
grasp_synthesizer.synthesize_grasp(config['object_mesh_path'], [0., 0., 0., 1., 0., 0., 0.], 1.0)

import copy
import json
import click
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from curobo.util_file import load_yaml
from curobo.types.robot import RobotConfig
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig, PoseCostMetric
from curobo.util.trajectory import get_interpolated_trajectory, InterpolateType

from env.base_env import BaseEnv
from util.util import save_rgb_images_to_video, sort_grasp_for_single_hand, sort_grasp_for_dual_hand, calculate_angle_between_quat_torch
from util.curobo_util import setup_curobo_utils

np.random.seed(42)
pregrasp_distance = 0.1


def _build_state(obs_t, hand_mode):
    """Per-step low-dim state. qpos is primary; ee_pose appended as auxiliary feature."""
    if hand_mode in (0, 3):
        return np.concatenate([obs_t['robot_0']['qpos'], obs_t['robot_0']['ee_pose']]).astype(np.float32)
    if hand_mode in (1, 4):
        return np.concatenate([obs_t['robot_1']['qpos'], obs_t['robot_1']['ee_pose']]).astype(np.float32)
    # bimanual
    return np.concatenate([
        obs_t['robot_0']['qpos'], obs_t['robot_0']['ee_pose'],
        obs_t['robot_1']['qpos'], obs_t['robot_1']['ee_pose'],
    ]).astype(np.float32)


def _build_action(joint_action, hand_mode):
    """Slice the 36D joint command to the active half for single-arm modes."""
    if hand_mode in (0, 3):
        return joint_action[:18].astype(np.float32)
    if hand_mode in (1, 4):
        return joint_action[18:].astype(np.float32)
    return joint_action.astype(np.float32)


def _save_episode_npz(out_path, episode_data, meta):
    arrays = {k: np.stack(v, axis=0) for k, v in episode_data.items()}
    arrays['meta'] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(out_path, **arrays)


def _mark_episode_done(output_dir, episode_idx):
    open(os.path.join(output_dir, f'episode_{episode_idx:05d}.done'), 'w').close()


def _get_completed_episodes(output_dir):
    import glob, re
    done = set()
    for f in glob.glob(os.path.join(output_dir, 'episode_*.done')):
        m = re.match(r'episode_(\d+)\.done$', os.path.basename(f))
        if m:
            done.add(int(m.group(1)))
    return done


def rollout_for_an_object(env, hand, object_scale, object_mesh_path, output_dir='.', skip_episodes=None):
    if hand == 3 or hand == 4:
        grasp_mode = copy.deepcopy(hand)
        hand -= 3
    else:
        grasp_mode = hand
    kin_model, ik_solver, motion_gen_common, motion_gen_lift = setup_curobo_utils(config_path=config['asset_path'], is_bimanual=hand == 2, left_motion_gen_config_path=config['robot']['ur5e_with_left_hand']['curobo_motion_gen_config_path'], right_motion_gen_config_path=config['robot']['ur5e_with_right_hand']['curobo_motion_gen_config_path'], left_ik_solver_config_path=config['robot']['ur5e_with_left_hand']['curobo_ik_solver_config_path'], right_ik_solver_config_path=config['robot']['ur5e_with_right_hand']['curobo_ik_solver_config_path'])
    env.set_object_path_and_scale_and_hand(f"{object_mesh_path}/mesh/simplified.obj", object_scale, hand, config['xy_step_str'])

    episode_idx = 0
    step_idx = 0
    stage_idx = 0  # 0 init to pregrasp, 1 pregrasp to grasp, 2 grasp to squeeze, 3 squeeze to lift
    step_in_stage_idx = 0
    image_list = []
    episode_data = None
    stage_boundaries = []
    num_episode = eval(config['xy_step_str'])[0] * eval(config['xy_step_str'])[1]
    object_name = os.path.basename(object_mesh_path.rstrip('/'))
    os.makedirs(output_dir, exist_ok=True)

    while True:
        if episode_idx >= num_episode:
            break

        if step_idx == 0:
            if skip_episodes and episode_idx in skip_episodes:
                print(f'[RESUME] episode {episode_idx} already done, skipping')
                episode_idx += 1
                continue
            image_list = []
            episode_data = {
                'point_cloud': [],
                'point_cloud_mask': [],
                'agent_pos': [],
                'action': [],
                'object_pose': [],
                'rgb_primary_0': [],
                'rgb_primary_1': [],
            }
            stage_boundaries = [0]

            obs = env.reset(episode_idx)
            while not env.is_object_in_boundary(env.get_object_pose()):
                obs = env.reset(episode_idx)

            init_ee_pose = [obs['robot_0']['ee_pose'], obs['robot_1']['ee_pose']]
            object_pose = env.get_object_pose()

            # update locked joints for base offset
            robot_pos_offset = env.robot_left_transformation[:3, 3]
            robot_rot_offset = R.from_matrix(env.robot_left_transformation[:3, :3]).as_euler('XYZ', degrees=False)
            base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
            robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_left_hand']['curobo_ik_solver_config_path']}")['robot_cfg']
            robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
            robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
            robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
            robot_cfg = RobotConfig.from_dict(robot_config_dict, ik_solver[0].tensor_args)
            ik_solver[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

            robot_pos_offset = env.robot_right_transformation[:3, 3]
            robot_rot_offset = R.from_matrix(env.robot_right_transformation[:3, :3]).as_euler('XYZ', degrees=False)
            base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
            robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_right_hand']['curobo_ik_solver_config_path']}")['robot_cfg']
            robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
            robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
            robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
            robot_cfg = RobotConfig.from_dict(robot_config_dict, ik_solver[1].tensor_args)
            ik_solver[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

            world_config = {
                "mesh": {
                    "object": {
                        "pose": object_pose.tolist(),
                        "file_path": f"{object_mesh_path}/mesh/simplified.obj",
                        "scale": [object_scale] * 3
                    },
                },
                "cuboid": {
                    "table": {
                        "dims": [3.0, 3.0, 0.2],
                        "pose": [0.0, 0.0, env.table_height - 0.1, 1.0, 0.0, 0.0, 0.0],
                    },
                },
            }
            world_config_without_object = {
                "cuboid": {
                    "table": {
                        "dims": [3.0, 3.0, 0.2],
                        "pose": [0.0, 0.0, env.table_height - 0.1, 1.0, 0.0, 0.0, 0.0],
                    },
                },
            }
            env.robot_world[0].clear_world_cache()
            env.robot_world[0].update_world(WorldConfig.from_dict(world_config))
            env.robot_world[1].clear_world_cache()
            env.robot_world[1].update_world(WorldConfig.from_dict(world_config))

            data_all = grasp_synthesizer.synthesize_grasp(f"{object_mesh_path}", object_pose.tolist(), object_scale)

            # find best grasp
            if hand == 0 or hand == 1:
                # adjust pregrasp to avoid collision with the object
                if hand == 0:
                    # grasp_direction_in_hand_frame = -np.array([-0.45, -0.4, 1.0])  # old, for leap
                    grasp_direction_in_hand_frame = -np.array([-0.25, -0.25, 1.0])  # better for xhand
                elif hand == 1:
                    # grasp_direction_in_hand_frame = -np.array([-0.45, 0.4, 1.0])  # old, for leap
                    grasp_direction_in_hand_frame = -np.array([-0.25, 0.25, 1.0])  # better for xhand
                grasp_direction_in_hand_frame = grasp_direction_in_hand_frame / np.linalg.norm(grasp_direction_in_hand_frame)
                grasp_direction = R.from_quat(data_all[:, 0, 1, 3:7], scalar_first=True).as_matrix() @ grasp_direction_in_hand_frame
                data_all[:, 0, 0, :3] = data_all[:, 0, 1, :3] - pregrasp_distance * grasp_direction
                data_all[:, 0, 0, 3:7] = data_all[:, 0, 1, 3:7]

                # filter out grasp that robot cannot reach at pregrasp
                goal = Pose.from_batch_list(data_all[:, 0, 0, :7].tolist(), ik_solver[hand].tensor_args)
                result = ik_solver[hand].solve_batch(goal)
                ik_success = result.success
                if ik_success.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                q_arm_pregrasp = result.solution[ik_success]
                data_all = data_all[ik_success.squeeze(dim=-1).cpu().numpy()]
                print(len(data_all), 'pregrasp ik')
                # filter out grasp with collision at pregrasp
                if hand == 0:
                    q_hand_pregrasp = data_all[:, 0, 1, 7:][:, env.LEFT_HAND_SIM_2_ROBOT_WORLD_INDEX]
                elif hand == 1:
                    q_hand_pregrasp = data_all[:, 0, 1, 7:][:, env.RIGHT_HAND_SIM_2_ROBOT_WORLD_INDEX]
                q_s_pregrasp = torch.cat([q_arm_pregrasp, torch.from_numpy(q_hand_pregrasp).cuda()], dim=-1)
                d_world, d_self = env.robot_world[hand].get_world_self_collision_distance_from_joints(q_s_pregrasp)
                without_collision = (d_world <= 0).cpu().numpy()
                if without_collision.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                data_all = data_all[without_collision]
                print(len(data_all), 'pregrasp coll')

                # filter out grasp that robot cannot reach at grasp
                goal = Pose.from_batch_list(data_all[:, 0, 1, :7].tolist(), ik_solver[hand].tensor_args)
                result = ik_solver[hand].solve_batch(goal)
                ik_success = result.success
                if ik_success.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                q_arm_grasp = result.solution[ik_success]
                data_all = data_all[ik_success.squeeze(dim=-1).cpu().numpy()]
                print(len(data_all), 'grasp ik')
                # # filter out grasp with collision at grasp
                # if hand == 0:
                #     q_hand_grasp = data_all[:, 0, 1, 7:][:, env.LEFT_HAND_SIM_2_ROBOT_WORLD_INDEX]  # grasp
                # elif hand == 1: 
                #     q_hand_grasp = data_all[:, 0, 1, 7:][:, env.RIGHT_HAND_SIM_2_ROBOT_WORLD_INDEX]  # grasp
                # q_s_grasp = torch.cat([q_arm_grasp, torch.from_numpy(q_hand_grasp).cuda()], dim=-1)
                # d_world, d_self = env.robot_world[hand].get_world_self_collision_distance_from_joints(q_s_grasp)
                # without_collision = (d_world <= 0.005).cpu().numpy()
                # if without_collision.sum() == 0:
                #     print('all data are filtered out')
                #     episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                # data_all = data_all[without_collision]
                # print(len(data_all), 'grasp coll')

                sorted_idx = sort_grasp_for_single_hand(obs[f'robot_{hand}']['ee_pose'], data_all[:, 0, 1, :7])
                best_grasp = data_all[sorted_idx[0]]

                key_poses = best_grasp[0:1, :, :7]  # shape: (1 for one hand or 2 for dual hands, 3, 7)
                key_qposes = best_grasp[0:1, :, 7:]
                if hand == 0:
                    more_open_qpos = np.where(np.abs(np.array(env.LEFT_HAND_HOME_JOINT)) < np.abs(key_qposes[0, 1, :]), np.array(env.LEFT_HAND_HOME_JOINT), key_qposes[0, 1, :])
                elif hand == 1:
                    more_open_qpos = np.where(np.abs(np.array(env.RIGHT_HAND_HOME_JOINT)) < np.abs(key_qposes[0, 1, :]), np.array(env.RIGHT_HAND_HOME_JOINT), key_qposes[0, 1, :])
                key_qposes[0, 1, :] = key_qposes[0, 0, :]
                key_qposes[0, 0, :] = key_qposes[0, 1, :] - (key_qposes[0, 1, :] - more_open_qpos) * 2 / 3
            elif hand == 2:
                # filter out pregrasp that robot cannot reach
                goal_left = Pose.from_batch_list(data_all[:, 0, 0, :7].tolist(), ik_solver[0].tensor_args)
                result = ik_solver[0].solve_batch(goal_left)
                ik_success = result.success
                if ik_success.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                q_arm_left = result.solution[ik_success]
                data_all = data_all[ik_success.squeeze(dim=-1).cpu().numpy()]
                goal_right = Pose.from_batch_list(data_all[:, 1, 0, :7].tolist(), ik_solver[1].tensor_args)
                result = ik_solver[1].solve_batch(goal_right)
                ik_success = result.success
                if ik_success.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                q_arm_left = q_arm_left[ik_success.squeeze(dim=-1)]
                q_arm_right = result.solution[ik_success]
                data_all = data_all[ik_success.squeeze(dim=-1).cpu().numpy()]
                print(len(data_all), 'grasp ik')

                # filter out pregrasp with collision (not grasp for dual hands)
                q_hand_left = data_all[:, 0, 0, 7:][:, env.LEFT_HAND_SIM_2_ROBOT_WORLD_INDEX]
                q_s_left = torch.cat([q_arm_left, torch.from_numpy(q_hand_left).cuda()], dim=-1)
                d_world, d_self = env.robot_world[0].get_world_self_collision_distance_from_joints(q_s_left)
                without_collision = (d_world <= 0.005).cpu().numpy()
                if without_collision.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                q_arm_right = q_arm_right[without_collision]
                data_all = data_all[without_collision]
                print(len(data_all), 'pregrasp coll left')
                q_hand_right = data_all[:, 1, 0, 7:][:, env.RIGHT_HAND_SIM_2_ROBOT_WORLD_INDEX]
                q_s_right = torch.cat([q_arm_right, torch.from_numpy(q_hand_right).cuda()], dim=-1)
                d_world, d_self = env.robot_world[1].get_world_self_collision_distance_from_joints(q_s_right)
                without_collision = (d_world <= 0.005).cpu().numpy()
                if without_collision.sum() == 0:
                    print('all data are filtered out')
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                data_all = data_all[without_collision]
                print(len(data_all), 'pregrasp coll right')

                sorted_idx = sort_grasp_for_dual_hand(obs['robot_0']['ee_pose'], obs['robot_1']['ee_pose'], data_all[:, 0, 1, :7], data_all[:, 1, 1, :7])
                best_grasp = data_all[sorted_idx[0]]

                key_poses = best_grasp[:, :, :7]
                key_qposes = best_grasp[:, :, 7:]

        if step_in_stage_idx == 0:
            print(f'episode {episode_idx}, stage {stage_idx}')
            if stage_idx == 0:
                motion_gen_common[0].reset(reset_seed=False)
                motion_gen_common[0].clear_world_cache()
                motion_gen_common[0].update_world(WorldConfig.from_dict(world_config))
                motion_gen_common[1].reset(reset_seed=False)
                motion_gen_common[1].clear_world_cache()
                motion_gen_common[1].update_world(WorldConfig.from_dict(world_config))
                if hand == 0:
                    locked_joints_dict = dict(zip(env.LEFT_HAND_SIM_JOINT_ORDER, key_qposes[0, 0].tolist()))
                    robot_pos_offset = env.robot_left_transformation[:3, 3]
                    robot_rot_offset = R.from_matrix(env.robot_left_transformation[:3, :3]).as_euler('XYZ', degrees=False)
                    base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
                    locked_joints_dict.update(base_offset_joints_dict)
                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_left_hand']['curobo_motion_gen_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = locked_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, motion_gen_common[0].tensor_args)
                    motion_gen_common[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                    motion_gen_lift[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                elif hand == 1:
                    locked_joints_dict = dict(zip(env.RIGHT_HAND_SIM_JOINT_ORDER, key_qposes[0, 0].tolist()))
                    robot_pos_offset = env.robot_right_transformation[:3, 3]
                    robot_rot_offset = R.from_matrix(env.robot_right_transformation[:3, :3]).as_euler('XYZ', degrees=False)
                    base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
                    locked_joints_dict.update(base_offset_joints_dict)
                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_right_hand']['curobo_motion_gen_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = locked_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, motion_gen_common[1].tensor_args)
                    motion_gen_common[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                    motion_gen_lift[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                elif hand == 2:
                    locked_joints_dict = dict(zip(env.LEFT_HAND_SIM_JOINT_ORDER, key_qposes[0, 0].tolist()))
                    robot_pos_offset = env.robot_left_transformation[:3, 3]
                    robot_rot_offset = R.from_matrix(env.robot_left_transformation[:3, :3]).as_euler('XYZ', degrees=False)
                    base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
                    locked_joints_dict.update(base_offset_joints_dict)
                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_left_hand']['curobo_motion_gen_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = locked_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, motion_gen_common[0].tensor_args)
                    motion_gen_common[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                    motion_gen_lift[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_left_hand']['curobo_ik_solver_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, ik_solver[0].tensor_args)
                    ik_solver[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_left_hand']['robot_world_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, env.robot_world[0].tensor_args)
                    env.robot_world[0].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

                    locked_joints_dict = dict(zip(env.RIGHT_HAND_SIM_JOINT_ORDER, key_qposes[1, 0].tolist()))
                    robot_pos_offset = env.robot_right_transformation[:3, 3]
                    robot_rot_offset = R.from_matrix(env.robot_right_transformation[:3, :3]).as_euler('XYZ', degrees=False)
                    base_offset_joints_dict = {'pos_x_joint': robot_pos_offset[0], 'pos_y_joint': robot_pos_offset[1], 'pos_z_joint': robot_pos_offset[2], 'rot_x_joint': robot_rot_offset[0], 'rot_y_joint': robot_rot_offset[1], 'rot_z_joint': robot_rot_offset[2]}
                    locked_joints_dict.update(base_offset_joints_dict)
                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_right_hand']['curobo_motion_gen_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = locked_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, motion_gen_common[1].tensor_args)
                    motion_gen_common[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
                    motion_gen_lift[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_right_hand']['curobo_ik_solver_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, ik_solver[1].tensor_args)
                    ik_solver[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)

                    robot_config_dict = load_yaml(f"{config['asset_path']}/{config['robot']['ur5e_with_right_hand']['robot_world_config_path']}")['robot_cfg']
                    robot_config_dict['kinematics']['urdf_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['urdf_path']}"
                    robot_config_dict['kinematics']['asset_root_path'] = f"{config['asset_path']}/{robot_config_dict['kinematics']['asset_root_path']}"
                    robot_config_dict['kinematics']['lock_joints'] = base_offset_joints_dict
                    robot_cfg = RobotConfig.from_dict(robot_config_dict, env.robot_world[1].tensor_args)
                    env.robot_world[1].kinematics.update_kinematics_config(robot_cfg.kinematics.kinematics_config)
            elif stage_idx == 1:
                motion_gen_common[0].clear_world_cache()
                motion_gen_common[1].clear_world_cache()
            if stage_idx == 3:
                motion_gen = motion_gen_lift
                pose_cost_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=motion_gen[0].tensor_args.to_device([1, 1, 1, 1, 1, 0]))
            else:
                motion_gen = motion_gen_common
                pose_cost_metric = None

            if hand == 0 or hand == 1:
                if stage_idx == 3:
                    goal_pose = Pose.from_list((obs[f'robot_{hand}']['ee_pose'] + np.array([0, 0, 0.15, 0, 0, 0, 0])).tolist())
                else:
                    goal_pose = Pose.from_list(key_poses[0, stage_idx].tolist())
                start_state = JointState.from_position(torch.from_numpy(obs[f'robot_{hand}']['qpos'][:6]).unsqueeze(dim=0).cuda())
                result = motion_gen[hand].plan_single(
                    start_state,
                    goal_pose,
                    MotionGenPlanConfig(pose_cost_metric=pose_cost_metric, max_attempts=300, timeout=30.0, enable_graph_attempt=10, ik_fail_return=5)
                )
                if result.success == False:
                    print(f"episode{episode_idx} stage{stage_idx} Trajectory Failed")
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                traj = result.get_interpolated_plan().position
                # adjust the length of the trajectory
                if stage_idx == 0:
                    traj_length = int(np.ceil(250 * np.linalg.norm(obs[f'robot_{hand}']['ee_pose'][:3] - goal_pose.position[:3].cpu().numpy())))  # smaller than 200 will cause the unexpected motion
                elif stage_idx == 1:
                    traj_length = 32
                elif stage_idx == 2:
                    traj_length = 12
                elif stage_idx == 3:
                    traj_length = 32
                out_traj_state = JointState.zeros(
                    [1, traj_length, 6], motion_gen[0].tensor_args, joint_names=result.optimized_plan.joint_names
                )  # [batch, traj_len, dof]
                try:
                    traj = get_interpolated_trajectory(traj.unsqueeze(0), out_traj_state, kind=InterpolateType.LINEAR)[0][0].position
                except:
                    print(f"get_interpolated_trajectory failed. Maybe caused by floating precision.")
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue

                # remove the pause
                if stage_idx == 0 or stage_idx == 1 or stage_idx == 3:
                    ee_pose_for_each_step = kin_model[hand].get_state(traj.clone())
                    delta_ee_pose = torch.linalg.norm((ee_pose_for_each_step.ee_position[1:] - ee_pose_for_each_step.ee_position[:-1]).abs(), dim=-1) + 0.5 * calculate_angle_between_quat_torch(ee_pose_for_each_step.ee_quaternion[1:], ee_pose_for_each_step.ee_quaternion[:-1])
                    avg_delta_ee_pose = delta_ee_pose.sum() / (traj_length * 0.9)

                    num_skip_last = 0
                    while num_skip_last + 2 <= delta_ee_pose.shape[0]:
                        if delta_ee_pose[-1-num_skip_last] + delta_ee_pose[-2-num_skip_last] < avg_delta_ee_pose:
                            traj[-2-num_skip_last] = traj[-1-num_skip_last]
                            mask = torch.ones(traj.shape[0], dtype=torch.bool)
                            mask[-1-num_skip_last] = False
                            traj = traj[mask]
                            delta_ee_pose[-2-num_skip_last] += delta_ee_pose[-1-num_skip_last]
                            mask = torch.ones(delta_ee_pose.shape[0], dtype=torch.bool)
                            mask[-1-num_skip_last] = False
                            delta_ee_pose = delta_ee_pose[mask]
                        else:
                            num_skip_last += 1
                traj = [traj[1:]]  # the 0th target qpos is the same as the start qpos
            elif hand == 2:
                if stage_idx == 3:
                    goal_pose_left = Pose.from_list((obs['robot_0']['ee_pose'] + np.array([0, 0, 0.15, 0, 0, 0, 0])).tolist())
                    goal_pose_right = Pose.from_list((obs['robot_1']['ee_pose'] + np.array([0, 0, 0.15, 0, 0, 0, 0])).tolist())
                else:
                    goal_pose_left = Pose.from_list(key_poses[0, stage_idx].tolist())
                    goal_pose_right = Pose.from_list(key_poses[1, stage_idx].tolist())
                start_state_left = JointState.from_position(torch.from_numpy(obs['robot_0']['qpos'][:6]).unsqueeze(dim=0).cuda())
                result_left = motion_gen[0].plan_single(
                    start_state_left,
                    goal_pose_left,
                    MotionGenPlanConfig(pose_cost_metric=pose_cost_metric, max_attempts=300, timeout=30.0, enable_graph_attempt=10, ik_fail_return=5)
                )
                start_state_right = JointState.from_position(torch.from_numpy(obs['robot_1']['qpos'][:6]).unsqueeze(dim=0).cuda())
                result_right = motion_gen[1].plan_single(
                    start_state_right,
                    goal_pose_right,
                    MotionGenPlanConfig(pose_cost_metric=pose_cost_metric, max_attempts=300, timeout=30.0, enable_graph_attempt=10, ik_fail_return=5)
                )
                if result_left.success == False or result_right.success == False:
                    print(f"episode{episode_idx} stage{stage_idx} Trajectory Failed")
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
                traj_left = result_left.get_interpolated_plan().position
                traj_right = result_right.get_interpolated_plan().position
                # adjust the length of the trajectory
                if stage_idx == 0:
                    traj_length_left = int(np.ceil(250 * np.linalg.norm(obs['robot_0']['ee_pose'][:3] - goal_pose_left.position[:3].cpu().numpy())))  # smaller than 200 will cause the unexpected motion
                    traj_length_right = int(np.ceil(250 * np.linalg.norm(obs['robot_1']['ee_pose'][:3] - goal_pose_right.position[:3].cpu().numpy())))  # smaller than 200 will cause the unexpected motion
                    traj_length = max(traj_length_left, traj_length_right)
                elif stage_idx == 1:
                    traj_length = 28
                elif stage_idx == 2:
                    traj_length = 12
                elif stage_idx == 3:
                    traj_length = 32
                out_traj_state_left = JointState.zeros(
                    [1, traj_length, 6], motion_gen[0].tensor_args, joint_names=result_left.optimized_plan.joint_names
                )  # [batch, traj_len, dof]
                out_traj_state_right = JointState.zeros(
                    [1, traj_length, 6], motion_gen[0].tensor_args, joint_names=result_right.optimized_plan.joint_names
                )  # [batch, traj_len, dof]
                try:
                    traj_left = get_interpolated_trajectory(traj_left.unsqueeze(0), out_traj_state_left, kind=InterpolateType.LINEAR)[0][0].position
                    traj_right = get_interpolated_trajectory(traj_right.unsqueeze(0), out_traj_state_right, kind=InterpolateType.LINEAR)[0][0].position
                except:
                    print(f"get_interpolated_trajectory failed. Maybe caused by floating precision.")
                    _mark_episode_done(output_dir, episode_idx)
                    episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue

                # remove the pause
                if stage_idx == 0 or stage_idx == 1 or stage_idx == 3:
                    ee_pose_left_for_each_step = kin_model[0].get_state(traj_left.clone())
                    ee_pose_right_for_each_step = kin_model[1].get_state(traj_right.clone())
                    delta_ee_pose = torch.linalg.norm((ee_pose_left_for_each_step.ee_position[1:] - ee_pose_left_for_each_step.ee_position[:-1]).abs(), dim=-1) + 0.5 * calculate_angle_between_quat_torch(ee_pose_left_for_each_step.ee_quaternion[1:], ee_pose_left_for_each_step.ee_quaternion[:-1]) + torch.linalg.norm((ee_pose_right_for_each_step.ee_position[1:] - ee_pose_right_for_each_step.ee_position[:-1]).abs(), dim=-1) + 0.5 * calculate_angle_between_quat_torch(ee_pose_right_for_each_step.ee_quaternion[1:], ee_pose_right_for_each_step.ee_quaternion[:-1])
                    avg_delta_ee_pose = delta_ee_pose.sum() / (traj_length * 0.9)

                    num_skip_last = 0
                    while num_skip_last + 2 <= delta_ee_pose.shape[0]:
                        if delta_ee_pose[-1-num_skip_last] + delta_ee_pose[-2-num_skip_last] < avg_delta_ee_pose:
                            traj_left[-2-num_skip_last] = traj_left[-1-num_skip_last]
                            mask = torch.ones(traj_left.shape[0], dtype=torch.bool)
                            mask[-1-num_skip_last] = False
                            traj_left = traj_left[mask]
                            traj_right[-2-num_skip_last] = traj_right[-1-num_skip_last]
                            mask = torch.ones(traj_right.shape[0], dtype=torch.bool)
                            mask[-1-num_skip_last] = False
                            traj_right = traj_right[mask]
                            delta_ee_pose[-2-num_skip_last] += delta_ee_pose[-1-num_skip_last]
                            mask = torch.ones(delta_ee_pose.shape[0], dtype=torch.bool)
                            mask[-1-num_skip_last] = False
                            delta_ee_pose = delta_ee_pose[mask]
                        else:
                            num_skip_last += 1
                traj = [traj_left[1:], traj_right[1:]]  # the 0th target qpos is the same as the start qpos

            # plan qpos
            if hand == 0 or hand == 1:
                if stage_idx == 3:
                    qposes = np.tile(key_qposes[0, 2], [traj[0].shape[0], 1])[None]
                else:
                    qposes = np.linspace(obs[f'robot_{hand}']['qpos'][6:], key_qposes[0, stage_idx], traj[0].shape[0] + 1)[1:][None]
            elif hand == 2:
                if stage_idx == 3:
                    qposes = [np.tile(key_qposes[0, 2], [traj[0].shape[0], 1]), np.tile(key_qposes[1, 2], [traj[1].shape[0], 1])]
                else:
                    qposes = [np.linspace(obs['robot_0']['qpos'][6:], key_qposes[0, stage_idx], traj[0].shape[0] + 1)[1:], np.linspace(obs['robot_1']['qpos'][6:], key_qposes[1, stage_idx], traj[1].shape[0] + 1)[1:]]

        if hand == 0:
            joint_action_0 = np.concatenate([traj[0][step_in_stage_idx].cpu().numpy(), qposes[0][step_in_stage_idx]])
            joint_action_1 = env.init_qpos[1]
        elif hand == 1:
            joint_action_0 = env.init_qpos[0]
            joint_action_1 = np.concatenate([traj[0][step_in_stage_idx].cpu().numpy(), qposes[0][step_in_stage_idx]])
        elif hand == 2:
            joint_action_0 = np.concatenate([traj[0][step_in_stage_idx].cpu().numpy(), qposes[0][step_in_stage_idx]])
            joint_action_1 = np.concatenate([traj[1][step_in_stage_idx].cpu().numpy(), qposes[1][step_in_stage_idx]])
        joint_action = np.concatenate([joint_action_0, joint_action_1])

        obs = env.get_obs()  # ensure the obs is updated; same instance reused below to avoid double sampling
        image_list.append(obs['Primary_0']['color_image'])

        # record (s_t, a_t) BEFORE stepping; reuse this same obs for both video and dataset
        episode_data['point_cloud'].append(obs['point_cloud'])
        episode_data['point_cloud_mask'].append(obs['point_cloud_mask'])
        episode_data['agent_pos'].append(_build_state(obs, grasp_mode))
        episode_data['action'].append(_build_action(joint_action, grasp_mode))
        episode_data['object_pose'].append(env.get_object_pose().astype(np.float32))
        episode_data['rgb_primary_0'].append(obs['Primary_0']['color_image'])
        episode_data['rgb_primary_1'].append(obs['Primary_1']['color_image'])

        obs = env.step(joint_action)

        if stage_idx < 2 and env.check_object_moved():
            print('object moved, filtered out')
            _mark_episode_done(output_dir, episode_idx)
            episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue

        # filter out trajectories in which EE pos moves backward or upward
        if hand == 0 or hand == 1:
            if obs[f'robot_{hand}']['ee_pose'][0] < init_ee_pose[hand][0] - 0.3 or obs[f'robot_{hand}']['ee_pose'][2] > init_ee_pose[hand][2] + 0.15:
                print('redundant motion, filtered out')
                _mark_episode_done(output_dir, episode_idx)
                episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue
        elif hand == 2:
            if obs['robot_0']['ee_pose'][0] < init_ee_pose[0][0] - 0.3 or obs['robot_0']['ee_pose'][2] > init_ee_pose[0][2] + 0.15 or obs['robot_1']['ee_pose'][0] < init_ee_pose[1][0] - 0.3 or obs['robot_1']['ee_pose'][2] > init_ee_pose[1][2] + 0.15:
                print('redundant motion, filtered out')
                _mark_episode_done(output_dir, episode_idx)
                episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0; continue

        step_idx += 1
        step_in_stage_idx += 1
        if step_in_stage_idx == traj[0].shape[0]:
            stage_idx += 1
            step_in_stage_idx = 0
            stage_boundaries.append(step_idx)

        if stage_idx > 3:
            os.makedirs(output_dir, exist_ok=True)
            success = bool(obs['success'])
            save_rgb_images_to_video(image_list, os.path.join(output_dir, f'demo_{episode_idx}_{success}.mp4'))

            if success:
                meta = {
                    'episode_idx': int(episode_idx),
                    'hand_mode': int(hand if grasp_mode in (3, 4) else grasp_mode),  # executed hand: 0/1/2
                    'bodex_mode': int(grasp_mode),  # original 0/1/2/3/4
                    'object_mesh_path': str(object_mesh_path),
                    'object_name': object_name,
                    'object_scale': float(object_scale),
                    'object_init_pose': env.object_init_pose.tolist(),
                    'object_final_pose': env.get_object_pose().tolist(),
                    'success': True,
                    'control_hz': int(env.control_hz),
                    'num_steps': int(step_idx),
                    'stage_boundaries': stage_boundaries,
                    'state_layout': 'qpos(18)+ee_pose(7)' if grasp_mode != 2 else 'qpos(18)+ee_pose(7)+qpos(18)+ee_pose(7)',
                    'action_layout': 'arm(6)+hand(12)' if grasp_mode != 2 else 'arm(6)+hand(12)+arm(6)+hand(12)',
                }
                _save_episode_npz(
                    os.path.join(output_dir, f'episode_{episode_idx:05d}.npz'),
                    episode_data, meta,
                )

            _mark_episode_done(output_dir, episode_idx)
            episode_idx += 1; step_idx = 0; stage_idx = 0; step_in_stage_idx = 0


@click.command()
@click.option('--hand', required=True, type=int)
@click.option('--object_scale_list', required=True, type=str)
@click.option('--object-root', default=None, type=str, help='Root dir for batch scanning, e.g. asset/object_mesh')
@click.option('--object-names', default=None, type=str, help='Comma-separated subset of object names to process')
@click.option('--output-root', default='outputs/batch_run', type=str, help='Root output dir for batch mode')
def main(hand, object_scale_list, object_root, object_names, output_root):
    import json as _json
    import traceback

    scale_list = eval(object_scale_list)
    env = BaseEnv(config)

    if object_root is None:
        # single-object legacy mode
        object_mesh_path = config['object_mesh_path']
        for object_scale in scale_list:
            rollout_for_an_object(env, hand, object_scale, object_mesh_path)
        return

    # batch mode
    object_root_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), object_root) if not os.path.isabs(object_root) else object_root

    # scan valid objects
    all_dirs = sorted(os.listdir(object_root_abs))
    if object_names:
        requested = [n.strip() for n in object_names.split(',')]
        all_dirs = [d for d in all_dirs if d in requested]

    valid_objects = []
    skipped_objects = []
    for d in all_dirs:
        mesh_path = os.path.join(object_root_abs, d, 'mesh', 'simplified.obj')
        if os.path.isfile(mesh_path):
            valid_objects.append(d)
        else:
            skipped_objects.append(d)
            print(f'[SKIP] {d}: mesh/simplified.obj not found')

    print(f'Found {len(valid_objects)} valid objects, skipped {len(skipped_objects)}')

    batch_results = {'processed': [], 'skipped': skipped_objects, 'failed': []}

    for obj_name in valid_objects:
        obj_dir = os.path.join(object_root_abs, obj_name)
        obj_results = {'object': obj_name, 'scales': {}}

        for scale in scale_list:
            scale_tag = f'scale_{scale}'
            output_dir = os.path.join(output_root, obj_name, scale_tag)
            skip_episodes = _get_completed_episodes(output_dir)
            if skip_episodes:
                print(f'  [RESUME] {len(skip_episodes)} episodes already done, skipping: {sorted(skip_episodes)}')
            print(f'\n=== Processing {obj_name} @ scale={scale} -> {output_dir} ===')
            try:
                rollout_for_an_object(env, hand, scale, obj_dir, output_dir, skip_episodes=skip_episodes)
                obj_results['scales'][scale_tag] = 'success'
            except Exception as e:
                msg = f'{type(e).__name__}: {e}'
                print(f'[FAIL] {obj_name} @ {scale}: {msg}')
                traceback.print_exc()
                obj_results['scales'][scale_tag] = f'failed: {msg}'
                batch_results['failed'].append({'object': obj_name, 'scale': scale, 'error': msg})

        # per-object summary
        os.makedirs(os.path.join(output_root, obj_name), exist_ok=True)
        with open(os.path.join(output_root, obj_name, 'summary.json'), 'w') as f:
            _json.dump(obj_results, f, indent=2)

        batch_results['processed'].append(obj_name)

    # batch summary
    os.makedirs(output_root, exist_ok=True)
    summary = {
        'total_objects': len(valid_objects) + len(skipped_objects),
        'processed': len(batch_results['processed']),
        'skipped': len(batch_results['skipped']),
        'failed_count': len(batch_results['failed']),
        'skipped_objects': batch_results['skipped'],
        'failed_details': batch_results['failed'],
    }
    with open(os.path.join(output_root, 'batch_summary.json'), 'w') as f:
        _json.dump(summary, f, indent=2)
    print(f'\nBatch done: {summary["processed"]} processed, {summary["skipped"]} skipped, {summary["failed_count"]} failed')
    print(f'Summary: {os.path.join(output_root, "batch_summary.json")}')


if __name__ == '__main__':
    main()
