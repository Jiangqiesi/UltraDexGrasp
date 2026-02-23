import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from bodex.geom.sdf.world import WorldConfig
from bodex.util.world_cfg_generator import get_world_config_dataloader
from bodex.util_file import load_yaml, load_json, get_configs_path


def pos_quat_to_mat(pos_quat):
    pos = pos_quat[:3]
    quat = pos_quat[3:]
    mat = np.eye(4) 
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return mat

class GraspSynthesizer:
    def __init__(self, hand, hand_type='xhand', dof=12, num_grasp=500):
        self.hand = hand
        self.hand_type = hand_type
        self.dof = dof
        if self.hand == 0:
            config_path = f'manip/sim_{hand_type}_sim2real/fc_left.yml'
        elif self.hand == 1:
            config_path = f'manip/sim_{hand_type}_sim2real/fc_right.yml'
        if self.hand == 3:
            config_path = f'manip/sim_{hand_type}_sim2real/fc_left_three_finger.yml'
            self.hand = 0
        elif self.hand == 4:
            config_path = f'manip/sim_{hand_type}_sim2real/fc_right_three_finger.yml'
            self.hand = 1
        elif self.hand == 2:
            config_path = f'manip/sim_{hand_type}_sim2real/fc_dual.yml'
        self.manip_config_data = load_yaml(f'{get_configs_path()}/{config_path}')
        self.manip_config_data['seed_num'] = num_grasp
        if hand_type == 'leap':
            self.bodex_2_sim_q_idx = [0, 12, 4, 8, 1, 13, 5, 9, 2, 14, 6, 10, 3, 15, 7, 11]
        elif hand_type == 'xhand':
            bodex_joint_order = ['right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2', 'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2', 'right_hand_mid_joint1', 'right_hand_mid_joint2', 'right_hand_ring_joint1', 'right_hand_ring_joint2', 'right_hand_pinky_joint1', 'right_hand_pinky_joint2']
            isaac_sim_joint_order = ['right_hand_index_bend_joint', 'right_hand_mid_joint1', 'right_hand_pinky_joint1', 'right_hand_ring_joint1', 'right_hand_thumb_bend_joint', 'right_hand_index_joint1', 'right_hand_mid_joint2', 'right_hand_pinky_joint2', 'right_hand_ring_joint2', 'right_hand_thumb_rota_joint1', 'right_hand_index_joint2', 'right_hand_thumb_rota_joint2']
            sapien_joint_order = ['right_hand_thumb_bend_joint', 'right_hand_index_bend_joint', 'right_hand_mid_joint1', 'right_hand_ring_joint1', 'right_hand_pinky_joint1', 'right_hand_thumb_rota_joint1', 'right_hand_index_joint1', 'right_hand_mid_joint2', 'right_hand_ring_joint2', 'right_hand_pinky_joint2', 'right_hand_thumb_rota_joint2', 'right_hand_index_joint2']
            self.bodex_2_sim_q_idx = [np.where(np.array(bodex_joint_order) == item)[0][0] for item in sapien_joint_order]
        else:
            raise ValueError(f'Invalid hand type: {hand_type}')
        self.grasp_solver = None

    def synthesize_grasp(self, object_path, object_pose, object_scale):
        if self.hand == 0 or self.hand == 1:
            from bodex.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
        elif self.hand == 2:
            from bodex.wrap.reacher.grasp_solver_bi import GraspSolver, GraspSolverConfig

        object_pose = np.array(object_pose)
        object_pos = object_pose[:3].copy()
        object_pose[:3] = 0.0
        object_pose = object_pose.tolist()
        obj_code = object_path.split('/')[-1]
        full_path = f'{object_path}/mesh/simplified.obj'
        manip_name = f"{obj_code}_scale{str(int(object_scale * 100)).zfill(3)}"

        world_cfg = {'cuboid': {'table': {'dims': [2.0, 2.0, 0.2], 'pose': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}}, 'mesh': {}}
        world_cfg["mesh"][manip_name] = {
            "scale": object_scale,
            "pose": object_pose,
            "file_path": full_path,
            "urdf_path": f'{object_path}/urdf/coacd.urdf',
        }

        json_data = load_json(f'{object_path}/info/simplified.json')

        world_info_dict = {
            "world_cfg": [world_cfg],
            "obj_code": [obj_code],
            "manip_name": [manip_name],
            "obj_path": [full_path],
            "obj_scale": torch.tensor([object_scale], dtype=torch.float64),
            "obj_pose": torch.tensor([object_pose], dtype=torch.float64),
            "obj_gravity_center": torch.from_numpy(np.array([object_pose[:3] + R.from_quat(object_pose[3:], scalar_first=True).as_matrix() @ json_data["gravity_center"] * object_scale])).to(torch.float64),
            "obj_obb_length": torch.tensor([object_scale * np.linalg.norm(json_data["obb"]) / 2], dtype=torch.float64),
        }

        # get object pose, scale, and mesh. then calculate the z_min to set the height of the table
        object_mesh_path = world_info_dict['obj_path'][0]
        object_pose = world_info_dict['obj_pose'][0].cpu().numpy()
        object_pose = pos_quat_to_mat(object_pose)
        object_scale = world_info_dict['obj_scale'][0].item()
        object_mesh = trimesh.load(object_mesh_path)
        object_mesh.apply_transform(object_pose)
        object_mesh.apply_scale(object_scale)
        z_min = object_mesh.bounds[0][2]
        world_info_dict['world_cfg'][0]['cuboid']['table']['pose'][2] = z_min - world_info_dict['world_cfg'][0]['cuboid']['table']['dims'][2] / 2 #+ 0.01  # raise the table by 1cm for better collision avoidance

        if self.grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                        world_model=world_info_dict['world_cfg'],
                        manip_name_list=world_info_dict['manip_name'],
                        manip_config_data=self.manip_config_data,
                        obj_gravity_center=world_info_dict['obj_gravity_center'],
                        obj_obb_length=world_info_dict['obj_obb_length'],
                        use_cuda_graph=False,
                        store_debug=False,
                    )
            self.grasp_solver = GraspSolver(grasp_config)
        else:
            world_model = [WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict['world_cfg']]
            self.grasp_solver.update_world(world_model, world_info_dict['obj_gravity_center'], world_info_dict['obj_obb_length'], world_info_dict['manip_name'])

        result = self.grasp_solver.solve_batch_env(return_seeds=self.grasp_solver.num_seeds)

        if self.hand == 0 or self.hand == 1:
            squeeze_scale = 1.0
            squeeze_pose_qpos = torch.cat([result.solution[..., 1, :7], result.solution[..., 1, 7:] + (result.solution[..., 1, 7:] - result.solution[..., 0, 7:]) * squeeze_scale], dim=-1)
        elif self.hand == 2:
            squeeze_scale = 0.4
            squeeze_pose_qpos = torch.cat([result.solution[..., 1, :] + (result.solution[..., 1, :] - result.solution[..., 0, :]) * squeeze_scale], dim=-1)
        all_hand_pose_qpos = torch.cat([result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2)
        all_hand_pose_qpos = all_hand_pose_qpos[0].cpu().numpy()

        if self.hand == 0 or self.hand == 1:
            grasp_pose = np.zeros([all_hand_pose_qpos.shape[0], 1, all_hand_pose_qpos.shape[1], 7 + self.dof])
            grasp_pose[:, 0, :, :7] = all_hand_pose_qpos[:, :, :7]
            grasp_pose[:, 0, :, 7:] = all_hand_pose_qpos[:, :, 7:][:, :, self.bodex_2_sim_q_idx]
        elif self.hand == 2:
            if self.hand_type == 'leap':
                grasp_pose = np.zeros([all_hand_pose_qpos.shape[0], 2, all_hand_pose_qpos.shape[1], 7 + self.dof])
                grasp_pose[:, 0, :, :3] = all_hand_pose_qpos[:, :, :3]
                grasp_pose[:, 0, :, 3:7] = R.from_euler('XYZ', all_hand_pose_qpos[:, :, 3:6].reshape([-1, 3]), degrees=False).as_quat(scalar_first=True).reshape([-1, 3, 4])
                grasp_pose[:, 0, :, 7:] = np.concatenate([all_hand_pose_qpos[:, :, 6:10], all_hand_pose_qpos[:, :, 32:44]], axis=-1)[:, :, self.bodex_2_sim_q_idx]
                grasp_pose[:, 1, :, :3] = all_hand_pose_qpos[:, :, 10:13]
                grasp_pose[:, 1, :, 3:7] = R.from_euler('XYZ', all_hand_pose_qpos[:, :, 13:16].reshape([-1, 3]), degrees=False).as_quat(scalar_first=True).reshape([-1, 3, 4])
                grasp_pose[:, 1, :, 7:] = all_hand_pose_qpos[:, :, 16:32][:, :, self.bodex_2_sim_q_idx]
            elif self.hand_type == 'xhand':
                # right first for dual xhand in bodex
                grasp_pose = np.zeros([all_hand_pose_qpos.shape[0], 2, all_hand_pose_qpos.shape[1], 7 + self.dof])
                grasp_pose[:, 1, :, :3] = all_hand_pose_qpos[:, :, :3]
                grasp_pose[:, 1, :, 3:7] = R.from_euler('XYZ', all_hand_pose_qpos[:, :, 3:6].reshape([-1, 3]), degrees=False).as_quat(scalar_first=True).reshape([-1, 3, 4])
                grasp_pose[:, 1, :, 7:] = all_hand_pose_qpos[:, :, 6:18][:, :, self.bodex_2_sim_q_idx]
                grasp_pose[:, 0, :, :3] = all_hand_pose_qpos[:, :, 18:21]
                grasp_pose[:, 0, :, 3:7] = R.from_euler('XYZ', all_hand_pose_qpos[:, :, 21:24].reshape([-1, 3]), degrees=False).as_quat(scalar_first=True).reshape([-1, 3, 4])
                grasp_pose[:, 0, :, 7:] = all_hand_pose_qpos[:, :, 24:36][:, :, self.bodex_2_sim_q_idx]

        grasp_pose[:, :, :, :3] += object_pos
        grasp_pose = grasp_pose.astype(np.float32)

        # filter out grasps with nan quat
        nan_mask = np.isnan(grasp_pose).any(axis=(1,2,3))
        if np.sum(nan_mask) > 0:
            print(f"[INFO] Dropping {np.sum(nan_mask)} nan grasps.")
        grasp_pose = grasp_pose[~nan_mask]

        return grasp_pose
