import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pxr import Gf


def save_rgb_images_to_video(images, output_filename, fps=30):
    import subprocess
    height, width, layers = images[0].shape
    command = ['ffmpeg',
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{width}x{height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_filename]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    process.wait()

def pos_quat_to_mat(pos_quat):
    pos = pos_quat[..., :3]
    quat = pos_quat[..., 3:]
    if len(pos_quat.shape) == 1:
        mat = np.eye(4)
    elif len(pos_quat.shape) == 2:
        mat = np.tile(np.eye(4), (pos_quat.shape[0], 1, 1))
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    return mat

def mat_to_pos_quat(mat):
    pos = mat[..., :3, 3]
    quat = R.from_matrix(mat[..., :3, :3]).as_quat(scalar_first=True)
    pos_quat = np.concatenate([pos, quat], axis=-1)
    return pos_quat

def calculate_angle_between_vector(a, b):
    """
    a: (B, 3)
    b: (B, 3)
    return: (B,) 每一对向量的夹角（弧度）
    """
    a = np.asarray(a)
    b = np.asarray(b)
    dot_product = np.einsum('ij,ij->i', a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cos_theta = dot_product / (norm_a * norm_b)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

def calculate_angle_between_quat(q1, q2_array):
    q1_rot = R.from_quat(q1, scalar_first=True)
    q2_rot = R.from_quat(q2_array, scalar_first=True)
    angle = (q1_rot.inv() * q2_rot).magnitude()

    return angle

def calculate_angle_between_quat_torch(q1, q2):
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(0.0, 1.0)
    vnorm = torch.sqrt(torch.clamp(1.0 - dot * dot, min=0.0))
    angle = 2.0 * torch.atan2(vnorm, torch.clamp(dot, min=1e-9))

    return angle

def calculate_pose_distance(anchor_pose, target_poses):
    pos_errs = np.linalg.norm(anchor_pose[:3] - target_poses[:, :3], axis=1)
    rot_errs = calculate_angle_between_quat(anchor_pose[3:7], target_poses[:, 3:7])
    distances = 1.0 * pos_errs + 1.0 * rot_errs

    return distances

def sort_grasp_for_single_hand(init_pose, grasp_poses):
    distances = calculate_pose_distance(init_pose, grasp_poses)
    sorted_idx = np.argsort(distances)

    return sorted_idx

def sort_grasp_for_dual_hand(init_pose_0, init_pose_1, grasp_poses_0, grasp_poses_1):
    distances_0 = calculate_pose_distance(init_pose_0, grasp_poses_0)
    distances_1 = calculate_pose_distance(init_pose_1, grasp_poses_1)
    x_z_diff = np.linalg.norm(grasp_poses_0[:, 0:1] - grasp_poses_1[:, 0:1], axis=-1) + np.linalg.norm(grasp_poses_0[:, 2:3] - grasp_poses_1[:, 2:3], axis=-1)
    distances = distances_0 + distances_1 + 20 * x_z_diff
    sorted_idx = np.argsort(distances)

    return sorted_idx

def composite_pose(base_pose, relative_pose):
    base_rot = R.from_quat(base_pose[..., 3:7], scalar_first=True)
    relative_rot = R.from_quat(relative_pose[..., 3:7], scalar_first=True)
    final_rot = base_rot * relative_rot
    rotated_relative_pos = base_rot.apply(relative_pose[..., :3])
    final_pos = rotated_relative_pos + base_pose[..., :3]
    final_quat = final_rot.as_quat(scalar_first=True)

    return np.concatenate([final_pos, final_quat], axis=-1)
