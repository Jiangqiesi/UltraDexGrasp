# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UltraDexGrasp 是一个用于双臂灵巧手机器人通用抓取的数据生成框架（ICRA 2026）。通过基于优化的抓取合成器和基于规划的轨迹生成模块，支持多种抓取策略（双指捏取、三指三脚架、全手抓取、双手协作抓取），在仿真中生成抓取演示轨迹，同时输出 MP4 视频和结构化训练数据集（npz / zarr）。

## 环境配置

```bash
conda create -n ultradexgrasp python=3.10 -y
conda activate ultradexgrasp

# PyTorch（根据 CUDA 版本调整）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

# 第三方依赖（均需在 third_party/ 下 clone 并 pip install -e . --no-build-isolation）
# - pytorch3d
# - curobo (需 git-lfs)
# - BODex_api (额外需要 torch-scatter、conda install coal、conda install boost=1.84.0，以及编译 cpp 扩展)
```

## 运行命令

```bash
# 生成演示数据（单物体）
# --hand: 0=左手全手, 1=右手全手, 2=双手, 3=左手三指, 4=右手三指
python rollout.py --hand 0 --object_scale_list '[0.08]'

# 批量模式（扫描物体目录）
python rollout.py --hand 0 --object_scale_list '[0.08]' \
    --object-root asset/object_mesh \
    --output-root outputs/batch_run

# 将 npz 转为 DP3 兼容的 zarr 格式
python scripts/npz_to_zarr.py \
    --npz-root outputs/batch_run \
    --out-root datasets \
    --include-rgb   # 可选，存 RGB 流（量大）
```

物体 mesh 路径在 `env/config/env.yaml` 的 `object_mesh_path` 字段中配置（单物体模式）。

## 核心架构

### 数据流与执行管线

`rollout.py` 是唯一入口。执行流程：
1. 加载 `env/config/env.yaml` 配置，初始化 `GraspSynthesizer`（预热 BODex solver）
2. 创建 `BaseEnv` SAPIEN 仿真环境（双 UR5e + XHand）
3. 对每个 episode：合成抓取姿态 → IK 可达性过滤 → 碰撞过滤 → 按距离排序选最优抓取
4. 四阶段轨迹执行：init→pregrasp（带碰撞避障运动规划）→ grasp（接近）→ squeeze（合手指）→ lift（抬升）
5. 每阶段用 cuRobo MotionGen 规划臂部轨迹，手指关节线性插值
6. 成功判定：物体 z 坐标抬升 > 0.1m，输出 `demo_{idx}_{success}.mp4` + `episode_{idx:05d}.npz`（仅成功） + `episode_{idx:05d}.done`（所有终止路径，用于断点续传）

### 断点续传

批量模式下每个 episode 在**任何**终止路径（早期过滤、运动规划失败、物体移动、正常结束）都会写一个空的 `episode_{idx:05d}.done` 标记。重新运行同一命令时，`_get_completed_episodes()` 扫描 `output_dir/episode_*.done`，已标记的 episode 直接跳过（不调用 `env.reset()` 和 grasp synthesis）。

- 跳过粒度：成功和失败的 episode 都跳过（避免确定性失败的网格位置反复重试）
- 仅 batch 模式（`--object-root`）启用，单物体 legacy 模式不扫描（因为输出无 object/scale 子目录隔离）
- **迁移老数据**：本机制之前生成的数据没有 `.done` 标记，重跑会覆盖。为已有 `demo_*.mp4` 回填标记的一行命令：
  ```bash
  python3 -c "import os,re,glob; [open(os.path.join(os.path.dirname(p), f'episode_{int(re.match(r\"demo_(\d+)_\", os.path.basename(p)).group(1)):05d}.done'),'w').close() for p in glob.glob('outputs/batch_run/**/demo_*.mp4', recursive=True)]"
  ```

### 关键模块关系

- `env/base_env.py` (`BaseEnv`): SAPIEN 物理仿真环境。管理双臂机器人（robot_left/robot_right）、物体加载、相机、点云观测。内部维护两个 cuRobo `RobotWorld` 实例用于运动学和碰撞检测。
- `util/bodex_util.py` (`GraspSynthesizer`): 封装 BODex 抓取合成。输入物体 mesh/pose/scale，输出多个候选抓取姿态（含 pregrasp/grasp/squeeze 三个关键帧）。注意 BODex 的关节顺序与 SAPIEN 不同，通过 `bodex_2_sim_q_idx` 映射。
- `util/curobo_util.py` (`setup_curobo_utils`): 初始化 cuRobo 的 IK solver、FK model、MotionGen（分 common 和 lift 两套，lift 阶段无物体碰撞约束且位姿精度更宽松）。
- `env/util/synthetic_pc_util.py` (`SyntheticPC`): 用独立 SAPIEN 场景渲染机器人在指定关节角下的合成点云，用于观测中的机器人点云部分。
- `scripts/npz_to_zarr.py`: 将 rollout 输出的 npz 按 bodex_mode 分组转为 DP3 兼容的 zarr 格式。

### 重要设计细节

- `--hand` 参数 3/4（三指模式）在 `rollout_for_an_object` 开头映射回 0/1（`hand -= 3`），`grasp_mode` 变量保留原始值（0-4）。控制接口和 action 维度不变，只有 BODex 配置文件不同。
- 双手模式（hand=2）中 BODex 输出右手在前、左手在后的顺序，代码中做了交换。
- 机器人基座通过 `lock_joints`（6 个虚拟关节 pos_xyz + rot_xyz）设置世界坐标偏移，左臂基座 y=+0.45，右臂基座 y=-0.45，桌面高度 0.714m。
- 轨迹后处理会移除末尾的"停顿"帧（delta_ee_pose 过小的帧）。
- 物体初始化时每 4 个 episode 使用纯 z 轴随机旋转，其余 episode 使用完全随机旋转（单手）或离散 90 度组合旋转（双手）。
- `get_obs()` 内点云有随机采样（`np.random.permutation` + FPS `random_start_point=True`），同一物理状态调两次结果不同。rollout 主循环中 `obs = env.get_obs()` 的返回值被同时用于视频采集和数据记录，不额外调用第二次。

### obs-action 时序对齐

主循环的时序为：
```
obs_t  = env.get_obs()         # s_t，同时用于视频帧和数据记录
record(obs_t, joint_action)    # (s_t, a_t) 配对
obs_t1 = env.step(joint_action) # 执行 a_t，返回 s_{t+1}（内部也调 get_obs，但只取其 success 字段）
```
`env.step()` 内部调 `get_obs()` 但不用于记录，避免 `o_{t+1}` 对 `a_t` 的错位。

### 数据集格式

**per-episode npz**（`episode_{idx:05d}.npz`，仅成功轨迹）：

| 字段 | 形状 | 说明 |
|------|------|------|
| `point_cloud` | (T, 2400, 3) float32 | 场景点云，桌面坐标系 |
| `point_cloud_mask` | (T, 2400, 1) uint8 | 物体 mask（1=物体，0=背景/机器人） |
| `agent_pos` | (T, 25) or (T, 50) float32 | qpos(18)+ee_pose(7)，双手为 ×2 |
| `action` | (T, 18) or (T, 36) float32 | 关节目标位置（PD control 驱动目标） |
| `object_pose` | (T, 7) float32 | 物体位姿 pos+quat |
| `rgb_primary_0/1` | (T, H, W, 3) uint8 | 双目 RGB（可选） |
| `meta` | JSON string | episode 元数据，见下 |

meta 字段：`episode_idx`, `hand_mode`(执行侧 0/1/2), `bodex_mode`(原始 0-4), `object_name`, `object_scale`, `object_init_pose`, `object_final_pose`, `success`, `control_hz`(20), `num_steps`, `stage_boundaries`(各 stage 起始 step), `state_layout`, `action_layout`。

**zarr 数据集**（`scripts/npz_to_zarr.py` 转换后）按 bodex_mode 分组：
- `datasets/single_left/dataset.zarr` — bodex_mode 0 或 3
- `datasets/single_right/dataset.zarr` — bodex_mode 1 或 4
- `datasets/bimanual/dataset.zarr` — bodex_mode 2

zarr 结构：`data/{point_cloud, point_cloud_mask, agent_pos, action, object_pose, rgb_*}` + `meta/{episode_ends, bodex_mode, object_name, object_scale, object_init_pose, object_final_pose, success, num_steps}`。

### 对接 DP3 的额外工作

zarr schema 与 DP3 约定对齐，但还需：
1. 写 Dataset class（参考 `3D-Diffusion-Policy/diffusion_policy_3d/dataset/`），声明 `agent_pos`/`action` 维度
2. 写 task config yaml，指定 `shape_meta` 和归一化范围
3. 若要使用 `point_cloud_mask`，需在 point encoder 中自行拼接 `(N,3)+(N,1)` 为 `(N,4)`，DP3 默认 3 通道

### 配置结构

`env/config/env.yaml` 定义：
- `obs_type`: 观测类型列表（rgb, point_cloud）
- `xy_step_str`: episode 网格采样步数，如 `'[10, 15]'` 表示 150 个 episode
- `object_mesh_path`: 物体资产目录（需包含 `mesh/simplified.obj`, `mesh/coacd.obj`, `urdf/coacd.urdf`, `info/simplified.json`）
- `robot`: 左右手的 URDF、cuRobo 配置路径

### 物体资产格式

每个物体目录需包含：
- `mesh/simplified.obj` - 简化 mesh（用于抓取合成和碰撞检测）
- `mesh/coacd.obj` - CoACD 凸分解 mesh
- `urdf/coacd.urdf` - URDF 格式碰撞模型
- `info/simplified.json` - 包含 `gravity_center` 和 `obb` 信息
