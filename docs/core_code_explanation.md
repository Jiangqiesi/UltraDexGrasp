# UltraDexGrasp 核心代码详解

## 1. 项目总览

UltraDexGrasp 是一个面向双臂机器人通用灵巧抓取的演示数据生成流水线（ICRA 2026）。项目不涉及模型训练，唯一目标是：给定物体网格，自动合成抓取姿态、规划运动轨迹、在仿真中执行并录制演示视频。

### 1.1 核心流程概览

```
CLI 输入 (--hand, --object_scale_list)
    │
    ▼
rollout.py main()
    ├── 创建 BaseEnv (SAPIEN 仿真环境)
    └── 对每个 object_scale 调用 rollout_for_an_object()
         │
         ├── 1. setup_curobo_utils() → 获取 IK 求解器、运动规划器
         ├── 2. env.set_object_path_and_scale_and_hand() → 配置物体
         └── 3. Episode 循环：
              ├── env.reset() → 重置场景
              ├── GraspSynthesizer.synthesize_grasp() → BODex 合成候选抓取
              ├── IK 求解 + 碰撞过滤 → 筛选可行抓取
              ├── sort_grasp_*() → 按距离排序选最优
              └── 4 阶段运动规划与执行：
                   ├── Stage 0: 初始位 → 预抓取位
                   ├── Stage 1: 预抓取位 → 抓取位
                   ├── Stage 2: 抓取位 → 握紧位
                   ├── Stage 3: 握紧位 → 抬起位
                   └── 保存演示视频
```

### 1.2 第一方代码文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `rollout.py` | 529 | CLI 入口 + 流水线编排器 |
| `env/base_env.py` | 537 | SAPIEN 仿真环境（场景、机器人、相机、观测） |
| `env/config/env.yaml` | 22 | 运行时配置（资产路径、机器人配置、观测类型） |
| `env/util/synthetic_pc_util.py` | 301 | 合成点云生成（SyntheticPC 类） |
| `env/util/point_cloud_util.py` | 38 | 点云裁剪/保存辅助函数 |
| `env/util/util.py` | 41 | 环境数学辅助（四元数角度、fovy 计算） |
| `util/bodex_util.py` | 153 | 抓取合成封装（GraspSynthesizer 类） |
| `util/curobo_util.py` | 125 | cuRobo IK/运动规划配置工厂 |
| `util/util.py` | 106 | 通用辅助（姿态转换、视频保存、抓取排序） |

---

## 2. rollout.py — 流水线编排器

### 2.1 模块级初始化（第 1–12 行）

脚本在 import 阶段执行两件事：

1. 加载 `env/config/env.yaml` 到全局 `config` 字典，并将 `asset_path` 和 `object_mesh_path` 转为绝对路径（第 1–7 行）
2. 通过 `sys.argv[2]` 提前创建 `GraspSynthesizer` 实例并执行一次 `synthesize_grasp`（第 8–12 行）

> ⚠️ 注意：第 2 步在 click 解析 CLI 参数之前执行，使用 `sys.argv[2]` 直接读取命令行参数。这是一个设计缺陷——如果命令行格式变化可能导致索引错误。

### 2.2 main() 函数（第 519–529 行）

```python
@click.command()
@click.option('--hand', required=True, type=int)
@click.option('--object_scale_list', required=True, type=str)
def main(hand, object_scale_list):
    env = BaseEnv(config)
    for object_scale in eval(object_scale_list):
        rollout_for_an_object(env, hand, object_scale)
```

- `--hand`：0=左手全手，1=右手全手，2=双手，3=左手三指，4=右手三指
- `--object_scale_list`：Python 字面量字符串，通过 `eval()` 解析为列表

### 2.3 rollout_for_an_object() — 核心流水线（第 37–516 行）

#### 2.3.1 手部模式映射（第 37–42 行）

```python
if hand == 3 or hand == 4:
    grasp_mode = copy.deepcopy(hand)
    hand -= 3  # 3→0(左), 4→1(右)
```

三指捏取模式在内部映射为单手模式（0 或 1），差异由 `GraspSynthesizer` 的配置文件区分。

#### 2.3.2 cuRobo 工具初始化（第 43 行）

```python
kin_model, ik_solver, motion_gen_common, motion_gen_lift = setup_curobo_utils(
    config_path=config['asset_path'],
    is_bimanual=(hand == 2),
    left_motion_gen_config_path=...,
    right_motion_gen_config_path=...,
    ...
)
```

返回四组工具，每组包含 `[左臂, 右臂]` 两个实例：
- `kin_model`：正运动学模型（CudaRobotModel），用于计算末端执行器位姿
- `ik_solver`：逆运动学求解器，批量求解关节角
- `motion_gen_common`：通用运动规划器（Stage 0–2）
- `motion_gen_lift`：抬起运动规划器（Stage 3，使用 `hold_partial_pose` 保持部分姿态）

#### 2.3.3 Episode 循环（第 53–516 行）

每个 episode 对应物体在桌面上的一个网格位置，由 `config['xy_step_str']` 决定网格密度。

**环境重置与状态获取（第 60–66 行）：**
```python
obs = env.reset(episode_idx)
# 检查物体是否在边界内
object_pose = env.get_object_pose()  # [x,y,z, w,x,y,z] 标量优先四元数
```

**运动学配置更新（第 68–86 行）：**

根据机器人基座在世界坐标系中的变换，计算 `base_offset_joints_dict`，更新 cuRobo IK 求解器的运动学配置，使其与 SAPIEN 中的机器人位姿一致。

**世界碰撞模型更新（第 88–115 行）：**

构建包含物体网格和桌面的 `world_config` 字典，更新两个 `robot_world` 实例的碰撞环境。

#### 2.3.4 抓取合成与筛选

**合成候选抓取（第 116 行）：**
```python
data_all = grasp_synthesizer.synthesize_grasp(object_mesh_path, object_pose.tolist(), object_scale)
```

`data_all` 的数据结构：
- 单手：`shape [N, 1, M, 7+dof]` — N 个候选，1 只手，M 个关键姿态，每个姿态含 7 维位姿 + dof 维手指关节角
- 双手：`shape [N, 2, M, 7+dof]` — 同上但包含左右手

其中 7 维位姿为 `[x, y, z, w, qx, qy, qz]`（标量优先四元数）。

**单手筛选流程（第 119–191 行）：**

1. 计算预抓取偏移（沿手掌法向后退一段距离）→ 预抓取位姿
2. 对预抓取位姿批量 IK 求解 → 过滤 IK 失败的候选
3. 拼接臂关节 + 手指关节 → 碰撞检测（`robot_world.get_world_self_collision_distance_from_joints`）
4. 过滤碰撞候选
5. `sort_grasp_for_single_hand()` 按与当前末端位姿的距离排序
6. 选取最优抓取，提取 `key_poses`（关键位姿序列）和 `key_qposes`（关键手指关节角序列）

**双手筛选流程（第 191–238 行）：**

与单手类似，但分别对左右手进行 IK 求解和碰撞检测，使用 `sort_grasp_for_dual_hand()` 综合排序（额外惩罚两手间距不合理的候选）。

#### 2.3.5 四阶段运动规划与执行（第 239–516 行）

| 阶段 | 目标 | 规划器 | 手指动作 |
|------|------|--------|---------|
| Stage 0 | 初始位 → 预抓取位 | motion_gen_common | 张开 → 张开 |
| Stage 1 | 预抓取位 → 抓取位 | motion_gen_common | 张开 → 抓取 |
| Stage 2 | 抓取位 → 握紧位 | motion_gen_common | 抓取 → 握紧 |
| Stage 3 | 握紧位 → 抬起位 | motion_gen_lift | 保持握紧 |

每个阶段的执行流程：

1. **阶段初始化**（`step_in_stage_idx == 0`，第 239–336 行）：重置运动规划器的世界缓存，更新运动学配置，Stage 3 使用 `PoseCostMetric` 的 `hold_partial_pose` 保持手指姿态
2. **运动规划**（第 338–464 行）：调用 `motion_gen[hand].plan_single(start_state, goal_pose, plan_config)` 生成轨迹
3. **轨迹后处理**：插值、去除末尾停顿帧（通过比较相邻帧末端位姿变化量，使用 `calculate_angle_between_quat_torch`）
4. **手指关节插值**（第 465–476 行）：`np.linspace` 在当前手指角和目标手指角之间线性插值
5. **环境步进**（第 477–491 行）：拼接臂关节 + 手指关节为 `joint_action`，调用 `env.step(joint_action)`，记录相机图像
6. **质量检查**（第 493–506 行）：检测物体是否意外移动、末端是否有冗余运动，不合格则丢弃该 episode

#### 2.3.6 错误处理模式

所有失败（IK 无解、碰撞过滤后无候选、运动规划失败、轨迹插值异常、物体移动）均通过 `print()` 输出信息后 `continue` 跳到下一个 episode。唯一的 `try/except` 包裹 `get_interpolated_trajectory`（第 365–369 行），使用裸 `except` 捕获所有异常。

---

## 3. env/base_env.py — SAPIEN 仿真环境

`BaseEnv` 是整个仿真环境的核心类，负责 SAPIEN 物理场景搭建、机器人加载、相机配置、观测生成和仿真步进。

### 3.1 构造函数与初始化（第 44–59 行）

```python
def __init__(self, config, with_object=True, control_hz=20, timestep=1/240, ray_tracing=False)
```

- `control_hz=20`：控制频率，`frame_skip = 1/(control_hz * timestep) = 12`，即每个控制步执行 12 次物理仿真
- 初始化流程：`set_up_physics_and_render()` → `set_up_scene()` → `scene.update_render()`

### 3.2 物理引擎与渲染配置（第 60–74 行）

```python
def set_up_physics_and_render(self, ray_tracing)
```

关键 PhysX 参数：
- `contact_offset=0.02, rest_offset=0.0`：碰撞检测偏移
- `solver_position_iterations=25`：位置求解器迭代次数（较高，保证精度）
- `gravity=[0,0,-9.81]`：标准重力
- `static_friction=2.0, dynamic_friction=2.0`：高摩擦系数（适合抓取场景）
- 可选光线追踪渲染（`ray_tracing=True` 时设置 RT shader 和降噪器）

### 3.3 场景搭建（第 75–246 行）

`set_up_scene()` 依次调用：

#### 3.3.1 灯光（第 83–89 行）
- 环境光 `[1.0, 1.0, 1.0]`
- 两个对称点光源位于桌面上方

#### 3.3.2 桌面（第 90–119 行）
- 运动学刚体（kinematic），半尺寸 `[0.6, 0.8, 0.03]`
- 放置在 `table_height` 高度
- 同时具有碰撞体和视觉体

#### 3.3.3 视觉边界（第 221–246 行）
- 红色细条标记桌面边界，仅用于可视化调试，无碰撞体

### 3.4 机器人加载（第 120–179 行）

```python
def set_up_robot(self)
```

加载两台 UR5E + 灵巧手（左/右）：

1. **URDF 加载**：通过 `scene.create_urdf_loader()` 加载，`fix_root_link=True` 固定基座
2. **关节配置**：
   - 驱动参数：`stiffness=1000, damping=100, force_limit=1e10, mode='force'`（位置控制模式）
   - 摩擦力设为 0
   - 所有连杆禁用重力（`link.disable_gravity = True`）
3. **关节名称映射**：
   - `LEFT_HAND_SIM_JOINT_ORDER` / `RIGHT_HAND_SIM_JOINT_ORDER`：SAPIEN 中手部关节的名称顺序（取活动关节的最后 `hand_dof` 个）
   - `LEFT_HAND_ROBOT_WORLD_JOINT_ORDER` / `RIGHT_HAND_ROBOT_WORLD_JOINT_ORDER`：cuRobo RobotWorld 中的关节顺序
   - `LEFT_HAND_SIM_2_ROBOT_WORLD_INDEX`：SAPIEN → RobotWorld 的关节索引映射
4. **初始关节角**：`init_qpos = [左臂6轴 + 左手dof, 右臂6轴 + 右手dof]`
5. **合成点云工具**：创建两个 `SyntheticPC` 实例用于生成机器人几何体的合成点云

### 3.5 cuRobo RobotWorld 集成

#### 3.5.1 初始化（第 247–276 行）

```python
def init_robot_world(self)
```

- 加载 `robot_world_config_path` 指定的 YAML 配置
- 通过 `RobotWorldConfig.load_from_config()` 创建两个 `RobotWorld` 实例
- 碰撞缓存初始化时传入静态场景网格和桌面长方体
- `collision_activation_distance=0.005`

#### 3.5.2 运动学同步（第 180–212 行）

```python
def init_robot(self)
```

- 设置 SAPIEN 机器人的根位姿（`robot_left_transformation` / `robot_right_transformation`）
- 计算基座偏移 `base_offset_joints_dict`（位置 + 旋转）
- 通过 `RobotConfig.from_dict()` 创建配置并调用 `robot_world[i].kinematics.update_kinematics_config()` 同步

> 关键点：SAPIEN 物理引擎和 cuRobo RobotWorld 是两套独立系统。BaseEnv 负责在每次 reset 时将 SAPIEN 中的机器人位姿同步到 cuRobo 的运动学模型中。

### 3.6 物体创建（第 369–416 行）

```python
def init_object(self, episode_idx)
```

1. 移除上一个物体实体
2. 随机采样物体朝向（绕 Z 轴旋转）
3. 根据 `episode_idx` 从网格位移数组中选取 XY 位置
4. 通过 trimesh 加载网格计算 `z_min`，确定物体放置高度
5. 创建 SAPIEN 动态刚体：`add_convex_collision_from_file` + `add_visual_from_file`
6. 质量计算：`mass = max(density * volume, 0.1)`

> ⚠️ 网格必须是 watertight（水密）的，否则 `trimesh.is_watertight` 检查失败会抛出 `RuntimeError`。

### 3.7 相机与观测系统

#### 3.7.1 相机配置（第 296–324 行）

- 创建 2 个相机：`Primary_0`、`Primary_1`
- 分辨率 640×480，fovy = π/3
- 固定位姿，从不同角度俯视桌面

#### 3.7.2 观测生成 get_obs()（第 440–529 行）

这是最复杂的方法，生成完整的观测字典：

**机器人状态（第 441–462 行）：**
```python
obs['robot_0'] = {
    'qpos': numpy array,      # 全部关节角
    'qvel': numpy array,      # 全部关节速度
    'ee_pose': numpy array(7)  # 末端位姿 [x,y,z,w,qx,qy,qz]
}
```

末端位姿通过 cuRobo 正运动学计算：将前 6 个关节角转为 torch tensor → `robot_world.get_kinematics()` → 提取位置和四元数。

**相机观测（第 464–492 行）：**

对每个相机执行 `camera.take_picture()`，根据 `obs_type` 提取：
- `'rgb'`：`camera.get_picture('Color')[:,:,:3]` → uint8
- `'depth'`：`-position[..., 2]`（从 Position 图的 Z 通道取反）
- `'point_cloud'`：
  1. 获取 Position 图（OpenGL 相机坐标系）
  2. 过滤无效像素（`position[..., 3] < 1`）
  3. 通过 `camera.get_model_matrix()` 变换到世界坐标系
  4. 获取 Segmentation 图，构建物体掩码

**点云融合与下采样（第 495–527 行）：**

1. 拼接多相机点云，减去 `table_height` 归一化
2. 裁剪到感兴趣区域（`crop_point_cloud`）
3. 随机采样至 2000 点 → `real_pc`
4. 获取合成机器人点云：`synthetic_pc_left.get_pc_at_qpos(qpos)` → 变换到世界坐标系 → 采样
5. 获取合成桌面点云
6. 拼接所有点云 → 再次裁剪 → `pytorch3d.ops.sample_farthest_points` 最远点采样至最终数量
7. 存入 `obs['point_cloud']`

> ⚠️ 最远点采样使用 `.cuda()` 推送到 GPU，要求 CUDA 可用。

### 3.8 运行循环

#### reset()（第 333–346 行）
```
init_robot() → init_object() → init_camera() → warm_up() → 记录 object_init_pose → get_obs()
```

#### warm_up()（第 420–426 行）
- 以 `init_qpos` 为目标执行 20 次控制循环（每次 `frame_skip` 步物理仿真），使机器人稳定

#### step()（第 427–439 行）
```python
def step(self, action, get_obs=True):
    self.apply_action(action)           # 设置关节驱动目标
    for _ in range(self.frame_skip):    # 12 步物理仿真
        self.scene.step()
    self.scene.update_render()
    obs = self.get_obs()
    obs['success'] = (object_z - init_z > 0.1)  # 抬起超过 10cm 视为成功
    return obs
```

#### check_object_moved()（第 534–537 行）
- 位置偏移 > 0.01m 或旋转角度 > π/18（10°）则判定物体被意外移动

---

## 4. 工具模块详解

### 4.1 util/bodex_util.py — 抓取合成封装

#### pos_quat_to_mat(pos_quat)（第 11–17 行）

将 7 维位姿向量（位置 + 标量优先四元数 `[w,x,y,z]`）转换为 4×4 齐次变换矩阵。使用 `scipy.spatial.transform.Rotation`。

#### class GraspSynthesizer

**构造函数（第 20–47 行）：**

```python
def __init__(self, hand, hand_type='xhand', dof=12, num_grasp=500)
```

- `hand`：手部索引（0–4），3/4 在内部重映射为 0/1 并选择三指配置文件
- `hand_type`：`'xhand'` 或 `'leap'`，决定关节顺序映射
- `num_grasp`：BODex 优化的种子数量
- 加载 BODex 配置 YAML（`bodex.util_file.load_yaml`）
- 构建 `bodex_2_sim_q_idx`：BODex 关节顺序 → 仿真关节顺序的索引映射

**synthesize_grasp()（第 49–153 行）：**

```python
def synthesize_grasp(self, object_path, object_pose, object_scale) -> numpy.ndarray
```

核心流程：
1. 根据手部模式动态导入 BODex 的 `GraspSolver`（单手）或 `GraspSolverBi`（双手）
2. 构建世界配置：物体网格路径、URDF、桌面信息
3. 加载 `simplified.json` 获取重心和 OBB 尺寸
4. 通过 trimesh 计算缩放后网格的 `z_min`，确定桌面高度
5. 懒初始化：首次调用创建 `GraspSolver`，后续调用 `update_world()` 更新
6. 调用 `solver.solve_batch_env()` 生成抓取方案
7. 后处理：生成"握紧"姿态（在两个关键帧间外推）、重排关节顺序（`bodex_2_sim_q_idx`）、偏移位置、过滤 NaN

**返回数据格式：**
- 单手：`[N, 1, M, 7+dof]` — N 个有效抓取，M 个关键姿态
- 双手：`[N, 2, M, 7+dof]` — 包含左右手
- dtype: `float32`

> ⚠️ 四元数全局使用标量优先 `[w, x, y, z]` 约定。

### 4.2 util/curobo_util.py — cuRobo 配置工厂

```python
def setup_curobo_utils(
    config_path, interpolation_dt=0.05, is_bimanual=False,
    left_motion_gen_config_path=..., right_motion_gen_config_path=...,
    left_ik_solver_config_path=..., right_ik_solver_config_path=...,
    device='cuda:0'
) -> (kin_model, ik_solver, motion_gen_common, motion_gen_lift)
```

**返回值（均为 `[左, 右]` 列表）：**

| 返回值 | 类型 | 用途 |
|--------|------|------|
| `kin_model` | `CudaRobotModel` | 正运动学，计算末端位姿 |
| `ik_solver` | `IKSolver` | 逆运动学批量求解 |
| `motion_gen_common` | `MotionGen` | Stage 0–2 运动规划 |
| `motion_gen_lift` | `MotionGen` | Stage 3 抬起规划（宽松位姿约束） |

**关键实现细节：**
- 加载 YAML 后重写 `urdf_path` 和 `asset_root_path` 为绝对路径
- `motion_gen_common` 初始化时传入一个示例场景网格以预热碰撞缓存，并调用 `warmup()`
- `motion_gen_lift` 在非双手模式下使用更宽松的位置/旋转阈值
- 所有张量通过 `TensorDeviceType(device)` 确保在指定 GPU 上

### 4.3 util/util.py — 通用辅助函数

| 函数 | 签名 | 用途 | 调用位置 |
|------|------|------|---------|
| `save_rgb_images_to_video` | `(images, output_filename, fps=30)` | 通过 ffmpeg 子进程将 RGB 图像列表写入视频 | rollout.py 第 514 行 |
| `pos_quat_to_mat` | `(pos_quat)` | 位姿→变换矩阵（支持批量） | 内部使用 |
| `mat_to_pos_quat` | `(mat)` | 变换矩阵→位姿（支持批量） | 内部使用 |
| `calculate_angle_between_vector` | `(a, b)` | 批量计算 3D 向量夹角（弧度） | 内部使用 |
| `calculate_angle_between_quat` | `(q1, q2_array)` | scipy 四元数角度差 | 内部使用 |
| `calculate_angle_between_quat_torch` | `(q1, q2)` | torch GPU 四元数角度差 | rollout.py 第 374、443 行 |
| `calculate_pose_distance` | `(anchor, targets)` | 位置 L2 + 旋转角度加权距离 | 排序函数内部 |
| `sort_grasp_for_single_hand` | `(init_pose, grasp_poses)` | 按位姿距离排序候选抓取 | rollout.py 第 180 行 |
| `sort_grasp_for_dual_hand` | `(init0, init1, grasps0, grasps1)` | 双手综合排序（含间距惩罚） | rollout.py 第 233 行 |
| `composite_pose` | `(base_pose, relative_pose)` | 位姿复合（基座 × 相对） | rollout.py 内部 |

> ⚠️ `pos_quat_to_mat` 在 `util/util.py` 和 `util/bodex_util.py` 中重复实现。`save_rgb_images_to_video` 在 `util/util.py` 和 `env/util/util.py` 中也有重复。

### 4.4 env/util/synthetic_pc_util.py — 合成点云生成

#### class SyntheticPC

```python
def __init__(self, urdf_path, image_size=[256, 256])
```

创建一个独立的 SAPIEN 场景，加载机器人 URDF，配置 8 个环绕相机，用于渲染机器人几何体的合成点云（用于遮挡模拟）。

**核心方法：**

| 方法 | 功能 |
|------|------|
| `setup_scene()` | 创建 SAPIEN 场景、灯光、相机 |
| `load_robot(urdf_path)` | 加载 URDF，配置关节驱动 |
| `setup_camera()` | 添加 8 个硬编码位置的环绕相机 |
| `get_synthetic_table_pc()` | 生成桌面区域的密集网格点云（80×240 分辨率） |
| `get_pc(num_point=None)` | 多相机拍照 → 过滤背景 → 变换到世界坐标 → 可选 FPS 下采样 |
| `get_pc_at_qpos(qpos, num_point)` | 设置关节角后调用 `get_pc()`，返回该姿态下的机器人点云 |

**在 base_env.py 中的使用：**
- 创建两个实例（左/右手，第 176–179 行）
- 在 `get_obs()` 中调用 `get_pc_at_qpos()` 获取机器人合成点云（第 505、510 行）
- 合成点云与真实相机点云融合后进行最远点采样

> ⚠️ `get_pc()` 中的 FPS 下采样使用 `pytorch3d.ops.sample_farthest_points` 并调用 `.cuda()`，要求 GPU 可用。

### 4.5 env/util/point_cloud_util.py — 点云辅助

| 函数 | 功能 | 调用位置 |
|------|------|---------|
| `add_gaussian_noise(pc, sigma=0.02)` | 逐点乘性高斯噪声 | 未使用（可用于数据增强） |
| `save_pc_as_ply(pc, path)` | 保存 Nx3/Nx6 点云为 PLY 文件 | base_env.py 导入但注释掉 |
| `crop_point_cloud(pc, boundary)` | 按轴对齐包围盒裁剪点云 | base_env.py 第 501、522 行 |

### 4.6 env/util/util.py — 环境数学辅助

| 函数 | 功能 | 调用位置 |
|------|------|---------|
| `save_rgb_images_to_video(images, filename, fps)` | ffmpeg 视频保存（与 util/util.py 重复） | 未被 base_env 使用 |
| `calculate_angle_between_quat(q1, q2_array)` | 四元数旋转角度差 | base_env.py 第 536 行 |
| `calculate_fovy(fy, image_height)` | 焦距→视场角（度） | 仅模块内使用 |

> ⚠️ 该文件在导入时执行 `print(fovy)` 副作用（第 38–42 行）。由于 `base_env.py` 在顶部导入此模块，创建 `BaseEnv` 时会向 stdout 输出一行内容。

---

## 5. 数据流总结

### 5.1 关键数据结构

| 数据 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `data_all`（候选抓取） | numpy float32 | `[N, H, M, 7+dof]` | H=手数(1/2), M=关键姿态数, 7=位姿, dof=手指自由度 |
| `key_poses`（选中抓取位姿） | numpy | `[H, M, 7]` | 最优抓取的关键位姿序列 |
| `key_qposes`（选中手指角） | numpy | `[H, M, dof]` | 最优抓取的关键手指关节角序列 |
| `traj`（臂轨迹） | torch tensor | `[T, 6]` per arm | 运动规划输出的关节角轨迹 |
| `qposes`（手指轨迹） | numpy | `[1, T, dof]` | 线性插值的手指关节角轨迹 |
| `joint_action` | numpy | `[arm_dof+hand_dof] × 2` | 拼接后的完整关节指令 |
| `obs` | dict | — | 包含 robot_0/1 状态、相机图像、点云 |

### 5.2 模块间调用关系

```
rollout.py
  ├── util/bodex_util.GraspSynthesizer
  │     └── BODex_api (GraspSolver / GraspSolverBi)
  ├── util/curobo_util.setup_curobo_utils
  │     └── cuRobo (IKSolver, MotionGen, CudaRobotModel)
  ├── util/util (排序、四元数计算、视频保存)
  └── env/base_env.BaseEnv
        ├── SAPIEN (Scene, PhysX, Renderer, Camera)
        ├── cuRobo RobotWorld (碰撞检测、正运动学)
        ├── env/util/synthetic_pc_util.SyntheticPC
        │     └── pytorch3d (最远点采样)
        ├── env/util/point_cloud_util (裁剪)
        └── env/util/util (四元数角度)
```

### 5.3 四元数约定

全项目统一使用**标量优先**四元数：`[w, x, y, z]`。所有 `scipy.Rotation` 调用均传入 `scalar_first=True`。在阅读或修改涉及四元数的代码时务必注意此约定。

### 5.4 已知代码重复

| 函数 | 位置 1 | 位置 2 |
|------|--------|--------|
| `pos_quat_to_mat` | `util/util.py:27` | `util/bodex_util.py:11` |
| `save_rgb_images_to_video` | `util/util.py:7` | `env/util/util.py:5` |
| `save_pc_as_ply` | `env/util/synthetic_pc_util.py:8` | `env/util/point_cloud_util.py:10` |

### 5.5 GPU 依赖点

以下操作要求 CUDA 可用：
1. `SyntheticPC.get_pc()` 中的 `pytorch3d.ops.sample_farthest_points`（`.cuda()`）
2. `BaseEnv.get_obs()` 中的最远点采样（`.cuda()`）
3. `BaseEnv.get_obs()` 中的 cuRobo 正运动学（`torch.tensor(...).cuda()`）
4. `setup_curobo_utils()` 中所有 cuRobo 组件（`TensorDeviceType(device='cuda:0')`）
5. `rollout.py` 中的 IK 求解和运动规划（cuRobo 内部使用 CUDA）

---

## 6. 配置文件 env/config/env.yaml

```yaml
obs_type: ['rgb', 'point_cloud']    # 观测类型列表
xy_step_str: '[10, 15]'             # 物体放置网格 [x步数, y步数]

object_mesh_path: asset/object_mesh/bowl  # 物体网格目录

asset_path: "asset"                 # 资产根目录
robot:
  ur5e_with_left_hand:
    hand_type: "xhand"
    hand_dof: 12
    curobo_motion_gen_config_path: '...'
    curobo_ik_solver_config_path: '...'
    robot_world_config_path: '...'
    urdf_path: '...'
  ur5e_with_right_hand:
    # 同上，右手配置
```

关键配置项说明：
- `obs_type`：控制 `get_obs()` 生成哪些观测模态
- `xy_step_str`：通过 `eval()` 解析为列表，决定每个 object_scale 下的 episode 数量
- `object_mesh_path`：物体网格目录，需包含 `mesh/simplified.obj`、`urdf/coacd.urdf`、`info/simplified.json`
- `robot` 下的路径均相对于 `asset_path`，由代码拼接为绝对路径
