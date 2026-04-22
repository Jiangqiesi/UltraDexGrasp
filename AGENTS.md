# AGENTS.md — UltraDexGrasp

## 项目概述

UltraDexGrasp 是一个用于双臂机器人通用灵巧抓取的数据/演示生成流水线（ICRA 2026）。它通过 BODex 合成抓取姿态，通过 cuRobo 规划运动轨迹，并在 SAPIEN 仿真环境中执行轨迹以生成演示视频和数据。

这不是一个训练仓库。没有模型训练、评估循环或打包流程。唯一的入口点用于生成抓取演示轨迹。

## 入口

```
python rollout.py --hand <0-4> --object_scale_list '[<float>, ...]'
```

手部模式：0=左手全手抓取，1=右手全手抓取，2=双手抓取，3=左手三指捏取，4=右手三指捏取。物体网格在 `env/config/env.yaml` 中的 `object_mesh_path` 配置。

## 架构

```
rollout.py                  # CLI 编排器（click）。抓取合成 → IK → 运动规划 → 环境步进 → 视频保存
env/
  base_env.py               # SAPIEN 场景、cuRobo RobotWorld、相机、碰撞检测、点云观测
  config/env.yaml           # 运行时配置：资产路径、机器人 URDF/cuRobo 配置、手部自由度、观测类型
  util/
    synthetic_pc_util.py    # SyntheticPC 类 — 从网格生成合成点云
    point_cloud_util.py     # 点云裁剪/保存辅助函数
    util.py                 # 环境数学辅助函数（注意：导入时有 print 副作用）
util/
  bodex_util.py             # GraspSynthesizer — 封装 BODex_api 进行抓取姿态优化
  curobo_util.py            # cuRobo 配置封装（IK 求解器、运动生成配置加载）
  util.py                   # 通用辅助函数：姿态转换、四元数运算、视频保存、抓取排序
```

第一方代码总计：8 个 Python 文件，约 1830 行。

## 关键依赖

| 包名 | 使用位置 | 备注 |
|------|---------|------|
| torch | 全局 | 未列入 requirements.txt — 需根据 CUDA 版本手动安装 |
| sapien==3.0.0b1 | base_env.py | 物理仿真引擎 |
| pytorch3d | base_env.py | `sample_farthest_points` 用于点云下采样 |
| cuRobo | rollout.py, base_env.py, curobo_util.py | IK 求解 + 运动生成 |
| BODex_api | bodex_util.py | 基于优化的抓取姿态合成 |
| click | rollout.py | CLI 参数解析 |
| trimesh | base_env.py | 网格加载/处理 |
| hydra-core | 通过 OmegaConf 加载配置 | |

三个重型依赖（pytorch3d、cuRobo、BODex_api）克隆到 `third_party/` 并以可编辑模式安装，需要 CUDA 和编译的 C++/CUDA 扩展。

## 环境搭建

需要 conda（用于 `coal`、`boost`）+ pip + 手动安装 third_party。详见 README.md。要点：
- Python 3.10，根据 CUDA 版本选择 torch wheel
- `third_party/` 中的仓库需逐个克隆并 `pip install -e .`
- `conda install coal boost` 安装 BODex 碰撞几何依赖
- `env/activate_uv.sh` 配置 .venv + COAL LD_LIBRARY_PATH

## 代码规范

- 未配置 linter、formatter 或类型检查器
- 全局无类型注解
- 4 空格缩进，snake_case 命名
- 项目模块使用绝对导入（如 `from util.bodex_util import GraspSynthesizer`）
- 使用 `print()` 输出 — 无日志框架
- 中英文混合注释；文档字符串稀少
- 无 `__init__.py` 文件 — 非标准 Python 包

## 测试

无第一方测试。无 pytest.ini、conftest.py 或测试目录。`third_party/` 中的包有各自的测试套件（cuRobo 用 pytest，pytorch3d 用 unittest），但属于外部代码。

`rollout.py` 本身充当集成测试 — 如果能生成视频，说明流水线正常工作。

## 构建 / CI

无。无 GitHub Actions、Makefile 或 Dockerfile。按 README 手动搭建。

## 注意事项

- `env/util/util.py` 在导入时有 `print(fovy)` 副作用 — 导入该模块会向 stdout 输出内容
- `util/bodex_util.py` 使用条件/内联导入（BODex_api 在方法内部导入）
- torch 是关键依赖但未列入 requirements.txt（需根据 CUDA 版本手动安装）
- `third_party/` 包含完整克隆的仓库（约 2700 个文件）— 除非调查上游问题，否则搜索时应排除
- 机器人配置引用 `asset/` 中的 URDF/YAML 文件 — 路径为相对路径，通过 cuRobo 配置系统解析
- `object_scale_list` CLI 参数是 Python 字面量字符串，通过 `eval()` 解析（在 rollout.py 中）

## 目录指引

| 路径 | 内容 | 何时修改 |
|------|------|---------|
| `rollout.py` | 主流水线编排器 | 添加新抓取策略或修改 rollout 循环 |
| `env/base_env.py` | 仿真环境 | 修改场景设置、相机、碰撞、观测 |
| `env/config/env.yaml` | 运行时配置 | 切换物体、机器人或观测类型 |
| `env/util/` | 点云工具 | 修改点云生成或处理逻辑 |
| `util/bodex_util.py` | 抓取合成封装 | 修改抓取优化参数 |
| `util/curobo_util.py` | 运动规划配置 | 修改 IK/运动生成配置 |
| `util/util.py` | 通用辅助函数 | 添加姿态/数学工具函数 |
| `asset/` | URDF、网格、配置文件 | 添加新机器人或物体 |
| `third_party/` | 外部依赖 | 仅在调试上游问题时修改 |
