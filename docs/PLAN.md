# 批量物体输入改造方案

## Summary
将当前“单物体 + 多 scale”的入口扩展为“目录扫描批量物体 + 全局 scale 列表”的批处理模式，默认扫描 `asset/object_mesh/*/mesh/simplified.obj`，对每个物体依次复用现有 rollout 流程。输出按物体分目录保存，单个物体或单个 scale 失败时记录并继续，不中断整批任务。

## Key Changes
- 扩展 `rollout.py` CLI，保留现有 `--hand` 和 `--object_scale_list`，新增批量输入参数：
  - `--object-root`：物体根目录，默认可设为 `asset/object_mesh`
  - `--object-names`：可选，仅处理指定物体名子集；未传时扫描全部
  - `--output-root`：批量输出根目录
- 将当前依赖全局 `config['object_mesh_path']` 的单物体逻辑拆成“每次 rollout 显式接收 object_dir/object_mesh_path”：
  - `env.set_object_path_and_scale_and_hand(...)` 已支持动态换物体，可继续复用
  - BODex 输入、cuRobo world mesh 路径、视频命名都改为使用当前循环物体，而不是固定读 `env.yaml`
- 新增目录扫描约定：
  - 仅把存在 `mesh/simplified.obj` 的子目录视为有效物体
  - 物体 ID 直接使用子目录名，如 `bowl`、`drill`
  - 无效目录记录为 skipped，不报错退出
- 调整输出组织，避免覆盖：
  - 输出结构建议为 `<output_root>/<object_name>/scale_<scale>/`
  - 视频文件保留 episode/success 信息，如 `demo_<episode>_<success>.mp4`
  - 每个物体目录下额外写一个简短结果摘要，记录成功/失败的 scale 与失败原因
  - 批次根目录写总汇总文件，包含 processed/skipped/failed 计数
- 失败处理改为批处理友好：
  - 单个物体某个 scale 抛错时捕获异常并记录
  - 继续后续 scale 和后续物体
  - 最终返回批量统计，而不是中途退出
- 配置层保持最小改动：
  - `env/config/env.yaml` 中的 `object_mesh_path` 退化为单物体默认值或兼容旧模式
  - 批量模式下不再把它作为唯一数据源

## Public Interface Changes
- CLI 从单入口扩展为双模式：
  - 旧模式：`python rollout.py --hand <0-4> --object_scale_list '[...]'`
  - 新批量模式：`python rollout.py --hand <0-4> --object_scale_list '[...]' --object-root asset/object_mesh --output-root outputs/batch_run`
- 若同时传 `--object-root` 与旧的单物体配置，批量模式优先；未传 `--object-root` 时继续兼容当前单物体行为

## Test Plan
- 单物体回归：
  - 用当前 `drill` 跑一次，确认旧命令仍可执行，结果路径正确
- 批量扫描：
  - 扫描现有 `asset/object_mesh`，确认识别到 `bowl/can/drill/ship/ship3`
  - 人为加入一个缺少 `mesh/simplified.obj` 的目录，确认被跳过并记录
- 多物体多 scale：
  - 至少选 2 个物体、2 个 scale，确认每个组合都会独立输出到对应目录，不发生覆盖
- 失败继续：
  - 人为指定一个不存在或损坏的物体目录，确认该项失败被记录，后续物体继续执行
- 汇总结果：
  - 检查批次总汇总与物体级摘要中的成功/失败/跳过计数是否一致

## Assumptions
- 目录扫描按现有仓库约定进行：`asset/object_mesh/<object_name>/mesh/simplified.obj`
- 所有物体共享同一组 `--object_scale_list`
- 输出按物体分目录保存
- 单项失败采用“记录后继续”策略
- 本次改造只覆盖批量物体输入与结果组织，不额外引入并行调度、任务队列或 manifest 配置文件
