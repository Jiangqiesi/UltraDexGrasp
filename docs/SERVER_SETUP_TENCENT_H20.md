# TencentOS + H20 服务器环境配置

本文档面向这台服务器：

- OS: TencentOS Server 3.2, RHEL/CentOS 8 系
- Kernel: 5.4.241 tlinux4
- GPU: 8 * NVIDIA H20
- CUDA toolkit: `/usr/local/cuda-12.8`, `nvcc 12.8.93`
- Python 环境工具: `uv 0.11.8`

结论先说：不要直接照 README 的通用安装命令执行。README 默认 Ubuntu/apt，并示例安装 PyTorch `cu118`。这台机器的本地编译器是 CUDA 12.8，而 PyTorch3D、cuRobo、BODex 都会编译 CUDA/C++ 扩展；如果用 `cu118` wheel 再拿 CUDA 12.8 的 `nvcc` 编译，容易触发 CUDA major version mismatch。推荐使用 Python 3.10 + PyTorch 2.4.1 + CUDA 12.4 wheel，并用本机 CUDA 12.8 编译扩展。这样保持 PyTorch major CUDA 版本为 12，同时贴近 PyTorch3D/BODex 的 2.4.1 兼容区间。

## README 直接配置的主要风险

1. OS 包管理器不匹配
   - README 使用 `sudo apt install git-lfs`。
   - TencentOS 是 RHEL/CentOS 系，应使用 `dnf` 或 `yum`。

2. CUDA/PyTorch 版本不匹配
   - README 示例是 `torch==2.4.1` + `cu118`。
   - 本机 `nvcc` 是 12.8，编译 PyTorch3D/cuRobo/BODex 扩展时可能因为 PyTorch CUDA 11.x 与本机 CUDA 12.x major mismatch 失败。
   - 本文推荐 `torch==2.4.1` + `cu124`，再用 CUDA 12.8 编译。minor mismatch 通常只是警告。

3. 最新 PyTorch CUDA 12.8 路线风险更高
   - PyTorch 官方已有 `cu128` wheel，但本项目当前 third_party 依赖按 PyTorch 2.4.1 路线写得更明确。
   - 若直接上最新 PyTorch，PyTorch3D、BODex、torch-scatter 的 ABI/API 风险更大。

4. COAL/Boost 链接路径容易丢
   - BODex 的 `coal_openmp_wrapper` 会链接 `coal`、`boost_filesystem`、`qhull`、`octomap`、`assimp` 等 native 库。
   - 必须在安装 wrapper 和运行 rollout 前设置好 `COAL_PREFIX`/`LD_LIBRARY_PATH`。仓库已有 `env/activate_uv.sh`，本文会使用它。

5. `ffmpeg` 不在 Python requirements 中
   - `util/util.py` 直接调用系统命令 `ffmpeg` 保存 mp4。
   - 没有 `ffmpeg` 时 rollout 后期保存视频会失败。

6. SAPIEN headless 渲染依赖 Vulkan/NVIDIA 驱动栈
   - Python 包安装成功不等于 SAPIEN 可以渲染。
   - 需要检查 Vulkan loader 和 NVIDIA ICD。

7. 8 卡不会被单进程自动吃满
   - 代码里大量使用 `.cuda()`，默认只用当前可见的第 0 张 GPU。
   - 多卡跑数据要用多个进程配合 `CUDA_VISIBLE_DEVICES` 和不同输出目录。

8. `rollout.py` 参数顺序敏感
   - 文件开头有 `hand = eval(sys.argv[2])`，因此必须保持 `--hand <id>` 放在第一个参数位置。
   - 推荐命令形态：`python rollout.py --hand 0 --object_scale_list '[0.08]'`。

## 0. 基础系统包

如果当前用户有 root 权限：

```bash
dnf install -y \
  git git-lfs gcc gcc-c++ make cmake ninja-build \
  libgomp mesa-libGL glib2 libX11 libXext libXrender libSM libICE \
  vulkan-loader vulkan-tools

git lfs install
```

如果没有 `dnf`，改用 `yum`。如果 `ffmpeg` 可以从系统源安装，也一并装上：

```bash
dnf install -y ffmpeg || yum install -y ffmpeg
```

很多 CentOS/RHEL 系环境默认源没有 `ffmpeg`。这种情况下后面用 `imageio-ffmpeg` 给 `.venv/bin/ffmpeg` 做一个本地兜底。

## 1. 进入仓库并设置 CUDA

```bash
cd /path/to/UltraDexGrasp

export CUDA_HOME=/usr/local/cuda
export CUDA128_HOME=/usr/local/cuda
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

nvcc --version
```

预期看到 CUDA 12.8。

建议把构建临时目录放到本机盘，避免在共享文件系统上编译太慢：

```bash
export TMPDIR=/tmp/${USER:-root}/ultradexgrasp-build
export TORCH_EXTENSIONS_DIR=/tmp/${USER:-root}/torch_extensions
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
```

## 2. 创建 uv 虚拟环境

```bash
uv python install 3.10
uv venv .venv --python 3.10
source .venv/bin/activate

uv pip install -U pip setuptools wheel packaging ninja cmake pybind11
```

不要把依赖装进系统 Python。服务器是 root 环境时尤其要避免污染全局环境。

## 3. 安装 PyTorch

推荐 PyTorch 2.4.1 + cu124：

```bash
uv pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

检查 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

H20 通常按 Hopper `sm_90` 编译即可：

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16
```

如果上面的 Python 检查打印的 capability 不是 `(9, 0)`，按实际值修改 `TORCH_CUDA_ARCH_LIST`。

## 4. 安装项目 Python 依赖

先固定 NumPy 到 BODex 更保守的 1.25.x，再装本项目 requirements：

```bash
uv pip install numpy==1.25.2
uv pip install -r requirements.txt
uv pip install coal pybind11
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

如果系统没有 `ffmpeg`，使用 Python wheel 兜底：

```bash
uv pip install imageio-ffmpeg
ln -sf "$(python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)" .venv/bin/ffmpeg

ffmpeg -version
```

可选工具，只在处理网格或转 zarr 时需要：

```bash
uv pip install coacd lxml zarr
```

## 5. 准备 third_party

如果服务器仓库还没有 `third_party/`，按下面克隆并固定到当前工作区验证过的提交。固定提交可以避免上游 main 分支变化带来的新问题。

```bash
mkdir -p third_party

git clone https://github.com/facebookresearch/pytorch3d.git third_party/pytorch3d
git -C third_party/pytorch3d checkout b6a77ad

git clone https://github.com/NVlabs/curobo.git third_party/curobo
git -C third_party/curobo checkout d64c4b0

git clone https://github.com/yangsizhe/BODex_api.git third_party/BODex_api
git -C third_party/BODex_api checkout 6072a07
```

如果这些目录已经存在：

```bash
git -C third_party/pytorch3d rev-parse --short HEAD
git -C third_party/curobo rev-parse --short HEAD
git -C third_party/BODex_api rev-parse --short HEAD
git -C third_party/curobo lfs pull || true
git -C third_party/BODex_api lfs pull || true
```

## 6. 编译安装 PyTorch3D、cuRobo、BODex

```bash
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=16
export TMPDIR=/tmp/${USER:-root}/ultradexgrasp-build
export TORCH_EXTENSIONS_DIR=/tmp/${USER:-root}/torch_extensions
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

FORCE_CUDA=1 uv pip install -e third_party/pytorch3d --no-build-isolation
uv pip install -e third_party/curobo --no-build-isolation
uv pip install -e third_party/BODex_api --no-build-isolation
```

安装 BODex 的 COAL OpenMP wrapper。这里先使用仓库提供的激活脚本，它会根据 pip 安装的 `coal` 计算 `COAL_PREFIX` 并设置 `LD_LIBRARY_PATH`：

```bash
source env/activate_uv.sh

cd third_party/BODex_api/src/bodex/geom/cpp
python setup.py install
cd -
```

以后每次运行 rollout 前都建议执行：

```bash
source env/activate_uv.sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

## 7. 安装验证

```bash
source env/activate_uv.sh

python - <<'PY'
import torch
import sapien
import pytorch3d
import curobo
import bodex
import coal
import coal_openmp_wrapper
from pytorch3d.ops import sample_farthest_points
from pxr import Gf

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("imports ok")
PY
```

检查 Vulkan：

```bash
vulkaninfo --summary | head -80
find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -name '*nvidia*.json' -print 2>/dev/null
```

如果 SAPIEN 报 Vulkan/renderer 相关错误，优先确认 NVIDIA 驱动、`vulkan-loader`、NVIDIA ICD JSON 是否存在。必要时设置：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

路径以服务器实际 `find` 结果为准。

## 8. 最小运行测试

单卡 smoke test：

```bash
source env/activate_uv.sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0 python rollout.py --hand 0 --object_scale_list '[0.08]'
```

注意：

- `--hand 0` 必须紧跟在脚本名后面，原因是 `rollout.py` 开头读取 `sys.argv[2]`。
- `object_scale_list` 是 Python 字面量字符串，外层引号不能省。
- 默认物体来自 `env/config/env.yaml` 的 `object_mesh_path`。

当前分支的 `rollout.py` 也支持批量物体参数：

```bash
CUDA_VISIBLE_DEVICES=0 python rollout.py \
  --hand 0 \
  --object_scale_list '[0.08]' \
  --object-root asset/object_mesh \
  --object-names drill,bowl \
  --output-root outputs/h20_smoke_gpu0
```

## 9. 8 卡运行建议

单个 Python 进程默认只用一张可见 GPU。要用满 8 张 H20，建议按物体或任务切分成 8 个进程，每个进程绑定一张卡和独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python rollout.py --hand 0 --object_scale_list '[0.08]' --object-root asset/object_mesh --object-names drill --output-root outputs/gpu0 &
CUDA_VISIBLE_DEVICES=1 python rollout.py --hand 0 --object_scale_list '[0.08]' --object-root asset/object_mesh --object-names bowl  --output-root outputs/gpu1 &
wait
```

不要让多个进程写同一个 `output-root` 的同一个物体/scale 目录，否则 `.done`、`.npz`、`.mp4` 可能互相覆盖或误判 resume 状态。

## 10. 常见问题

### 编译时报 CUDA 版本不匹配

先确认：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
nvcc --version
```

若 PyTorch 是 `+cu118`，需要卸载并改装本文的 `cu124`：

```bash
uv pip uninstall torch torchvision torchaudio torch-scatter
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

### 找不到 `coal_openmp_wrapper` 或 `libcoal.so`

重新激活并重装 wrapper：

```bash
source env/activate_uv.sh
echo "$COAL_PREFIX"
echo "$LD_LIBRARY_PATH"

cd third_party/BODex_api/src/bodex/geom/cpp
python setup.py install
cd -
```

### 找不到 `ffmpeg`

```bash
uv pip install imageio-ffmpeg
ln -sf "$(python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)" .venv/bin/ffmpeg
ffmpeg -version
```

### SAPIEN import 成功但渲染失败

检查：

```bash
nvidia-smi
vulkaninfo --summary | head -80
find /usr/share/vulkan/icd.d /etc/vulkan/icd.d -name '*nvidia*.json' -print 2>/dev/null
```

如果没有 NVIDIA ICD JSON，通常需要管理员补齐 NVIDIA Vulkan driver/runtime 组件。

## 参考依据

- `README.md`: 原始通用安装流程。
- `requirements.txt`: 第一方运行依赖，但不含 torch、torch-scatter、ffmpeg。
- `env/activate_uv.sh`: uv 环境激活和 COAL 动态库路径设置。
- `third_party/pytorch3d/INSTALL.md`: 当前 PyTorch3D 安装说明列出的 PyTorch 兼容版本包含 2.4.1。
- `third_party/BODex_api/README.md`: BODex 原安装流程固定在 PyTorch 2.4.1 路线。
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/
- PyG wheel index: https://data.pyg.org/whl/
