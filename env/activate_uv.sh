#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}" >&2
  return 1 2>/dev/null || exit 1
fi

source "${VENV_PATH}/bin/activate"

# Prevent external ROS / MoveIt overlays from polluting this project environment.
unset PYTHONPATH

# Keep local console entry points and CUDA tools ahead of any global shims.
export PATH="${VENV_PATH}/bin:${PATH}"

# Pip-installed COAL wheels ship native libraries under cmeel.prefix.
COAL_PREFIX="$(python - <<'PY'
from pathlib import Path
import coal

print(Path(coal.__file__).resolve().parents[4])
PY
)"
export COAL_PREFIX
export LD_LIBRARY_PATH="${COAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "Activated UltraDexGrasp uv environment: ${VENV_PATH}"
