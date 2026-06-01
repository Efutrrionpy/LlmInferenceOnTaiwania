#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ENV="${VLLM_ENV:-/work/${USER}/llm/.venv-vllm-v100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_SETUP:-0}" != "1" ]; then
  echo "Refusing to install on a login node." >&2
  echo "Submit slurm/setup_vllm_env.slurm, or set ALLOW_LOGIN_SETUP=1 only if you really mean it." >&2
  exit 1
fi

mkdir -p "$(dirname "${VLLM_ENV}")"

env_is_usable() {
  [ -x "${VLLM_ENV}/bin/python" ] &&
    "${VLLM_ENV}/bin/python" -c "import sys; raise SystemExit(sys.version_info < (3, 9))" >/dev/null 2>&1
}

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load miniforge/24.7.1-2 2>/dev/null || module load miniconda3/conda24.5.0_py3.9 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if [ -d "${VLLM_ENV}" ] && ! env_is_usable; then
    mv "${VLLM_ENV}" "${VLLM_ENV}.broken-$(date +%Y%m%d%H%M%S)"
  fi
  if ! env_is_usable; then
    conda create -y -p "${VLLM_ENV}" "python=${PYTHON_VERSION}"
  fi
  conda activate "${VLLM_ENV}"
else
  PYTHON_BIN=""
  for candidate in python3.12 python3.11 python3.10 python3.9; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
  if [ -z "${PYTHON_BIN}" ]; then
    echo "Could not find conda or Python 3.9+." >&2
    echo "On Taiwania 2, try: module load miniforge/24.7.1-2" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${VLLM_ENV}"
  # shellcheck disable=SC1090
  source "${VLLM_ENV}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements-vllm-v100.txt"

python - <<'PY'
import ray
import torch
import transformers
import vllm

print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
