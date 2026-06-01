#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/work/${USER}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load miniforge/24.7.1-2 2>/dev/null || module load miniconda3/conda24.5.0_py3.9 2>/dev/null || true
  module load cuda/12.8 2>/dev/null || module load cuda/12.1 2>/dev/null || true
fi

VLLM_ENV="${VLLM_ENV:-/work/${USER}/llm/.venv-vllm-v100}"
if [ ! -x "${VLLM_ENV}/bin/python" ]; then
  echo "Missing vLLM environment: ${VLLM_ENV}" >&2
  echo "Run: bash scripts/setup_vllm_v100_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${VLLM_ENV}"
  elif [ -f "${VLLM_ENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VLLM_ENV}/bin/activate"
  else
    export PATH="${VLLM_ENV}/bin:${PATH}"
  fi
else
  if [ -f "${VLLM_ENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VLLM_ENV}/bin/activate"
  else
    export PATH="${VLLM_ENV}/bin:${PATH}"
  fi
fi
