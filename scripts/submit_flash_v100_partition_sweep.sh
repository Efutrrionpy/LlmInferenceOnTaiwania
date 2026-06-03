#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available in this shell. Run this helper on the HPC login node." >&2
  exit 1
fi

PARTITIONS="${PARTITIONS:-256 512 1024}"

for partition_size in ${PARTITIONS}; do
  echo "Submitting FLASH_ATTN_V100 partition_size=${partition_size}"
  EXPERIMENT_NAME="1cat-llama31-405b-c128-p${partition_size}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}" \
  CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-128}" \
  RUNS_PER_CONCURRENCY_FACTOR="${RUNS_PER_CONCURRENCY_FACTOR:-2}" \
  VLLM_FLASH_V100_DECODE_PARTITION_SIZE="${partition_size}" \
  VLLM_FLASH_V100_ENABLE_PAGED_PREFILL="${VLLM_FLASH_V100_ENABLE_PAGED_PREFILL:-1}" \
  sbatch slurm/vllm_1cat_llama31_405b_sweep_16v100.slurm
done
