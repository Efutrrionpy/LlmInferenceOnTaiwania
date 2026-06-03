#!/usr/bin/env bash
#SBATCH -J qwen397moe
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-397b-a17b-gptq-tp16-v100-16gpu}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-1cat-qwen35-397b-a17b-sweep-16v100-ray-tp16-${ATTENTION_BACKEND:-FLASH_ATTN_V100}}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/hf-cache-qwen35-397b-a17b-gptq}"
export TP_SIZE="${TP_SIZE:-16}"
export PP_SIZE="${PP_SIZE:-1}"
export EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-ray}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32}"
export RUNS_PER_CONCURRENCY_FACTOR="${RUNS_PER_CONCURRENCY_FACTOR:-2}"
export MIN_RUNS="${MIN_RUNS:-4}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN_V100}"
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---disable-custom-all-reduce}"
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-2400}"

exec bash "${PROJECT_DIR}/slurm/vllm_1cat_qwen3_235b_moe_sweep_16v100.slurm"
