#!/usr/bin/env bash
#SBATCH -J qwen235moe1n
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B-GPTQ-Int4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-235b-a22b-gptq-1cat-v100-8gpu}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-1cat-qwen3-235b-a22b-sweep-8v100-mp}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-mp}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
export CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32 64}"

exec bash "${PROJECT_DIR}/slurm/vllm_1cat_qwen3_235b_moe_sweep_16v100.slurm"
