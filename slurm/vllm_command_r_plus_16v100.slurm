#!/usr/bin/env bash
#SBATCH -J commandrplus16v100
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

# 100B-class dense-model runner. This public GPTQ checkpoint is about
# 58.7GB and has been smoke-tested on Taiwania 2 with vLLM 0.7.0.
export MODEL_ID="${MODEL_ID:-alpindale/c4ai-command-r-plus-GPTQ}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-command-r-plus-gptq}"
export QUANTIZATION="${QUANTIZATION:-gptq}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-command-r-plus-smoke}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/hf-cache-command-r-plus-gptq}"

export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

export INPUT_TOKENS="${INPUT_TOKENS:-512}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
export WARMUP="${WARMUP:-2}"
export RUNS="${RUNS:-8}"
export CONCURRENCY="${CONCURRENCY:-1}"

exec bash slurm/vllm_70b_16v100.slurm
