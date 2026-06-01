#!/usr/bin/env bash
#SBATCH -J llama405bgptq16v100
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

# 405B GPTQ INT4 capacity smoke test. The tested cache is about 205G, so keep
# it under /work. This is intentionally tiny and is not a throughput benchmark.
export MODEL_ID="${MODEL_ID:-hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama31-405b-gptq}"
export QUANTIZATION="${QUANTIZATION:-gptq}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-llama31-405b-gptq-smoke}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/hf-cache-llama31-405b-gptq}"

export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

export INPUT_TOKENS="${INPUT_TOKENS:-128}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
export WARMUP="${WARMUP:-0}"
export RUNS="${RUNS:-1}"
export CONCURRENCY="${CONCURRENCY:-1}"

exec bash slurm/vllm_70b_16v100.slurm
