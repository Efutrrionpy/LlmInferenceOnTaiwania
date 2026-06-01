#!/usr/bin/env bash
#SBATCH -J vllm110b16v100
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

# 111B-class smoke test. Keep the cache in /work to avoid filling the
# observed 100G /home quota; this checkpoint is about 61.5GB.
export MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-110B-Chat-GPTQ-Int4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen1.5-110b-gptq-int4}"
export QUANTIZATION="${QUANTIZATION:-gptq}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen110b-smoke}"
export HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/hf-cache-110b}"

export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

export INPUT_TOKENS="${INPUT_TOKENS:-512}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
export WARMUP="${WARMUP:-1}"
export RUNS="${RUNS:-4}"
export CONCURRENCY="${CONCURRENCY:-1}"

exec bash slurm/vllm_70b_16v100.slurm
