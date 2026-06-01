#!/usr/bin/env bash
#SBATCH -J vllm70b16v100
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
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-baseline-70b-gptq-int4}"
QUANTIZATION="${QUANTIZATION:-gptq}"

TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-2}"
RAY_PORT="${RAY_PORT:-6379}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-ray}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/llm-hf-cache}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-20}"
CONCURRENCY="${CONCURRENCY:-1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-manual}"

RUN_DIR="${PROJECT_DIR}/runs/${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"

readarray -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
NODE_COUNT="${#NODES[@]}"
CURRENT_HOST="$(hostname)"
CURRENT_SHORT="$(hostname -s)"
HEAD_NODE="${NODES[0]}"
for node in "${NODES[@]}"; do
  if [ "${node}" = "${CURRENT_HOST}" ] || [ "${node}" = "${CURRENT_SHORT}" ] || [[ "${CURRENT_HOST}" = "${node}".* ]]; then
    HEAD_NODE="${node}"
    break
  fi
done

WORKER_NODE=""
for node in "${NODES[@]}"; do
  if [ "${node}" != "${HEAD_NODE}" ]; then
    WORKER_NODE="${node}"
    break
  fi
done

HEAD_IP="$(hostname -I | awk '{print $1}')"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Project dir: ${PROJECT_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Node count: ${NODE_COUNT}"
echo "Head node: ${HEAD_NODE} (${HEAD_IP})"
echo "Worker node: ${WORKER_NODE:-none}"
echo "Current host: ${CURRENT_HOST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Model: ${MODEL_ID}"
echo "Served name: ${SERVED_MODEL_NAME}"
echo "Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}"
echo "Quantization: ${QUANTIZATION:-auto}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max num seqs: ${MAX_NUM_SEQS}"
echo "Max batched tokens: ${MAX_NUM_BATCHED_TOKENS:-auto}"
echo "Distributed executor backend: ${DISTRIBUTED_EXECUTOR_BACKEND}"
echo "Benchmark: warmup=${WARMUP}, runs=${RUNS}, concurrency=${CONCURRENCY}, input=${INPUT_TOKENS}, output=${OUTPUT_TOKENS}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Attention backend override: ${VLLM_ATTENTION_BACKEND:-auto}"
echo "NCCL_DEBUG: ${NCCL_DEBUG:-unset}"
echo "NCCL_DEBUG_SUBSYS: ${NCCL_DEBUG_SUBSYS:-unset}"
echo "NCCL_ALGO: ${NCCL_ALGO:-unset}"
echo "NCCL_PROTO: ${NCCL_PROTO:-unset}"
echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-unset}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-unset}"
echo "NCCL_NET_GDR_LEVEL: ${NCCL_NET_GDR_LEVEL:-unset}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-unset}"
echo "HF_HOME: ${HF_HOME}"

cat >"${RUN_DIR}/experiment.env" <<EOF
SLURM_JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_NAME=${EXPERIMENT_NAME}
MODEL_ID=${MODEL_ID}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
QUANTIZATION=${QUANTIZATION}
TP_SIZE=${TP_SIZE}
PP_SIZE=${PP_SIZE}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND}
INPUT_TOKENS=${INPUT_TOKENS}
OUTPUT_TOKENS=${OUTPUT_TOKENS}
WARMUP=${WARMUP}
RUNS=${RUNS}
CONCURRENCY=${CONCURRENCY}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}
NCCL_DEBUG=${NCCL_DEBUG:-}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}
NCCL_ALGO=${NCCL_ALGO:-}
NCCL_PROTO=${NCCL_PROTO:-}
HF_HOME=${HF_HOME}
NODE_COUNT=${NODE_COUNT}
GPUS_PER_NODE=${GPUS_PER_NODE}
HEAD_NODE=${HEAD_NODE}
HEAD_IP=${HEAD_IP}
WORKER_NODE=${WORKER_NODE}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-}
NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
EOF

RAY_PIDS=()
SERVER_PID=""

cleanup() {
  set +e
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
  for pid in "${RAY_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  cd "${PROJECT_DIR}" && source scripts/env_twnia2.sh && ray stop -f >/dev/null 2>&1 || true
  if [ -n "${WORKER_NODE}" ]; then
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 -w "${WORKER_NODE}" \
      bash -lc "cd '${PROJECT_DIR}' && source scripts/env_twnia2.sh && ray stop -f" \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_ray_head() {
  bash -lc "
    cd '${PROJECT_DIR}'
    source scripts/env_twnia2.sh
    export VLLM_HOST_IP='${HEAD_IP}'
    ray stop -f >/dev/null 2>&1 || true
    ray start --head \
      --node-ip-address='${HEAD_IP}' \
      --port='${RAY_PORT}' \
      --num-cpus='${SLURM_CPUS_PER_TASK}' \
      --num-gpus='${GPUS_PER_NODE}' \
      --block
  " >"${RUN_DIR}/ray-head.log" 2>&1 &
  RAY_PIDS+=("$!")
}

start_ray_worker() {
  srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" -w "${WORKER_NODE}" bash -lc "
    cd '${PROJECT_DIR}'
    source scripts/env_twnia2.sh
    WORKER_IP=\"\$(hostname -I | awk '{print \$1}')\"
    export VLLM_HOST_IP=\"\${WORKER_IP}\"
    ray stop -f >/dev/null 2>&1 || true
    ray start --address='${RAY_ADDRESS}' \
      --node-ip-address=\"\${WORKER_IP}\" \
      --num-cpus='${SLURM_CPUS_PER_TASK}' \
      --num-gpus='${GPUS_PER_NODE}' \
      --block
  " >"${RUN_DIR}/ray-worker-${WORKER_NODE}.log" 2>&1 &
  RAY_PIDS+=("$!")
}

wait_for_ray() {
  local active_nodes
  for _ in $(seq 1 60); do
    active_nodes="$(
      cd "${PROJECT_DIR}"
      source scripts/env_twnia2.sh
      python -c "import ray; ray.init(address='${RAY_ADDRESS}', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive')))" 2>/dev/null | tail -n 1
    )"
    if [ "${active_nodes}" = "${NODE_COUNT}" ]; then
      echo "Ray cluster is ready: ${active_nodes}/${NODE_COUNT} nodes"
      return 0
    fi
    echo "Waiting for Ray cluster: ${active_nodes:-0}/${NODE_COUNT} nodes"
    sleep 5
  done
  echo "Ray cluster did not become ready" >&2
  return 1
}

start_vllm_server() {
  local quantization_args=""
  local max_num_batched_tokens_args=""
  if [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "auto" ] && [ "${QUANTIZATION}" != "none" ]; then
    quantization_args="--quantization '${QUANTIZATION}'"
  fi
  if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
    max_num_batched_tokens_args="--max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}'"
  fi

  bash -lc "
    cd '${PROJECT_DIR}'
    source scripts/env_twnia2.sh
    export RAY_ADDRESS='${RAY_ADDRESS}'
    export VLLM_HOST_IP='${HEAD_IP}'
    export HF_HUB_DISABLE_XET='${HF_HUB_DISABLE_XET}'
    python -m vllm.entrypoints.openai.api_server \
      --host 0.0.0.0 \
      --port '${VLLM_PORT}' \
      --model '${MODEL_ID}' \
      --served-model-name '${SERVED_MODEL_NAME}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --pipeline-parallel-size '${PP_SIZE}' \
      --distributed-executor-backend '${DISTRIBUTED_EXECUTOR_BACKEND}' \
      --dtype float16 \
      ${quantization_args} \
      --max-model-len '${MAX_MODEL_LEN}' \
      --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      ${max_num_batched_tokens_args} \
      --trust-remote-code \
      --disable-log-requests \
      ${EXTRA_VLLM_ARGS}
  " >"${RUN_DIR}/vllm-server.log" 2>&1 &
  SERVER_PID="$!"
}

run_benchmark() {
  bash -lc "
    cd '${PROJECT_DIR}'
    source scripts/env_twnia2.sh
    python scripts/benchmark_openai_completion.py \
      --base-url 'http://127.0.0.1:${VLLM_PORT}/v1' \
      --model '${SERVED_MODEL_NAME}' \
      --tokenizer '${MODEL_ID}' \
      --input-tokens '${INPUT_TOKENS}' \
      --output-tokens '${OUTPUT_TOKENS}' \
      --warmup '${WARMUP}' \
      --runs '${RUNS}' \
      --concurrency '${CONCURRENCY}' \
      --results-dir '${RUN_DIR}' \
      --server-timeout 2400 \
      --request-timeout 900
  " | tee "${RUN_DIR}/benchmark.log"
}

if [ "${DISTRIBUTED_EXECUTOR_BACKEND}" = "ray" ]; then
  start_ray_head
  sleep 10
  if [ -n "${WORKER_NODE}" ]; then
    start_ray_worker
  fi
  wait_for_ray
else
  if [ "${NODE_COUNT}" != "1" ]; then
    echo "DISTRIBUTED_EXECUTOR_BACKEND=${DISTRIBUTED_EXECUTOR_BACKEND} is only supported by this script for single-node jobs" >&2
    exit 1
  fi
  echo "Skipping Ray startup for distributed executor backend: ${DISTRIBUTED_EXECUTOR_BACKEND}"
fi
start_vllm_server
run_benchmark

echo "Baseline finished. Summary: ${RUN_DIR}/summary.json"
