#!/usr/bin/env bash
#SBATCH -J qwen235moe
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
cd "${PROJECT_DIR}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B-GPTQ-Int4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-235b-a22b-gptq-1cat-v100}"
QUANTIZATION="${QUANTIZATION:-gptq}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1cat-qwen3-235b-a22b-sweep}"

TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-2}"
EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-ray}"
RAY_PORT="${RAY_PORT:-6379}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32 64}"
RUNS_PER_CONCURRENCY_FACTOR="${RUNS_PER_CONCURRENCY_FACTOR:-4}"
MIN_RUNS="${MIN_RUNS:-8}"
INPUT_TOKENS="${INPUT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARMUP="${WARMUP:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN_V100}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-2400}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-/work/${USER}/hf-cache-qwen3-235b-a22b-gptq}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/work/${USER}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/work/${USER}/apptainer-tmp}"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "/work/${USER}/triton-cache-1cat"

IMAGE="${IMAGE:-docker://nvidia/cuda:12.8.1-devel-ubuntu24.04}"
PYBIN="${PYBIN:-${PROJECT_DIR}/.venv-1cat-vllm-sm70/bin/python}"
VENV_BIN="$(dirname "${PYBIN}")"
RUN_DIR="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}-${SLURM_JOB_ID:-manual}"
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
echo "Model: ${MODEL_ID}"
echo "Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}"
echo "Executor backend: ${EXECUTOR_BACKEND}"
echo "Concurrency levels: ${CONCURRENCY_LEVELS}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "Max num seqs: ${MAX_NUM_SEQS}"
echo "Max batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "Attention backend: ${ATTENTION_BACKEND}"
echo "Server start timeout: ${SERVER_START_TIMEOUT}s"
echo "Image: ${IMAGE}"
echo "Python: ${PYBIN}"
echo "HF_HOME: ${HF_HOME}"

cat >"${RUN_DIR}/experiment.env" <<EOF
EXPERIMENT_NAME=${EXPERIMENT_NAME}
MODEL_ID=${MODEL_ID}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
QUANTIZATION=${QUANTIZATION}
TP_SIZE=${TP_SIZE}
PP_SIZE=${PP_SIZE}
EXECUTOR_BACKEND=${EXECUTOR_BACKEND}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}
CONCURRENCY_LEVELS=${CONCURRENCY_LEVELS}
INPUT_TOKENS=${INPUT_TOKENS}
OUTPUT_TOKENS=${OUTPUT_TOKENS}
WARMUP=${WARMUP}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS}
HF_HOME=${HF_HOME}
NODE_COUNT=${NODE_COUNT}
GPUS_PER_NODE=${GPUS_PER_NODE}
IMAGE=${IMAGE}
VLLM_ATTENTION_BACKEND=${ATTENTION_BACKEND}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT}
VLLM_USE_V1=1
EOF

run_in_container() {
  apptainer exec --nv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --bind "/work/${USER}:/work/${USER}" \
    "${IMAGE}" \
    /bin/bash -lc "$1"
}

write_runtime_scripts() {
  cat >"${RUN_DIR}/ray-head.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
export PATH="${VENV_BIN}:\${PATH}"
export HOME="/work/${USER}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}"
export VLLM_HOST_IP="${HEAD_IP}"
ray stop -f >/dev/null 2>&1 || true
exec ray start --head \\
  --node-ip-address="${HEAD_IP}" \\
  --port="${RAY_PORT}" \\
  --num-cpus="${SLURM_CPUS_PER_TASK}" \\
  --num-gpus="${GPUS_PER_NODE}" \\
  --block
EOF

  cat >"${RUN_DIR}/ray-worker.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
export PATH="${VENV_BIN}:\${PATH}"
export HOME="/work/${USER}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}"
WORKER_IP="\$(hostname -I | awk '{print \$1}')"
export VLLM_HOST_IP="\${WORKER_IP}"
ray stop -f >/dev/null 2>&1 || true
exec ray start --address="${RAY_ADDRESS}" \\
  --node-ip-address="\${WORKER_IP}" \\
  --num-cpus="${SLURM_CPUS_PER_TASK}" \\
  --num-gpus="${GPUS_PER_NODE}" \\
  --block
EOF

  local quantization_args=""
  local max_num_batched_tokens_args=""
  if [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "auto" ] && [ "${QUANTIZATION}" != "none" ]; then
    quantization_args="--quantization '${QUANTIZATION}'"
  fi
  if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
    max_num_batched_tokens_args="--max-num-batched-tokens '${MAX_NUM_BATCHED_TOKENS}'"
  fi

  cat >"${RUN_DIR}/vllm-server.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
export PATH="${VENV_BIN}:\${PATH}"
export HOME="/work/${USER}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}"
export TRITON_CACHE_DIR="/work/${USER}/triton-cache-1cat"
export CC="\${CC:-gcc}"
export CXX="\${CXX:-g++}"
export RAY_ADDRESS="${RAY_ADDRESS}"
export VLLM_HOST_IP="${HEAD_IP}"
export VLLM_USE_V1="1"
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
export VLLM_LOGGING_LEVEL="INFO"
exec "${PYBIN}" -m vllm.entrypoints.openai.api_server \\
  --host 0.0.0.0 \\
  --port "${VLLM_PORT}" \\
  --model "${MODEL_ID}" \\
  --served-model-name "${SERVED_MODEL_NAME}" \\
  --tensor-parallel-size "${TP_SIZE}" \\
  --pipeline-parallel-size "${PP_SIZE}" \\
  --distributed-executor-backend "${EXECUTOR_BACKEND}" \\
  --dtype float16 \\
  ${quantization_args} \\
  --max-model-len "${MAX_MODEL_LEN}" \\
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \\
  --max-num-seqs "${MAX_NUM_SEQS}" \\
  ${max_num_batched_tokens_args} \\
  --trust-remote-code \\
  --disable-log-requests \\
  ${EXTRA_VLLM_ARGS}
EOF

  chmod +x "${RUN_DIR}/ray-head.sh" "${RUN_DIR}/ray-worker.sh" "${RUN_DIR}/vllm-server.sh"
}

RAY_PIDS=()
SERVER_PID=""

cleanup() {
  set +e
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  for pid in "${RAY_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  run_in_container "export PATH='${VENV_BIN}:\${PATH}'; ray stop -f" >/dev/null 2>&1 || true
  if [ -n "${WORKER_NODE}" ]; then
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 -w "${WORKER_NODE}" \
      bash -lc "apptainer exec --nv --bind '${PROJECT_DIR}:${PROJECT_DIR}' --bind '/work/${USER}:/work/${USER}' '${IMAGE}' /bin/bash -lc \"export PATH='${VENV_BIN}':\\\${PATH}; ray stop -f\"" \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_ray_head() {
  run_in_container "bash '${RUN_DIR}/ray-head.sh'" >"${RUN_DIR}/ray-head.log" 2>&1 &
  RAY_PIDS+=("$!")
}

start_ray_worker() {
  if [ -z "${WORKER_NODE}" ]; then
    return 0
  fi

  srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" -w "${WORKER_NODE}" \
    bash -lc "apptainer exec --nv --bind '${PROJECT_DIR}:${PROJECT_DIR}' --bind '/work/${USER}:/work/${USER}' '${IMAGE}' /bin/bash '${RUN_DIR}/ray-worker.sh'" \
    >"${RUN_DIR}/ray-worker-${WORKER_NODE}.log" 2>&1 &
  RAY_PIDS+=("$!")
}

wait_for_ray() {
  local active_nodes
  for _ in $(seq 1 90); do
    active_nodes="$(run_in_container "'${PYBIN}' -c \"import ray; ray.init(address='${RAY_ADDRESS}', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive')))\"" 2>/dev/null | tail -n 1 || true)"
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
  run_in_container "bash '${RUN_DIR}/vllm-server.sh'" >"${RUN_DIR}/vllm-server.log" 2>&1 &
  SERVER_PID="$!"
}

wait_for_server_or_die() {
  local attempts=$((SERVER_START_TIMEOUT / 5))
  if [ "${attempts}" -lt 1 ]; then
    attempts=1
  fi
  for _ in $(seq 1 "${attempts}"); do
    if [ -n "${SERVER_PID}" ] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM server exited before becoming ready" >&2
      tail -n 120 "${RUN_DIR}/vllm-server.log" >&2 || true
      return 1
    fi
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${VLLM_PORT}/v1/models', timeout=3)" >/dev/null 2>&1; then
      echo "vLLM server is ready"
      return 0
    fi
    sleep 5
  done
  echo "vLLM server did not become ready" >&2
  return 1
}

run_benchmark_level() {
  local concurrency="$1"
  local runs=$((concurrency * RUNS_PER_CONCURRENCY_FACTOR))
  if [ "${runs}" -lt "${MIN_RUNS}" ]; then
    runs="${MIN_RUNS}"
  fi

  local result_dir="${RUN_DIR}/c${concurrency}"
  mkdir -p "${result_dir}"
  cp "${RUN_DIR}/experiment.env" "${result_dir}/experiment.env"
  {
    echo "CONCURRENCY=${concurrency}"
    echo "RUNS=${runs}"
  } >>"${result_dir}/experiment.env"

  echo "Running benchmark: concurrency=${concurrency}, runs=${runs}"
  set +e
  run_in_container "cd '${PROJECT_DIR}' && '${PYBIN}' scripts/benchmark_openai_completion.py \
    --base-url 'http://127.0.0.1:${VLLM_PORT}/v1' \
    --model '${SERVED_MODEL_NAME}' \
    --tokenizer '${MODEL_ID}' \
    --input-tokens '${INPUT_TOKENS}' \
    --output-tokens '${OUTPUT_TOKENS}' \
    --warmup '${WARMUP}' \
    --runs '${runs}' \
    --concurrency '${concurrency}' \
    --results-dir '${result_dir}' \
    --server-timeout 60 \
    --request-timeout 1200" >"${result_dir}/benchmark.log" 2>&1
  local status="$?"
  set -e
  cat "${result_dir}/benchmark.log"
  echo "${status}" >"${result_dir}/exit_code"
  return 0
}

write_sweep_summary() {
  python - "${RUN_DIR}" ${CONCURRENCY_LEVELS} <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
levels = sys.argv[2:]
rows = []
for level in levels:
    cdir = run_dir / f"c{level}"
    summary_path = cdir / "summary.json"
    if not summary_path.exists():
        rows.append({"concurrency": int(level), "status": "missing summary"})
        continue
    summary = json.loads(summary_path.read_text())
    rows.append({
        "concurrency": int(level),
        "requests": summary.get("requests"),
        "aggregate_output_tokens_per_s": summary.get("aggregate_output_tokens_per_s"),
        "decode_tokens_per_s": summary.get("decode_tokens_per_s", {}).get("mean"),
        "latency_s_mean": summary.get("latency_s", {}).get("mean"),
        "ttft_s_mean": summary.get("ttft_s", {}).get("mean"),
    })
(run_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("| concurrency | requests | aggregate tok/s | decode tok/s | latency mean s | TTFT mean s |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
for row in rows:
    if "status" in row:
        print(f"| {row['concurrency']} |  | {row['status']} |  |  |  |")
    else:
        print(
            f"| {row['concurrency']} | {row['requests']} | "
            f"{row['aggregate_output_tokens_per_s']:.2f} | "
            f"{row['decode_tokens_per_s']:.2f} | "
            f"{row['latency_s_mean']:.3f} | "
            f"{row['ttft_s_mean']:.3f} |"
        )
PY
}

write_runtime_scripts
if [ "${EXECUTOR_BACKEND}" = "ray" ]; then
  start_ray_head
  sleep 10
  start_ray_worker
  wait_for_ray
fi
start_vllm_server
wait_for_server_or_die

for concurrency in ${CONCURRENCY_LEVELS}; do
  run_benchmark_level "${concurrency}"
done

grep -E "FLASH_ATTN_V100|attention backend|fallback|path active|CUDA|GPU|quant|KV cache" "${RUN_DIR}/vllm-server.log" >"${RUN_DIR}/backend-markers.log" 2>/dev/null || true
write_sweep_summary | tee "${RUN_DIR}/sweep_summary.md"

echo "Qwen3 MoE sweep finished. Summary: ${RUN_DIR}/sweep_summary.json"
