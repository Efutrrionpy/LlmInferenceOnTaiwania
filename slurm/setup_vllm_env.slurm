#!/usr/bin/env bash
#SBATCH -J setup-vllm-v100
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_DIR="${PROJECT_DIR}/runs/setup-${SLURM_JOB_ID:-manual}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"

echo "Project dir: ${PROJECT_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Node: $(hostname)"

bash scripts/setup_vllm_v100_env.sh 2>&1 | tee "${RUN_DIR}/setup.log"

source scripts/env_twnia2.sh
python - <<'PY' 2>&1 | tee "${RUN_DIR}/verify.log"
import ray
import torch
import transformers
import vllm

print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
PY
