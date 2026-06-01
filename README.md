# LLM Inference Batch Optimization on Taiwania 2

This repository contains scripts and experiment notes for LLM inference
throughput experiments on `2 x Taiwania 2 V100 nodes`
(`16 x V100-SXM2-32GB`) with vLLM.

The default target is:

- Model: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`
- Backend: `vllm==0.7.0`
- Parallelism: `tensor_parallel_size=8`, `pipeline_parallel_size=2`
- Precision/quantization: GPTQ 4-bit weights, `float16` activations
- Benchmark: single-request latency, `512` input tokens, `128` output tokens
- Repeats: `3` warmup requests, `20` measured requests

## Why vLLM 0.7.0

Current vLLM releases require NVIDIA compute capability `7.5+`, while V100 is
compute capability `7.0`. vLLM `0.7.0` still documents CUDA GPUs with compute
capability `7.0+`, including V100, so this baseline pins that version.

## Setup

On Taiwania 2, from the project directory:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Before submitting, replace `YOUR_ACCOUNT` in the Slurm files with your project
allocation account.

The setup script loads Taiwania 2's `miniforge` or `miniconda3` module and
creates a Python `3.10` environment at `/work/$USER/llm/.venv-vllm-v100`.

If you use a gated model such as Llama, set `HF_TOKEN` before submitting.
The default Qwen GPTQ-Int4 model does not require a gated license. Keep model
caches under `/work`; the tested Llama 3.1 405B GPTQ cache occupies about 205G.

## Submit

```bash
sbatch slurm/vllm_70b_16v100.slurm
```

For the 104B-class run:

```bash
sbatch slurm/vllm_command_r_plus_16v100.slurm
```

Observed 2-node results for Qwen 72B, Command R+ 104B, and Llama 3.1 405B GPTQ
are documented in `experiments.md` and `batch_inference_report.md`.

For a cached 405B-scale smoke test:

```bash
sbatch slurm/vllm_llama31_405b_gptq_16v100.slurm
```

Useful overrides:

```bash
MODEL_ID=Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
QUANTIZATION=gptq \
TP_SIZE=8 \
PP_SIZE=2 \
RUNS=10 \
WARMUP=3 \
CONCURRENCY=1 \
INPUT_TOKENS=512 \
OUTPUT_TOKENS=128 \
sbatch slurm/vllm_70b_16v100.slurm
```

HPC-oriented overrides:

```bash
EXPERIMENT_NAME=topo-tp16-pp1-nccl \
TP_SIZE=16 \
PP_SIZE=1 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET \
sbatch slurm/vllm_70b_16v100.slurm
```

For serving throughput experiments, increase both the benchmark concurrency and
vLLM scheduler limits:

```bash
EXPERIMENT_NAME=batch-c16 \
TP_SIZE=8 \
PP_SIZE=2 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
CONCURRENCY=16 \
RUNS=32 \
sbatch slurm/vllm_70b_16v100.slurm
```

## Results

Each run writes to:

```text
runs/<slurm-job-id>/
```

Important files:

- `summary.json`: mean, p50, p95 latency and token/s numbers
- `requests.jsonl`: one measured request per line
- `experiment.env`: Slurm, model, parallelism, NCCL, and benchmark settings
- `vllm-server.log`: vLLM startup and runtime log
- `ray-head.log`, `ray-worker-*.log`: Ray cluster logs

The project-level experiment table is in `experiments.md`.
