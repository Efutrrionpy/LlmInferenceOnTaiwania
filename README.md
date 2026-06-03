# Llama 405B Inference Throughput on Taiwania 2

This project evaluates throughput-oriented inference optimization for `Llama 3.1 405B GPTQ` on Taiwania 2 under a fixed HPC allocation: 2 nodes, 16 NVIDIA V100 GPUs, and one hour per Slurm job.

The main result is that the workload improves from `7.27 tok/s` for a single stock vLLM request to `351.62 tok/s` with continuous batching, a V100-compatible FlashAttention backend, and NCCL InfiniBand/GDRDMA transport. That is a `48.37x` improvement on the same 16-GPU allocation.

## Research Question

How much can large-model inference throughput improve on older V100 HPC nodes when the optimization is treated as a systems problem?

The study focuses on four levers:

- Continuous batching with vLLM.
- A V100-compatible FlashAttention backend.
- NCCL transport for cross-node tensor-parallel communication.
- GPU memory headroom for stable high-concurrency serving.

The goal is not to compare many models. The main workload is one large dense model, measured repeatedly under different serving and communication settings.

## Testbed

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x NVIDIA V100-SXM2-32GB |
| Job limit | One hour per Slurm job |
| Main model | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` |
| Quantization | GPTQ INT4 |
| Frameworks | stock vLLM `0.7.0`; V100 fork vLLM `1.1.0` |
| Attention backends | XFormers / stock; `FLASH_ATTN_V100` |
| Input / output | 497 input tokens, 128 output tokens |
| Max model length | `1024` |
| Main metric | aggregate generated output tokens per second |

`Concurrency` means the number of in-flight requests served at the same time. `Aggregate tok/s` is total generated output tokens divided by measured benchmark wall time.

## Main Result

The optimization path is:

| Step | Configuration | Parallelism | Concurrency | Aggregate tok/s | Improvement vs baseline |
| --- | --- | --- | ---: | ---: | ---: |
| Stock single request | stock vLLM / XFormers | `TP=8`, `PP=2` | 1 | 7.27 | 1.00x |
| Continuous batching | stock vLLM / XFormers | `TP=8`, `PP=2` | 128 | 120.65 | 16.60x |
| V100 FlashAttention | `FLASH_ATTN_V100` | `TP=8`, `PP=2` | 128 | 209.50 | 28.82x |
| NCCL IB/GDRDMA | `FLASH_ATTN_V100` + NCCL `NET/IB` | `TP=16`, `PP=1` | 128 | 351.62 | 48.37x |

The final result is not just a larger batch size. The largest HPC-specific gain came from exposing the correct NCCL InfiniBand transport inside the container. Without that, the same cross-node tensor-parallel layout was slower than the `TP=8`, `PP=2` layout.

## Batching And Attention

This table shows aggregate output-token throughput as concurrency increases.

| Backend | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XFormers / stock | 7.27 | 13.25 | 24.65 | 42.77 | 64.36 | 79.91 | 103.75 | 120.65 | 120.65 |
| `FLASH_ATTN_V100` | 7.12 | 13.70 | 25.66 | 43.39 | 57.42 | 83.73 | 126.67 | 209.50 | 209.50 |

At `c=128`, `FLASH_ATTN_V100` is `73.6%` faster than stock vLLM (`209.50` vs `120.65 tok/s`). The high-concurrency runs required `GPU_MEMORY_UTILIZATION=0.88`; higher memory utilization left too little room for initialization and KV cache.

For the FlashAttention run, decode partition size `512` was selected because it was the best measured setting. Partition size `256` was nearly identical, while `1024` was slower.

## NCCL Transport

The most important HPC result is that the interconnect transport changes which parallelism layout is best.

| Parallelism | NCCL transport | Mode | Aggregate tok/s | Mean latency | Mean TTFT | Mean decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `TP=8`, `PP=2` | default | CUDA graph | 209.50 | 78.09s | 0.80s | 1.64 |
| `TP=8`, `PP=2` | `NET/IB` + `GDRDMA` | CUDA graph | 204.70 | 80.00s | 0.75s | 1.60 |
| `TP=16`, `PP=1` | Socket | eager | 156.12 | 104.75s | 1.00s | 1.22 |
| `TP=16`, `PP=1` | `NET/IB` + `GDRDMA` | eager | 351.62 | 46.48s | 0.56s | 2.77 |

The interpretation is:

- `TP=8`, `PP=2` keeps tensor-parallel collectives inside each 8-GPU node. Adding IB/GDRDMA to this layout does not improve throughput (`209.50` to `204.70 tok/s`).
- `TP=16`, `PP=1` spreads tensor-parallel collectives across both nodes. With Socket transport it reaches only `156.12 tok/s`.
- After staging the minimal RDMA userspace libraries into the container, NCCL uses `NET/IB` with `GDRDMA`, and the same `TP=16`, `PP=1` layout reaches `351.62 tok/s`.

This makes NCCL transport the main HPC-side optimization. The job was already using an IB network interface, but NCCL was not using the InfiniBand transport until the container could see the required RDMA userspace libraries.

## Takeaways

Continuous batching is the first-order serving optimization. Stock vLLM improves from `7.27` to `120.65 tok/s` when concurrency increases from `1` to `128`.

The V100 FlashAttention backend matters at high concurrency. At `c=128`, it raises throughput from `120.65` to `209.50 tok/s` and greatly reduces TTFT.

NCCL transport matters only when the parallelism layout generates heavy cross-node tensor-parallel communication. This is why `TP=16`, `PP=1` is poor with Socket transport but becomes the best result with NCCL `NET/IB` and GDRDMA.

The result targets offline or batched serving throughput. Higher concurrency improves aggregate tok/s but increases per-request latency.

## Boundary Checks

A 1-node / 8 x V100 run was tested as a capacity boundary for `Llama 3.1 405B GPTQ`. With `TP=8`, `PP=1`, and `max_model_len=1024`, vLLM reported that the available KV cache could store only `480` tokens, so the main workload requires the 2-node setup.

Two MoE models were also tested as feasibility checks. They are not part of the main comparison because sparse MoE models have fewer active parameters per generated token.

| Model | Allocation | Parallelism | Backend | Peak aggregate tok/s |
| --- | --- | --- | --- | ---: |
| `Qwen3 235B-A22B MoE GPTQ` | 1 node / 8 x V100 | `TP=8`, `PP=1` | `FLASH_ATTN_V100` | 416.17 @ c=64 |
| `Qwen3.5 397B-A17B MoE GPTQ` | 2 nodes / 16 x V100 | `TP=16`, `PP=1` | `FLASH_ATTN_V100` | 74.16 @ c=64 |

## Reproduce

Do not install packages or run inference on the login node. Create the vLLM environment through Slurm on a compute node:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Run the V100 FlashAttention `c=128` baseline:

```bash
EXPERIMENT_NAME=1cat-llama31-405b-c128-flash-v100 \
GPU_MEMORY_UTILIZATION=0.88 \
MAX_NUM_SEQS=128 \
CONCURRENCY_LEVELS=128 \
RUNS_PER_CONCURRENCY_FACTOR=2 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE=512 \
sbatch slurm/vllm_1cat_llama31_405b_sweep_16v100.slurm
```

Run the NCCL IB/GDRDMA comparison after staging the host RDMA userspace libraries into a small directory during a Slurm job:

```bash
EXPERIMENT_NAME=1cat-llama31-405b-c128-tp16-pp1-eager-ib-p512 \
TP_SIZE=16 \
PP_SIZE=1 \
GPU_MEMORY_UTILIZATION=0.88 \
MAX_NUM_SEQS=128 \
CONCURRENCY_LEVELS=128 \
RUNS_PER_CONCURRENCY_FACTOR=2 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE=512 \
VLLM_FLASH_V100_ENABLE_PAGED_PREFILL=1 \
EXTRA_VLLM_ARGS=--enforce-eager \
NCCL_IB_DISABLE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET \
APPTAINERENV_LD_LIBRARY_PATH=/path/to/rdma-libs:${LD_LIBRARY_PATH:-} \
APPTAINERENV_LIBIBVERBS_DRIVER_DIR=/path/to/rdma-libs \
LIBIBVERBS_DRIVER_DIR=/path/to/rdma-libs \
sbatch slurm/vllm_1cat_llama31_405b_sweep_16v100.slurm
```

The expected NCCL evidence in `vllm-server.log` is `Using network IB` plus channels marked `via NET/IB/.../GDRDMA`.

The useful output files are:

```text
runs/<run-id>/summary.json
runs/<run-id>/experiment.env
runs/<run-id>/vllm-server.log
```
