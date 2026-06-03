# Llama 405B Inference Throughput on Taiwania 2

This project studies how far `Llama 3.1 405B GPTQ` inference throughput can be pushed on Taiwania 2 using a fixed HPC allocation: 2 nodes, 16 NVIDIA V100 GPUs, and one hour per Slurm job.

The main result is that continuous batching, a V100-specific FlashAttention backend, and true NCCL InfiniBand/GDRDMA transport improve aggregate output-token throughput from `7.27 tok/s` for a single stock request to `351.62 tok/s` at concurrency `128`.

## Research Question

How much can large-model inference throughput improve on older V100 HPC nodes by combining:

- vLLM continuous batching
- a V100-compatible FlashAttention backend
- NCCL transport tuning for cross-node tensor parallelism
- GPU memory headroom tuning for high concurrency

The focus is throughput for one large dense model, not a broad model leaderboard.

## Key Findings

- `Llama 3.1 405B GPTQ` runs on 2 nodes / 16 x V100 with `TP=8`, `PP=2`.
- Stock vLLM uses XFormers on this V100 stack.
- Stock vLLM improves from `7.27 tok/s` at `c=1` to `120.65 tok/s` at `c=128`.
- The V100 FlashAttention fork reaches `209.50 tok/s` at `c=128`.
- The best result is `351.62 tok/s` with `TP=16`, `PP=1`, and NCCL `NET/IB` with GDRDMA.
- The best result is `48.37x` higher than the single-request stock baseline.
- At `c=128`, `FLASH_ATTN_V100` is `73.6%` faster than stock vLLM (`209.50` vs `120.65 tok/s`).
- The default container path used NCCL Socket transport for the cross-node `TP=16`, `PP=1` run and reached only `156.12 tok/s`.
- Staging the minimal RDMA userspace libraries into the container enabled NCCL `NET/IB` and `GDRDMA`, raising the same `TP=16`, `PP=1` case to `351.62 tok/s`.
- The `c=128` runs require `GPU_MEMORY_UTILIZATION=0.88`; higher memory utilization left too little headroom for initialization and KV cache.

## System And Workload

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x NVIDIA V100-SXM2-32GB |
| Job limit | One hour per Slurm job |
| Model | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` |
| Quantization | GPTQ INT4 |
| Parallelism | Best result: `TP=16`, `PP=1`; batching comparison: `TP=8`, `PP=2` |
| Frameworks | stock vLLM `0.7.0`; V100 fork vLLM `1.1.0` |
| Attention backends | XFormers / stock; `FLASH_ATTN_V100` |
| Interconnect path | NCCL Socket; NCCL `NET/IB` with GDRDMA |
| Target input length | `512` tokens |
| Actual input length | `497` tokens |
| Output length | `128` tokens |
| Max model length | `1024` |
| GPU memory utilization | `0.94` for `c<=64`; `0.88` for `c=128` |

## Metrics

- `Concurrency`: number of in-flight requests served at the same time.
- `Aggregate tok/s`: total generated output tokens divided by measured wall time.
- `TTFT`: time to first token.
- `Decode tok/s`: mean per-request decoding speed after the first token is produced.
- `Latency`: end-to-end request latency.

`Aggregate tok/s` is the main throughput metric.

## Main Results

The table reports aggregate output-token throughput. Concurrency is the primary batching knob.

| Attention | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 | Peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XFormers / stock | 7.27 | 13.25 | 24.65 | 42.77 | 64.36 | 79.91 | 103.75 | 120.65 | 120.65 |
| `FLASH_ATTN_V100` | 7.12 | 13.70 | 25.66 | 43.39 | 57.42 | 83.73 | 126.67 | 209.50 | 209.50 |

At the highest-throughput point:

| Backend | Concurrency | Aggregate tok/s | Mean latency | Mean TTFT | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| XFormers / stock | 128 | 120.65 | 135.69s | 49.10s | 1.50 |
| `FLASH_ATTN_V100` | 128 | 209.50 | 78.09s | 0.80s | 1.64 |

The `FLASH_ATTN_V100` c=128 result was tuned with the decode partition-size knob:

| Decode partition size | Aggregate tok/s | Mean latency | Mean TTFT | Mean decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 209.27 | 78.15s | 1.05s | 1.65 |
| 512 | 209.50 | 78.09s | 0.80s | 1.64 |
| 1024 | 195.11 | 83.91s | 0.98s | 1.53 |

The first parallelism mapping check was run at the same c=128 workload:

| Parallelism | Mode | Aggregate tok/s | Mean latency | Mean TTFT | Mean decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| `TP=8`, `PP=2` | CUDA graph | 209.50 | 78.09s | 0.80s | 1.64 |
| `TP=16`, `PP=1` | eager | 156.12 | 104.75s | 1.00s | 1.22 |

The `TP=16`, `PP=1` run uses eager mode to avoid extra CUDA graph memory at c=128. In the default container environment, it was slower because cross-node tensor-parallel communication used NCCL Socket transport rather than NCCL's InfiniBand transport.

After staging only the RDMA userspace libraries needed by NCCL into the container, the same cross-node tensor-parallel setup used `NET/IB` and `GDRDMA`:

| Parallelism | NCCL transport | Mode | Aggregate tok/s | Mean latency | Mean TTFT | Mean decode tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `TP=16`, `PP=1` | Socket | eager | 156.12 | 104.75s | 1.00s | 1.22 |
| `TP=16`, `PP=1` | `NET/IB` + `GDRDMA` | eager | 351.62 | 46.48s | 0.56s | 2.77 |

This makes NCCL transport the largest HPC-side optimization in the project: enabling true IB/GDRDMA made the cross-node TP run `2.25x` faster than the same run over Socket transport, and `67.8%` faster than the previous `TP=8`, `PP=2` best.

## Interpretation

Continuous batching is the main throughput lever. On stock vLLM, aggregate throughput increases from `7.27 tok/s` at `c=1` to `120.65 tok/s` at `c=128`, a `16.60x` speedup on the same 16-GPU allocation.

The V100 FlashAttention fork matters most at high concurrency. At `c=128`, it increases aggregate throughput by `73.6%` over stock and greatly reduces TTFT. This makes the hardware-oriented optimization visible, not just a parameter tweak.

Parallelism layout depends on the interconnect actually exposed inside the container. With Socket transport, `TP=8`, `PP=2` is better because tensor-parallel collectives stay inside each 8-GPU node. With NCCL `NET/IB` and GDRDMA available, `TP=16`, `PP=1` becomes the fastest configuration because cross-node tensor-parallel collectives are no longer forced through Socket transport.

This is the most HPC-specific result in the study. The benchmark did not only tune batch size or vLLM flags; it exposed a container/runtime issue where the job was using an IB network interface but not the NCCL InfiniBand transport. Fixing that transport path improved the final dense-model throughput more than the attention backend alone.

The tradeoff is latency. More simultaneous requests keep the GPUs busier, but each request spends more time waiting behind prefill and batched decoding work. The result is suitable for throughput-oriented offline or batched serving workloads, not low-latency interactive serving.

The `c=128` runs also expose a capacity detail: the model can run at high concurrency only when enough memory is left for initialization and KV cache. Lowering `GPU_MEMORY_UTILIZATION` to `0.88` made the largest active batch stable.

## Capacity Boundary

A 1-node / 8 x V100 configuration was tested as a capacity boundary for `Llama 3.1 405B GPTQ`, not as a successful baseline. With `TP=8`, `PP=1`, and `max_model_len=1024`, vLLM reported that the available KV cache could store only `480` tokens. This workload therefore requires the 2-node setup.

## Additional Model Checks

These runs are kept as feasibility checks only. They are not the main comparison because the Qwen models are sparse MoE models with fewer active parameters per generated token.

| Model | Allocation | Parallelism | Backend | Peak aggregate tok/s |
| --- | --- | --- | --- | ---: |
| `Qwen3 235B-A22B MoE GPTQ` | 1 node / 8 x V100 | `TP=8`, `PP=1` | `FLASH_ATTN_V100` | 416.17 @ c=64 |
| `Qwen3.5 397B-A17B MoE GPTQ` | 2 nodes / 16 x V100 | `TP=16`, `PP=1` | `FLASH_ATTN_V100` | 74.16 @ c=64 |

These results show that model architecture strongly affects throughput and should not be mixed into the main dense-model optimization story.

## Reproduce

Do not install packages or run inference on the login node. Create the vLLM environment through Slurm on a compute node:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Submit the stock `Llama 3.1 405B GPTQ` c=128 run by overriding the shared Slurm script:

```bash
MODEL_ID=hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 \
SERVED_MODEL_NAME=llama31-405b-gptq \
QUANTIZATION=gptq \
HF_CACHE_ROOT=/work/$USER/hf-cache-llama31-405b-gptq \
TP_SIZE=8 \
PP_SIZE=2 \
MAX_MODEL_LEN=1024 \
GPU_MEMORY_UTILIZATION=0.88 \
INPUT_TOKENS=512 \
OUTPUT_TOKENS=128 \
WARMUP=1 \
CONCURRENCY=128 \
RUNS=256 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=32768 \
EXPERIMENT_NAME=llama405b-c128-i512-o128 \
sbatch slurm/vllm_70b_16v100.slurm
```

Run the V100 FlashAttention c=128 comparison with:

```bash
EXPERIMENT_NAME=1cat-llama31-405b-c128-flash-v100 \
GPU_MEMORY_UTILIZATION=0.88 \
MAX_NUM_SEQS=128 \
CONCURRENCY_LEVELS=128 \
RUNS_PER_CONCURRENCY_FACTOR=2 \
VLLM_FLASH_V100_DECODE_PARTITION_SIZE=512 \
sbatch slurm/vllm_1cat_llama31_405b_sweep_16v100.slurm
```

Run the NCCL IB/GDRDMA comparison by staging the host RDMA userspace libraries into a small directory during a Slurm job, then passing that directory into the container:

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

That script expects a prebuilt vLLM fork environment at `.venv-1cat-vllm-sm70` and runs inside a CUDA 12.8 Apptainer image. The benchmark script counts streaming token events so fixed-length streamed outputs are measured correctly.

Each run writes outputs to:

```text
runs/<run-id>/
```

The useful files are `summary.json`, `experiment.env`, and `vllm-server.log`.
