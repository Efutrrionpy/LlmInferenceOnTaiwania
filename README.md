# LLM Inference Throughput on Taiwania 2

This repository documents focused LLM inference experiments on Taiwania 2 using one-hour HPC allocations on NVIDIA V100 nodes. The main experiment asks how much aggregate output-token throughput can be improved for `Llama 3.1 405B GPTQ` by using vLLM continuous batching, then checks a V100-specific FlashAttention fork as a hardware-oriented extension.

The repository also includes successful MoE contrasts using `Qwen3 235B-A22B MoE GPTQ` on 1 node / 8 x V100 and `Qwen3.5 397B-A17B MoE GPTQ` on 2 nodes / 16 x V100. These are not apples-to-apples model comparisons: the Qwen models are sparse MoE models with fewer active parameters per generated token.

## TL;DR

- `Llama 3.1 405B GPTQ` loads successfully on 2 nodes / 16 x V100 with `TP=8`, `PP=2`.
- The benchmark uses target `512` input tokens and `128` output tokens. The tokenizer produced `497` actual input tokens for this prompt.
- With stock vLLM, increasing concurrency from `1` to `128` improves aggregate throughput from `7.27 tok/s` to `120.65 tok/s`, a `16.60x` speedup.
- The tradeoff is latency: mean latency rises from `17.60s` at c=1 to `135.69s` at c=128.
- On the stock V100/vLLM stack, the attention backend is XFormers, not FlashAttention.
- A V100-specific vLLM fork with `FLASH_ATTN_V100` raises the c=128 result to `208.54 tok/s`, a `72.8%` improvement over stock c=128.
- The c=128 Llama runs use `GPU_MEMORY_UTILIZATION=0.88` to leave enough GPU memory headroom for initialization and KV cache.
- `Qwen3 235B-A22B MoE GPTQ` runs on 1 node / 8 x V100 and reaches `416.17 tok/s` at c=64.
- `Qwen3.5 397B-A17B MoE GPTQ` loads on 2 nodes / 16 x V100 with `TP=16`, `PP=1`, and reaches `74.16 tok/s` at c=64 using `FLASH_ATTN_V100`.
- A 1-node `Llama 3.1 405B GPTQ` attempt cannot support the same workload shape because the available KV cache only supports `480` tokens while the required `max_model_len` is `1024`.

## Llama 3.1 405B GPTQ Setup

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x NVIDIA V100-SXM2-32GB, one hour per Slurm job |
| Model | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` |
| Framework | vLLM `0.7.0` stock baseline; vLLM `1.1.0` V100 fork comparison |
| Quantization | GPTQ INT4 |
| Parallelism | `TP_SIZE=8`, `PP_SIZE=2` |
| Attention backend | XFormers baseline; `FLASH_ATTN_V100` fork comparison |
| Target prompt length | `512` tokens |
| Actual prompt length | `497` tokens |
| Output length | `128` tokens |
| Max model length | `1024` |
| GPU memory utilization | `0.94` for c<=64; `0.88` for c=128 |

## Qwen3 235B-A22B MoE GPTQ Setup

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 1 node, 8 x NVIDIA V100-SXM2-32GB, one hour per Slurm job |
| Model | [`Qwen/Qwen3-235B-A22B-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen3-235B-A22B-GPTQ-Int4) |
| Framework | vLLM `1.1.0` V100 fork |
| Quantization | GPTQ INT4 |
| Parallelism | `TP_SIZE=8`, `PP_SIZE=1`, `distributed_executor_backend=mp` |
| Attention backend | `FLASH_ATTN_V100` |
| Target prompt length | `512` tokens |
| Actual prompt length | `498` tokens |
| Output length | `128` tokens |
| Max model length | `1024` |
| GPU memory utilization | `0.94` |

## Qwen3.5 397B-A17B MoE GPTQ Setup

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x NVIDIA V100-SXM2-32GB, one hour per Slurm job |
| Model | [`Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4) |
| Framework | vLLM `1.1.0` V100 fork |
| Quantization | GPTQ INT4 |
| Parallelism | `TP_SIZE=16`, `PP_SIZE=1`, `distributed_executor_backend=ray` |
| Attention backend | `FLASH_ATTN_V100` |
| Target prompt length | `512` tokens |
| Actual prompt length | `498` tokens |
| Output length | `128` tokens |
| Max model length | `1024` |
| GPU memory utilization | `0.92` |

## Metrics

- `Concurrency`: the number of in-flight requests served at the same time.
- `Requests`: the number of measured benchmark requests, excluding warmup.
- `Aggregate tok/s`: total generated output tokens divided by measured wall time. This is the main throughput metric.
- `Tok/s/GPU`: aggregate tok/s divided by the number of GPUs used in that experiment.
- `Decode tok/s`: mean per-request decoding speed after the first token is produced.
- `TTFT`: time to first token.

## Integrated Results

The main comparison uses peak aggregate output-token throughput for each model and attention backend. `XFormers / stock` is the stock vLLM attention path for the `Llama 3.1 405B GPTQ` baseline. The `Qwen3 235B-A22B MoE GPTQ` and `Qwen3.5 397B-A17B MoE GPTQ` rows use the V100 fork.

| Model | Allocation | Parallelism | XFormers / stock peak | `FLASH_ATTN_V100` peak | Flash gain |
| --- | --- | --- | ---: | ---: | ---: |
| Llama 3.1 405B GPTQ | 2 nodes / 16 x V100 | `TP=8`, `PP=2` | 120.65 tok/s @ c=128 | 208.54 tok/s @ c=128 | +72.8% |
| Qwen3 235B-A22B MoE GPTQ | 1 node / 8 x V100 | `TP=8`, `PP=1` | not measured | 416.17 tok/s @ c=64 | n/a |
| Qwen3.5 397B-A17B MoE GPTQ | 2 nodes / 16 x V100 | `TP=16`, `PP=1` | not available | 74.16 tok/s @ c=64 | n/a |

The batch-throughput sweep below reports aggregate output tok/s. It shows how concurrency increases total throughput while also increasing per-request latency.

| Model | Attention | GPUs | c=1 | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 | Peak tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.1 405B GPTQ | XFormers / stock | 16 | 7.27 | 13.25 | 24.65 | 42.77 | 64.36 | 79.91 | 103.75 | 120.65 | 120.65 |
| Llama 3.1 405B GPTQ | `FLASH_ATTN_V100` | 16 | 7.12 | 13.70 | 25.66 | 43.39 | 57.42 | 83.73 | 126.67 | 208.54 | 208.54 |
| Qwen3 235B-A22B MoE GPTQ | `FLASH_ATTN_V100` | 8 | 36.44 | 58.17 | 106.09 | 179.29 | 268.97 | 350.30 | 416.17 |  | 416.17 |
| Qwen3.5 397B-A17B MoE GPTQ | `FLASH_ATTN_V100` | 16 | 20.46 | 25.70 | 39.88 | 53.99 | 60.95 | 66.71 | 74.16 |  | 74.16 |

`Llama 3.1 405B GPTQ` is the only complete two-backend comparison. The V100 FlashAttention fork improves the highest-throughput point by about `72.8%` over stock vLLM c=128 (`208.54` vs `120.65 tok/s`) and greatly lowers TTFT. For `Qwen3.5 397B-A17B MoE GPTQ`, `FLASH_ATTN_V100` is the successful backend in this environment.

## Interpretation

Continuous batching is the main effective optimization in this setup. For a single request, the system produces about `7.27` output tokens per second. With 128 concurrent requests, vLLM keeps the GPUs busier and reaches `120.65` aggregate output tokens per second on the stock backend.

The cost is per-request responsiveness. As concurrency increases, each request spends more time waiting behind prefilling and batched decoding work. This is why aggregate tok/s increases while decode tok/s per request decreases.

From an HPC perspective, the result is still meaningful even though the absolute tok/s is not high. The workload includes `Llama 3.1 405B GPTQ` on V100 GPUs, and the experiment shows how throughput changes when the same fixed allocation is used for more simultaneous inference work. The c=128 runs also show a hardware capacity detail: lowering GPU memory utilization to `0.88` leaves enough headroom for the largest active batch. The V100 FlashAttention fork adds a large high-concurrency gain, but the main lesson remains that batching dominates the throughput improvement.

## Llama 3.1 405B GPTQ Capacity Note

For `Llama 3.1 405B GPTQ`, the 1-node / 8 x V100 configuration was tested as a capacity boundary, not as a successful throughput baseline. With `TP=8`, `PP=1`, `max_model_len=1024`, and the same model, vLLM reported that the available KV cache can store only `480` tokens. Therefore this workload shape requires the 2-node setup.

This also explains why the `Llama 3.1 405B GPTQ` result depends on quantization and parallelism. The model can be loaded on the 16-GPU allocation, but memory headroom is still tight enough that context length and KV cache capacity matter.

## Reproduce

Do not install packages or run inference on the login node. Create the vLLM environment through Slurm on a compute node:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Submit a `Llama 3.1 405B GPTQ` baseline run by overriding the shared Slurm script:

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

Each run writes outputs to:

```text
runs/<run-id>/
```

The useful files are `summary.json`, `experiment.env`, and `vllm-server.log`. `summary.json` contains benchmark statistics, `experiment.env` records the model and parallelism settings, and `vllm-server.log` shows the backend and capacity messages.

The V100 FlashAttention comparison was run with:

```bash
sbatch slurm/vllm_1cat_llama31_405b_sweep_16v100.slurm
```

That script expects a prebuilt vLLM fork environment at `.venv-1cat-vllm-sm70` and runs inside a CUDA 12.8 Apptainer image. The benchmark script counts streaming token events so fixed-length streamed outputs are measured correctly.

The `Qwen3 235B-A22B MoE GPTQ` comparison was run with:

```bash
sbatch slurm/vllm_1cat_qwen3_235b_moe_sweep_8v100.slurm
```

It uses the same V100 fork environment and benchmark script.

The `Qwen3.5 397B-A17B MoE GPTQ` sweep was run with:

```bash
sbatch slurm/vllm_1cat_qwen35_397b_moe_sweep_16v100.slurm
```

The same shared sweep driver exposes `ATTENTION_BACKEND` for additional backend experiments.
