# LLM Inference Throughput on Taiwania 2

This repository documents focused LLM inference experiments on Taiwania 2 using one-hour HPC allocations on NVIDIA V100 nodes. The main experiment asks how much aggregate output-token throughput can be improved for a 405B-class GPTQ model by using vLLM continuous batching, then checks a V100-specific FlashAttention fork as a hardware-oriented extension.

The repository also includes successful Qwen MoE contrasts using `Qwen/Qwen3-235B-A22B-GPTQ-Int4` on 1 node / 8 x V100 and `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` on 2 nodes / 16 x V100. These are not apples-to-apples model comparisons: the Qwen models are sparse MoE models with fewer active parameters per generated token.

## TL;DR

- The 405B GPTQ INT4 model loads successfully on 2 nodes / 16 x V100 with `TP=8`, `PP=2`.
- The benchmark uses target `512` input tokens and `128` output tokens. The tokenizer produced `497` actual input tokens for this prompt.
- With stock vLLM, increasing concurrency from `1` to `32` improves aggregate throughput from `7.27 tok/s` to `79.91 tok/s`, a `10.99x` speedup.
- The tradeoff is latency: mean latency rises from `17.60s` at c=1 to `51.22s` at c=32.
- On the stock V100/vLLM stack, the attention backend is XFormers, not FlashAttention.
- A V100-specific vLLM fork with `FLASH_ATTN_V100` raises the c=32 result to `83.73 tok/s`, but it is not uniformly faster at every concurrency level.
- The Qwen MoE GPTQ INT4 contrast runs on 1 node / 8 x V100 and reaches `416.17 tok/s` at c=64.
- The larger Qwen3.5 397B-A17B GPTQ INT4 MoE run loads on 2 nodes / 16 x V100 with `TP=16`, `PP=1`, and reaches `66.71 tok/s` at c=32 using `FLASH_ATTN_V100`.
- A 1-node 405B attempt cannot support the same workload shape because the available KV cache only supports `480` tokens while the required `max_model_len` is `1024`.

## 405B Setup

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
| GPU memory utilization | `0.94` |

## Qwen MoE Setup

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

## Qwen3.5 397B MoE Setup

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

## 405B Batch Throughput

All successful results below use 2 nodes / 16 x V100 with `TP=8`, `PP=2`. This is the stock vLLM baseline.

| Concurrency | Requests | Max seqs | Batched toks | Aggregate tok/s | Speedup | Tok/s/GPU | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 1 | 2,048 | 7.27 | 1.00x | 0.45 | 17.60 | 1.05 | 7.67 |
| 2 | 8 | 2 | 2,048 | 13.25 | 1.82x | 0.83 | 19.28 | 1.61 | 7.19 |
| 4 | 16 | 4 | 4,096 | 24.65 | 3.39x | 1.54 | 20.75 | 2.01 | 6.78 |
| 8 | 32 | 8 | 8,192 | 42.77 | 5.88x | 2.67 | 23.92 | 3.25 | 6.16 |
| 16 | 64 | 16 | 16,384 | 64.36 | 8.85x | 4.02 | 31.80 | 5.65 | 4.88 |
| 32 | 128 | 32 | 32,768 | 79.91 | 10.99x | 4.99 | 51.22 | 10.57 | 3.14 |

The best throughput in this baseline is c=32. However, c=16 is a useful balance point: it reaches `80.5%` of c=32 throughput with much lower mean latency (`31.80s` instead of `51.22s`).

## Qwen MoE Batch Throughput

This sweep uses `Qwen/Qwen3-235B-A22B-GPTQ-Int4` on 1 node / 8 x V100 with `TP=8`, `PP=1`, `max_num_seqs=64`, and `max_num_batched_tokens=65536`.

| Concurrency | Requests | Aggregate tok/s | Speedup | Tok/s/GPU | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 36.44 | 1.00x | 4.55 | 3.510 | 0.112 | 37.37 |
| 2 | 8 | 58.17 | 1.60x | 7.27 | 4.392 | 0.156 | 29.99 |
| 4 | 16 | 106.09 | 2.91x | 13.26 | 4.817 | 0.171 | 27.34 |
| 8 | 32 | 179.29 | 4.92x | 22.41 | 5.702 | 0.178 | 22.99 |
| 16 | 64 | 268.97 | 7.38x | 33.62 | 7.604 | 0.203 | 17.16 |
| 32 | 128 | 350.30 | 9.61x | 43.79 | 11.679 | 0.263 | 11.12 |
| 64 | 256 | 416.17 | 11.42x | 52.02 | 19.645 | 0.455 | 6.62 |

The MoE run shows the same batching pattern as the 405B run: aggregate throughput rises sharply with concurrency while per-request decode speed falls. Because this model has far fewer active parameters per token, it reaches much higher aggregate throughput even on half as many GPUs.

## Qwen3.5 397B MoE Batch Throughput

This sweep uses `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` on 2 nodes / 16 x V100 with `TP=16`, `PP=1`, `max_num_seqs=32`, `max_num_batched_tokens=8192`, and `FLASH_ATTN_V100`.

| Concurrency | Requests | Aggregate tok/s | Speedup | Tok/s/GPU | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 20.46 | 1.00x | 1.28 | 6.252 | 2.658 | 35.33 |
| 2 | 4 | 25.70 | 1.26x | 1.61 | 9.943 | 4.120 | 23.21 |
| 4 | 8 | 39.88 | 1.95x | 2.49 | 12.827 | 6.572 | 22.19 |
| 8 | 16 | 53.99 | 2.64x | 3.37 | 18.951 | 11.635 | 19.33 |
| 16 | 32 | 60.95 | 2.98x | 3.81 | 33.587 | 23.084 | 13.27 |
| 32 | 64 | 66.71 | 3.26x | 4.17 | 61.360 | 32.914 | 5.20 |

For this larger MoE model, batching still improves aggregate throughput, but the gain is smaller than the 235B-A22B sweep. The c=32 result is the highest throughput point, while c=16 is a more balanced point with `91.4%` of peak throughput and much lower latency.

## V100 FlashAttention Fork

The hardware-specific comparison uses a vLLM fork with `FLASH_ATTN_V100`, the same 405B GPTQ model, and the same 2-node `TP=8`, `PP=2` layout. For this sweep, `max_num_seqs=32` and `max_num_batched_tokens=32768` are fixed for all rows.

| Concurrency | Requests | Aggregate tok/s | Speedup | Tok/s/GPU | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 7.12 | 1.00x | 0.45 | 17.988 | 0.144 | 7.12 |
| 2 | 8 | 13.70 | 1.92x | 0.86 | 18.665 | 0.310 | 6.92 |
| 4 | 16 | 25.66 | 3.60x | 1.60 | 19.943 | 0.412 | 6.50 |
| 8 | 32 | 43.39 | 6.09x | 2.71 | 23.586 | 0.357 | 5.47 |
| 16 | 64 | 57.42 | 8.06x | 3.59 | 35.646 | 0.535 | 3.62 |
| 32 | 128 | 83.73 | 11.76x | 5.23 | 48.905 | 0.460 | 2.62 |

The fork is useful as an HPC-oriented experiment because it changes the GPU attention kernel path. It improves the highest-throughput point by about `4.8%` over stock vLLM c=32 (`83.73` vs `79.91 tok/s`) and greatly lowers TTFT. It does not dominate the stock baseline at every point: c=16 is lower than the stock result (`57.42` vs `64.36 tok/s`).

## Interpretation

Continuous batching is the main effective optimization in this setup. For a single request, the system produces about `7.27` output tokens per second. With 32 concurrent requests, vLLM keeps the GPUs busier and reaches `79.91` aggregate output tokens per second.

The cost is per-request responsiveness. As concurrency increases, each request spends more time waiting behind prefilling and batched decoding work. This is why aggregate tok/s increases while decode tok/s per request decreases.

From an HPC perspective, the result is still meaningful even though the absolute tok/s is not high. The workload is a 405B-class model on V100 GPUs, and the experiment shows how throughput changes when the same fixed allocation is used for more simultaneous inference work. The V100 FlashAttention fork adds a small high-concurrency gain, but the main lesson remains that batching dominates the throughput improvement.

## 405B Capacity Note

For the 405B model, the 1-node / 8 x V100 configuration was tested as a capacity boundary, not as a successful throughput baseline. With `TP=8`, `PP=1`, `max_model_len=1024`, and the same 405B GPTQ INT4 model, vLLM reported that the available KV cache can store only `480` tokens. Therefore this workload shape requires the 2-node setup.

This also explains why the 405B result depends on quantization and parallelism. The model can be loaded on the 16-GPU allocation, but memory headroom is still tight enough that context length and KV cache capacity matter.

## Reproduce

Do not install packages or run inference on the login node. Create the vLLM environment through Slurm on a compute node:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Submit a 405B baseline run by overriding the shared Slurm script:

```bash
MODEL_ID=hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4 \
SERVED_MODEL_NAME=llama31-405b-gptq \
QUANTIZATION=gptq \
HF_CACHE_ROOT=/work/$USER/hf-cache-llama31-405b-gptq \
TP_SIZE=8 \
PP_SIZE=2 \
MAX_MODEL_LEN=1024 \
GPU_MEMORY_UTILIZATION=0.94 \
INPUT_TOKENS=512 \
OUTPUT_TOKENS=128 \
WARMUP=1 \
CONCURRENCY=32 \
RUNS=128 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=32768 \
EXPERIMENT_NAME=llama405b-c32-i512-o128 \
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

The Qwen MoE comparison was run with:

```bash
sbatch slurm/vllm_1cat_qwen3_235b_moe_sweep_8v100.slurm
```

It uses the same V100 fork environment and benchmark script.

The larger Qwen3.5 397B MoE sweep was run with:

```bash
sbatch slurm/vllm_1cat_qwen35_397b_moe_sweep_16v100.slurm
```

The same shared sweep driver exposes `ATTENTION_BACKEND` for additional backend experiments.
