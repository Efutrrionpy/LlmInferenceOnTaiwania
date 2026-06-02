# 405B LLM Inference Baseline on Taiwania 2

This repository documents a focused LLM inference baseline on Taiwania 2 using a one-hour HPC allocation with 2 V100 nodes. The experiment asks how much aggregate output-token throughput can be improved for a 405B-class model by using vLLM continuous batching.

The main model is `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4`. It is a quantized GPTQ INT4 checkpoint, so it can be loaded on 16 x V100-SXM2-32GB even though a dense 405B model would not fit in this allocation.

## TL;DR

- The 405B GPTQ INT4 model loads successfully on 2 nodes / 16 x V100 with `TP=8`, `PP=2`.
- The benchmark uses target `512` input tokens and `128` output tokens. The tokenizer produced `497` actual input tokens for this prompt.
- Increasing concurrency from `1` to `32` improves aggregate throughput from `7.27 tok/s` to `79.91 tok/s`, a `10.99x` speedup.
- The tradeoff is latency: mean latency rises from `17.60s` at c=1 to `51.22s` at c=32.
- On this V100/vLLM stack, the attention backend is XFormers, not FlashAttention.
- A 1-node attempt cannot support this workload shape because the available KV cache only supports `480` tokens while the required `max_model_len` is `1024`.

## Experiment Setup

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x NVIDIA V100-SXM2-32GB, one hour per Slurm job |
| Model | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` |
| Framework | vLLM `0.7.0` |
| Quantization | GPTQ INT4 |
| Parallelism | `TP_SIZE=8`, `PP_SIZE=2` |
| Attention backend | XFormers |
| Target prompt length | `512` tokens |
| Actual prompt length | `497` tokens |
| Output length | `128` tokens |
| Max model length | `1024` |
| GPU memory utilization | `0.94` |

## Metrics

- `Concurrency`: the number of in-flight requests served at the same time.
- `Requests`: the number of measured benchmark requests, excluding warmup.
- `Aggregate tok/s`: total generated output tokens divided by measured wall time. This is the main throughput metric.
- `Tok/s/GPU`: aggregate tok/s divided by 16 GPUs.
- `Decode tok/s`: mean per-request decoding speed after the first token is produced.
- `TTFT`: time to first token.

## 405B Batch Throughput

All successful results below use 2 nodes / 16 x V100 with `TP=8`, `PP=2`.

| Concurrency | Requests | Max seqs | Batched toks | Aggregate tok/s | Speedup | Tok/s/GPU | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 1 | 2,048 | 7.27 | 1.00x | 0.45 | 17.60 | 1.05 | 7.67 |
| 2 | 8 | 2 | 2,048 | 13.25 | 1.82x | 0.83 | 19.28 | 1.61 | 7.19 |
| 4 | 16 | 4 | 4,096 | 24.65 | 3.39x | 1.54 | 20.75 | 2.01 | 6.78 |
| 8 | 32 | 8 | 8,192 | 42.77 | 5.88x | 2.67 | 23.92 | 3.25 | 6.16 |
| 16 | 64 | 16 | 16,384 | 64.36 | 8.85x | 4.02 | 31.80 | 5.65 | 4.88 |
| 32 | 128 | 32 | 32,768 | 79.91 | 10.99x | 4.99 | 51.22 | 10.57 | 3.14 |

The best throughput in this baseline is c=32. However, c=16 is a useful balance point: it reaches `80.5%` of c=32 throughput with much lower mean latency (`31.80s` instead of `51.22s`).

## Interpretation

Continuous batching is the main effective optimization in this setup. For a single request, the system produces about `7.27` output tokens per second. With 32 concurrent requests, vLLM keeps the GPUs busier and reaches `79.91` aggregate output tokens per second.

The cost is per-request responsiveness. As concurrency increases, each request spends more time waiting behind prefilling and batched decoding work. This is why aggregate tok/s increases while decode tok/s per request decreases.

From an HPC perspective, the result is still meaningful even though the absolute tok/s is not high. The workload is a 405B-class model on V100 GPUs, and the experiment shows how throughput changes when the same fixed allocation is used for more simultaneous inference work.

## Capacity Note

The 1-node / 8 x V100 configuration was tested as a capacity boundary, not as a successful throughput baseline. With `TP=8`, `PP=1`, `max_model_len=1024`, and the same 405B GPTQ INT4 model, vLLM reported that the available KV cache can store only `480` tokens. Therefore this workload shape requires the 2-node setup.

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
