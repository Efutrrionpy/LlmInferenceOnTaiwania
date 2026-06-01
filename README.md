# LLM Inference Batch Optimization on Taiwania 2

This repository documents LLM inference throughput experiments on Taiwania 2 V100 GPU nodes. The main question is: under a one-hour HPC allocation with up to 2 nodes and 16 V100 GPUs, how can we increase aggregate output-token throughput for large language model inference?

The experiments use vLLM `0.7.0`. V100 is a Volta GPU with compute capability `7.0`, while newer vLLM releases are less suitable for this hardware generation. Therefore, this project pins a vLLM version that still works on V100.

## TL;DR

- The most effective optimization is continuous batching, where multiple requests are admitted into the vLLM scheduler at the same time.
- For the 72B GPTQ model on 16 x V100, increasing concurrency from `1` to `64` improves aggregate throughput from `41.92 tok/s` to `481.82 tok/s`, a `11.49x` speedup.
- For the 104B GPTQ model, concurrency `16` reaches `201.72 tok/s`, which is `9.21x` faster than c=1.
- For the 405B GPTQ INT4 model, the model cache is about `205G` and can be loaded successfully on 16 x V100. At concurrency `64`, it reaches `103.85 tok/s`, a `15.12x` speedup over c=1.
- 2 nodes provide the highest total throughput, but 1 node has better per-GPU efficiency. More GPUs do not scale linearly because cross-node execution and pipeline parallelism introduce overhead.
- FlashAttention-2 is not a practical optimization path for this V100/vLLM stack. The runtime falls back to XFormers.

## Experiment Setup

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| Allocation | 2 nodes, 16 x V100-SXM2-32GB, one hour |
| Framework | vLLM `0.7.0` |
| Quantization | GPTQ / GPTQ INT4 |
| Default parallelism | `TP_SIZE=8`, `PP_SIZE=2` |
| Main benchmark shape | about 500 input tokens, 128 output tokens |
| Large-model benchmark shape | about 125 input tokens, 32 output tokens |

Tested models:

| Size | Model | Notes |
| ---: | --- | --- |
| 72B | `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` | main throughput curve |
| 104B | `alpindale/c4ai-command-r-plus-GPTQ` | 100B-class comparison, cache about 55G |
| 405B | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` | large-model capacity test, cache about 205G |

Metric meanings:

- `Concurrency`: the number of in-flight requests served at the same time.
- `Requests`: the total number of measured benchmark requests.
- `Aggregate tok/s`: total output tokens generated per second by the whole system. This is the main throughput metric in this project.
- `Decode tok/s`: per-request decoding speed after the first token is produced. As batch size increases, aggregate tok/s usually increases, while per-request decode tok/s usually decreases.

## Reproduce

Before submitting jobs, replace `YOUR_ACCOUNT` in the Slurm files with the project allocation account. For gated models such as Llama, set `HF_TOKEN` before submission.

Create the vLLM environment on a compute node. Do not install packages or run inference on the login node:

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

Submit the 72B baseline:

```bash
sbatch slurm/vllm_70b_16v100.slurm
```

Submit the 104B and 405B runs:

```bash
sbatch slurm/vllm_command_r_plus_16v100.slurm
sbatch slurm/vllm_llama31_405b_gptq_16v100.slurm
```

Batch throughput example:

```bash
EXPERIMENT_NAME=batch-c64 \
TP_SIZE=8 \
PP_SIZE=2 \
MAX_NUM_SEQS=64 \
MAX_NUM_BATCHED_TOKENS=16384 \
CONCURRENCY=64 \
RUNS=256 \
sbatch slurm/vllm_70b_16v100.slurm
```

Each run writes outputs to:

```text
runs/<run-id>/
```

The most important files are `summary.json`, `experiment.env`, and `vllm-server.log`. `summary.json` contains the benchmark statistics, `experiment.env` records Slurm/model/parallelism settings, and `vllm-server.log` can be used to check the backend, NCCL behavior, and model loading status.

## 72B Batch Throughput

The 2-node batch scaling runs use 16 x V100 with `TP_SIZE=8` and `PP_SIZE=2`.

| Concurrency | Requests | Aggregate tok/s | Speedup | Mean latency s | Mean TTFT s | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 41.92 | 1.00x | 3.050 | 0.259 | 45.56 |
| 4 | 16 | 121.50 | 2.90x | 4.205 | 0.480 | 34.14 |
| 8 | 32 | 219.81 | 5.24x | 4.651 | 0.648 | 31.81 |
| 16 | 64 | 322.95 | 7.70x | 6.334 | 1.210 | 24.91 |
| 24 | 96 | 350.08 | 8.35x | 8.761 | 1.668 | 18.00 |
| 32 | 128 | 406.27 | 9.69x | 10.068 | 2.150 | 16.14 |
| 48 | 192 | 453.71 | 10.82x | 13.521 | 3.197 | 12.43 |
| 64 | 256 | 481.82 | 11.49x | 16.979 | 4.181 | 10.04 |

Batching greatly improves total system throughput, but the tradeoff is higher latency. c=64 gives the highest throughput among the tested settings. c=48 is a more balanced point between throughput and latency: from c=48 to c=64, throughput improves by only about `6.2%`, while mean latency increases by about `25.6%`.

## 1-Node vs 2-Node

For batch inference, 2 nodes provide the highest total throughput, while 1 node provides better per-GPU efficiency.

| Nodes | GPUs | Concurrency | Aggregate tok/s | Tok/s/GPU | Mean latency s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 32 | 406.27 | 25.39 | 10.068 |
| 1 | 8 | 32 | 299.35 | 37.42 | 13.666 |
| 2 | 16 | 64 | 481.82 | 30.11 | 16.979 |
| 1 | 8 | 64 | 334.06 | 41.76 | 24.498 |

At c=32, the 1-node run uses only half the GPUs but still reaches `73.7%` of the 2-node throughput. This shows that cross-node parallelism can improve total throughput, but the scaling is not linear.

## Larger Models

The 104B and 405B runs test whether batch throughput gains also hold for larger models. The results show that larger models have slower single-stream throughput, but continuous batching still provides large aggregate throughput gains.

| Model | Cache | Concurrency | Requests | Output tokens | Aggregate tok/s | Speedup | Mean latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Command R+ 104B GPTQ | 55G | 1 | 8 | 128 | 21.90 | 1.00x | 5.840 |
| Command R+ 104B GPTQ | 55G | 16 | 64 | 128 | 201.72 | 9.21x | 10.138 |
| Llama 3.1 405B GPTQ INT4 | 205G | 1 | 4 | 32 | 6.87 | 1.00x | 4.655 |
| Llama 3.1 405B GPTQ INT4 | 205G | 2 | 8 | 32 | 12.12 | 1.77x | 5.249 |
| Llama 3.1 405B GPTQ INT4 | 205G | 4 | 16 | 32 | 22.00 | 3.20x | 5.785 |
| Llama 3.1 405B GPTQ INT4 | 205G | 8 | 32 | 32 | 39.50 | 5.75x | 6.464 |
| Llama 3.1 405B GPTQ INT4 | 205G | 16 | 64 | 32 | 60.46 | 8.80x | 8.450 |
| Llama 3.1 405B GPTQ INT4 | 205G | 32 | 128 | 32 | 77.90 | 11.34x | 13.114 |
| Llama 3.1 405B GPTQ INT4 | 205G | 64 | 256 | 32 | 103.85 | 15.12x | 19.678 |

The 405B result also answers the capacity question. Although 16 x V100 provides 512GB of theoretical GPU memory, a 405B model is only practical here because the weights are quantized. With the GPTQ INT4 checkpoint, the cache occupies about `205G`, and the model can be loaded and benchmarked successfully on this setup.

## Hardware Notes

Several hardware- and topology-oriented control experiments were also tested:

| Setting | Aggregate tok/s | Mean latency s | Observation |
| --- | ---: | ---: | --- |
| 1 node, `TP=8`, `PP=1`, multiprocessing | 47.47 | 2.694 | Best single-request setting; avoids cross-node pipeline overhead |
| 2 nodes, `TP=8`, `PP=2` | 42.60 | 3.002 | Default 16-GPU layout; more resources but slower for a single request |
| 2 nodes, `TP=4`, `PP=4` | 37.17 | 3.440 | More pipeline stages hurt single-request inference |
| 2 nodes, `NCCL_P2P_DISABLE=1` | 29.25 | 4.374 | Disabling intra-node GPU P2P causes a large slowdown |
| 1 node, forced `NCCL_PROTO=LL128` | 42.81 | 2.988 | Slower than NCCL auto-selection |
| 1 node, forced `NCCL_ALGO=Tree` | 47.12 | 2.714 | Close to auto-selection, but still slightly slower |

These results show that more GPUs are not automatically faster for single-request inference. A better HPC interpretation combines continuous batching with topology-aware parallelism: batching increases total throughput, while topology explains why adding another node does not scale linearly.

## Conclusion

The main optimization in this project is continuous batching. Under a fixed 16 x V100 allocation, the 72B GPTQ model improves from c=1 to c=64 by `11.49x` in aggregate output-token throughput. The 104B and 405B models show the same trend, so the result is not limited to one model.

From an HPC perspective, the goal is not to make a single request as fast as possible, but to process as many output tokens as possible within a fixed allocation. If the goal is maximum total throughput, 2 nodes are better. If the goal is GPU efficiency, 1 node can be better. On V100, FlashAttention-2 is not available, so the meaningful optimization story is vLLM continuous batching, parallelism topology, and NCCL/P2P control experiments.
