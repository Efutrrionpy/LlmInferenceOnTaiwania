# HPC Experiment Log

Target system:

- Taiwania 2 GPU partition: `nycugpu_queue`
- Allocation shape: 2 nodes, 16 x V100-SXM2-32GB, 64 CPU cores
- Model: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`
- vLLM: `0.7.0`
- Default prompt shape: 498 actual input tokens, 128 output tokens

## Completed Runs

| Job ID | Experiment | TP | PP | Concurrency | Max seqs | Batched tokens | Backend | State | Main result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 930083 | latency baseline | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | mean decode 44.81 tok/s, mean latency 3.16 s |
| 930187 | force FlashAttention | 8 | 2 | 1 | 4 | auto | requested FLASH_ATTN, fell back to XFormers | COMPLETED | mean decode 44.04 tok/s over 3 measured requests |
| 930277 | topology TP4/PP4 | 4 | 4 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 37.17 tok/s, decode 41.71 tok/s, latency 3.44 s |
| 930301 | batch c=1 updated baseline | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 41.92 tok/s, decode 45.56 tok/s, latency 3.05 s |
| 930314 | batch c=4 | 8 | 2 | 4 | 32 | 8192 | XFormers | COMPLETED | aggregate 121.50 tok/s, latency 4.21 s |
| 930321 | batch c=8 | 8 | 2 | 8 | 32 | 8192 | XFormers | COMPLETED | aggregate 219.81 tok/s, latency 4.65 s |
| 930349 | batch c=16 | 8 | 2 | 16 | 32 | 8192 | XFormers | COMPLETED | aggregate 322.95 tok/s, latency 6.33 s |
| 930811 | batch c=24, 2-node | 8 | 2 | 24 | 32 | 8192 | XFormers | COMPLETED | aggregate 350.08 tok/s, latency 8.76 s |
| 930833 | batch c=32, 2-node | 8 | 2 | 32 | 32 | 8192 | XFormers | COMPLETED | aggregate 406.27 tok/s, latency 10.07 s |
| 930859 | batch c=48, 2-node | 8 | 2 | 48 | 64 | 16384 | XFormers | COMPLETED | aggregate 453.71 tok/s, latency 13.52 s |
| 930884 | batch c=64, 2-node | 8 | 2 | 64 | 64 | 16384 | XFormers | COMPLETED | aggregate 481.82 tok/s, latency 16.98 s |
| 930912 | batch c=32, 1-node mp | 8 | 1 | 32 | 64 | 16384 | XFormers | COMPLETED | aggregate 299.35 tok/s, latency 13.67 s |
| 930937 | batch c=64, 1-node mp | 8 | 1 | 64 | 64 | 16384 | XFormers | COMPLETED | aggregate 334.06 tok/s, latency 24.50 s |
| 930423 | hardware 2-node TP8/PP2 NCCL | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 42.60 tok/s, decode 46.31 tok/s, latency 3.00 s |
| 930546 | hardware 2-node TP8/PP2 no P2P | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 29.25 tok/s, decode 34.79 tok/s, latency 4.37 s |
| 930559 | hardware 1-node TP8/PP1 | 8 | 1 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 47.24 tok/s, decode 51.82 tok/s, latency 2.71 s |
| 930678 | hardware 1-node TP8/PP1 max seqs 1 | 8 | 1 | 1 | 1 | 2048 | XFormers | COMPLETED | aggregate 47.41 tok/s, decode 52.00 tok/s, latency 2.70 s |
| 930712 | hardware 1-node TP8/PP1 mp backend | 8 | 1 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 47.47 tok/s, decode 52.07 tok/s, latency 2.69 s |
| 930727 | hardware 1-node TP8/PP1 mp + NCCL LL128 | 8 | 1 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 42.81 tok/s, decode 46.52 tok/s, latency 2.99 s |
| 930741 | hardware 1-node TP8/PP1 mp + NCCL Tree | 8 | 1 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 47.12 tok/s, decode 51.68 tok/s, latency 2.71 s |
| 933544 | non-Qwen Command R+ 104B GPTQ smoke | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | output=64, aggregate 21.16 tok/s, decode 23.15 tok/s, latency 3.02 s |
| 933548 | Command R+ 104B GPTQ c=1 o128 | 8 | 2 | 1 | 4 | auto | XFormers | COMPLETED | aggregate 21.90 tok/s, decode 23.08 tok/s, latency 5.84 s |
| 933549 | Command R+ 104B GPTQ batch c=16 | 8 | 2 | 16 | 32 | 8192 | XFormers | COMPLETED | aggregate 201.72 tok/s, latency 10.14 s, 9.21x over Command R+ c=1 |
| 933557 | Llama 3.1 405B GPTQ INT4 smoke | 8 | 2 | 1 | 1 | auto | XFormers | COMPLETED | cache 205G, output=16, aggregate 4.90 tok/s, decode 7.54 tok/s, latency 3.25 s, elapsed 19:58 |
| 933580 | Llama 3.1 405B GPTQ c=1 o32 | 8 | 2 | 1 | 1 | 2048 | XFormers | COMPLETED | aggregate 6.87 tok/s, latency 4.66 s |
| 933581 | Llama 3.1 405B GPTQ c=2 o32 | 8 | 2 | 2 | 2 | 2048 | XFormers | COMPLETED | aggregate 12.12 tok/s, latency 5.25 s |
| 933582 | Llama 3.1 405B GPTQ c=4 o32 | 8 | 2 | 4 | 4 | 2048 | XFormers | COMPLETED | aggregate 22.00 tok/s, latency 5.78 s |
| 933591 | Llama 3.1 405B GPTQ c=8 o32 | 8 | 2 | 8 | 8 | 8192 | XFormers | COMPLETED | aggregate 39.50 tok/s, latency 6.46 s |
| 933592 | Llama 3.1 405B GPTQ c=16 o32 | 8 | 2 | 16 | 16 | 8192 | XFormers | COMPLETED | aggregate 60.46 tok/s, latency 8.45 s |
| 933593 | Llama 3.1 405B GPTQ c=32 o32 | 8 | 2 | 32 | 32 | 8192 | XFormers | COMPLETED | aggregate 77.90 tok/s, latency 13.11 s |
| 933601 | Llama 3.1 405B GPTQ c=64 o32 | 8 | 2 | 64 | 64 | 16384 | XFormers | COMPLETED | aggregate 103.85 tok/s, latency 19.68 s, 15.12x over 405B c=1 |

## Batch Scaling

All 2-node batch scaling runs used `TP_SIZE=8`, `PP_SIZE=2`, `MAX_MODEL_LEN=2048`, 498 actual input tokens, and 128 output tokens.

| Job ID | Concurrency | Requests | Aggregate output tok/s | Speedup vs c=1 | Mean latency s | Mean TTFT s | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 930301 | 1 | 8 | 41.92 | 1.00x | 3.050 | 0.259 | 45.56 |
| 930314 | 4 | 16 | 121.50 | 2.90x | 4.205 | 0.480 | 34.14 |
| 930321 | 8 | 32 | 219.81 | 5.24x | 4.651 | 0.648 | 31.81 |
| 930349 | 16 | 64 | 322.95 | 7.70x | 6.334 | 1.210 | 24.91 |
| 930811 | 24 | 96 | 350.08 | 8.35x | 8.761 | 1.668 | 18.00 |
| 930833 | 32 | 128 | 406.27 | 9.69x | 10.068 | 2.150 | 16.14 |
| 930859 | 48 | 192 | 453.71 | 10.82x | 13.521 | 3.197 | 12.43 |
| 930884 | 64 | 256 | 481.82 | 11.49x | 16.979 | 4.181 | 10.04 |

Observation: continuous batching is the strongest optimization in these runs. It increases 2-node system throughput from 41.92 tok/s to 481.82 tok/s, an 11.49x gain. The tradeoff is clear: mean request latency rises from 3.05 s to 16.98 s, and per-request decode rate falls from 45.56 tok/s to 10.04 tok/s.

The high-concurrency region shows diminishing returns. Moving from c=48 to c=64 improves aggregate throughput by only 6.2%, while mean latency increases by 25.6%. Therefore c=48 is a reasonable throughput/latency operating point, while c=64 is the maximum-throughput point among the tested settings.

## 405B Batch Scaling

These 2-node runs used `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4`, `TP_SIZE=8`, `PP_SIZE=2`, `MAX_MODEL_LEN=512`, about 125 actual input tokens, and 32 output tokens.

| Job ID | Concurrency | Requests | Aggregate output tok/s | Speedup vs c=1 | Mean latency s | Mean TTFT s | Mean decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 933580 | 1 | 4 | 6.87 | 1.00x | 4.655 | 0.569 | 7.59 |
| 933581 | 2 | 8 | 12.12 | 1.77x | 5.249 | 0.900 | 7.15 |
| 933582 | 4 | 16 | 22.00 | 3.20x | 5.785 | 1.127 | 6.69 |
| 933591 | 8 | 32 | 39.50 | 5.75x | 6.464 | 1.357 | 6.11 |
| 933592 | 16 | 64 | 60.46 | 8.80x | 8.450 | 1.975 | 4.83 |
| 933593 | 32 | 128 | 77.90 | 11.34x | 13.114 | 3.266 | 3.18 |
| 933601 | 64 | 256 | 103.85 | 15.12x | 19.678 | 5.717 | 2.25 |

Observation: even the 405B GPTQ model benefits strongly from continuous batching. Aggregate throughput improves by 15.12x from c=1 to c=64, while mean latency rises from 4.66 s to 19.68 s. This is a useful capacity-and-throughput result for the HPC report: 16 x V100 can carry a 205G checkpoint and still gain throughput from request batching.

## Batch Topology

The 1-node batch runs used `TP_SIZE=8`, `PP_SIZE=1`, `DISTRIBUTED_EXECUTOR_BACKEND=mp`, `MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=16384`, `MAX_MODEL_LEN=2048`, 498 actual input tokens, and 128 output tokens.

| Job ID | Nodes | GPUs | Concurrency | Aggregate output tok/s | Tok/s/GPU | Mean latency s | Mean TTFT s | Mean decode tok/s | Observation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 930833 | 2 | 16 | 32 | 406.27 | 25.39 | 10.068 | 2.150 | 16.14 | Same-concurrency 2-node reference. |
| 930912 | 1 | 8 | 32 | 299.35 | 37.42 | 13.666 | 4.192 | 13.41 | Uses half the GPUs but reaches 73.7% of 2-node throughput; better GPU efficiency. |
| 930884 | 2 | 16 | 64 | 481.82 | 30.11 | 16.979 | 4.181 | 10.04 | Best total throughput among tested runs. |
| 930937 | 1 | 8 | 64 | 334.06 | 41.76 | 24.498 | 6.962 | 7.29 | Only +11.6% over 1-node c=32, with much worse latency; one node is near saturation. |

Batch topology interpretation:

- 2 nodes provide higher total throughput, especially at c=64, but scaling is sublinear: 16 GPUs deliver only 1.44x the throughput of the 8-GPU one-node c=64 run.
- 1 node is more GPU-efficient for these batch settings. At c=32 it reaches 37.42 tok/s/GPU, compared with 25.39 tok/s/GPU for 2 nodes. At c=64 it reaches 41.76 tok/s/GPU, compared with 30.11 tok/s/GPU for 2 nodes.
- The 2-node configuration is still useful when the goal is maximum aggregate throughput under a one-hour allocation. The 1-node configuration is preferable when the report emphasizes resource efficiency or latency.
- This gives a stronger HPC story than single-request tuning alone: throughput depends on admission control, scheduler batching, tensor/pipeline topology, and the cost of using another node.

## Hardware Topology

All hardware-focused runs used concurrency 1, 498 actual input tokens, 128 output tokens, and `MAX_MODEL_LEN=2048`. Batch scaling is intentionally not part of this comparison.

| Job ID | Setting | NCCL / hardware path | Aggregate output tok/s | Mean decode tok/s | Mean latency s | Mean TTFT s | Observation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 930559 | 1 node, 8 GPU, `TP_SIZE=8`, `PP_SIZE=1` | TP communicator stayed inside one node: `nRanks 8`, `nNodes 1`, `localRanks 8`; intra-node links used `via P2P/IPC` and `P2P/indirect` | 47.24 | 51.82 | 2.706 | 0.255 | Best single-stream latency and throughput. Removing the second node avoids pipeline overhead. |
| 930678 | 1 node, 8 GPU, `TP_SIZE=8`, `PP_SIZE=1`, `MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=2048` | Same one-node tensor-parallel layout as `930559`; this reduced scheduler/KV batching headroom for a pure single-stream run | 47.41 | 52.00 | 2.697 | 0.254 | Essentially tied with `930559`; vLLM single-stream overhead is not the main limiter. |
| 930712 | 1 node, 8 GPU, `TP_SIZE=8`, `PP_SIZE=1`, `DISTRIBUTED_EXECUTOR_BACKEND=mp` | Replaced Ray with local multiprocessing for a single-node job | 47.47 | 52.07 | 2.694 | 0.256 | Small best result: +0.5% over Ray one-node, showing runtime overhead is minor after topology is fixed. |
| 930727 | 1 node, 8 GPU, `TP_SIZE=8`, `PP_SIZE=1`, `DISTRIBUTED_EXECUTOR_BACKEND=mp`, `NCCL_PROTO=LL128` | Forced NCCL low-latency protocol | 42.81 | 46.52 | 2.988 | 0.258 | Worse than NCCL auto by 9.8%; protocol auto-selection matters. |
| 930741 | 1 node, 8 GPU, `TP_SIZE=8`, `PP_SIZE=1`, `DISTRIBUTED_EXECUTOR_BACKEND=mp`, `NCCL_ALGO=Tree` | Forced NCCL tree collectives | 47.12 | 51.68 | 2.714 | 0.257 | Slightly worse than NCCL auto; auto/Ring-like behavior is better for this workload. |
| 930423 | 2 nodes, 16 GPU, `TP_SIZE=8`, `PP_SIZE=2` | Baseline NCCL used IB and GPU Direct RDMA; TP groups were node-local (`nRanks 8`, `nNodes 1`, `localRanks 8`) with intra-node `via P2P/IPC` | 42.60 | 46.31 | 3.002 | 0.255 | Good default 16-GPU layout, but slower than single-node for one request because pipeline parallelism adds overhead. |
| 930546 | 2 nodes, 16 GPU, `TP_SIZE=8`, `PP_SIZE=2`, `NCCL_P2P_DISABLE=1` | P2P disabled; NCCL log showed `via SHM/direct/direct` inside a node and cross-node `via NET/IB/.../GDRDMA` | 29.25 | 34.79 | 4.374 | 0.718 | Disabling intra-node GPU P2P caused a large regression, showing TP collectives are sensitive to node-local GPU interconnect. |
| 930277 | 2 nodes, 16 GPU, `TP_SIZE=4`, `PP_SIZE=4` | More pipeline stages and smaller TP groups | 37.17 | 41.71 | 3.440 | 0.394 | More pipeline stages hurt this single-stream workload. |

Relative to the 2-node TP8/PP2 NCCL baseline:

- Best 1-node TP8/PP1 + mp improved aggregate throughput by 11.4%, mean decode throughput by 12.4%, and used half the GPUs.
- Per-GPU output-token efficiency improved by 122.9%: from 2.66 tok/s/GPU on the 2-node run to 5.93 tok/s/GPU on the best 1-node run.
- Restricting the 1-node run to `MAX_NUM_SEQS=1` changed throughput by only +0.4%, so the 1-node single-stream result is already close to the best simple vLLM configuration tested.
- Replacing Ray with local multiprocessing improved one-node throughput by only +0.5%, so the runtime layer is not the main bottleneck.
- Forcing NCCL LL128 reduced throughput by 9.8%, and forcing NCCL Tree reduced throughput by 0.7%; NCCL auto-selection was best among these collective settings.
- Disabling NCCL P2P reduced aggregate throughput by 31.3% and increased mean latency by 45.7%.
- TP4/PP4 reduced aggregate throughput by 12.7% and increased mean latency by 14.6%.

HPC interpretation:

- For single-request inference, more GPUs are not automatically faster. A second node introduces pipeline-parallel scheduling and communication overhead.
- On this V100 system, the useful fast path for this workload is node-local `TP=8` over GPU P2P/IPC.
- InfiniBand/GPU Direct RDMA is active and useful for cross-node paths, but this model layout keeps tensor-parallel collectives mostly node-local in the best 2-node configuration.
- A complete HPC optimization story should combine continuous batching with topology-aware parallelism and NCCL transport controls. Batching exposes throughput, while topology explains why the added GPUs do or do not scale efficiently.
- The useful positive optimization is topology-aware resource selection: using fewer GPUs on one node can be faster and much more GPU-efficient than using all allocated GPUs across two nodes for a single request.
- NCCL protocol/algorithm tuning is still valuable as a control experiment: bad forced settings show that collectives are on the critical path, while NCCL auto was already near the best tested setting.

Notes:

- Job `930083` wrote `$PROJECT_DIR/runs/930083/summary.json`.
- Job `930187` wrote `$PROJECT_DIR/runs/930187/summary.json`.
- Jobs `933544`, `933548`, and `933549` successfully ran `alpindale/c4ai-command-r-plus-GPTQ` as a non-Qwen 104B GPTQ candidate. The cache occupies about 55G under `/work/$USER/hf-cache-command-r-plus-gptq`.
- Jobs `933557`, `933580`-`933582`, `933591`-`933593`, and `933601` successfully ran `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4`; the cache occupies about 205G under `/work/$USER/hf-cache-llama31-405b-gptq`.
- vLLM logs for `930187` showed `Cannot use FlashAttention-2 backend for Volta and Turing GPUs` followed by `Using XFormers backend`.
- The large-model cache should stay under `/work`; `quota -s` showed no user quota, and the 405B GPTQ cache reached 205G successfully.
- FlashAttention-2 is not an optimization path on V100/Volta with this vLLM stack; the practical attention backend is XFormers.

## Remaining Experiment Ideas

These are optional follow-ups if more allocation time is available:

| Experiment | Purpose | Suggested settings |
| --- | --- | --- |
| `NCCL_IB_DISABLE=1` | Force TCP/socket path as a negative control for inter-node communication | `TP_SIZE=8 PP_SIZE=2 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` |
| `NCCL_NET_GDR_LEVEL=0` | Disable GPU Direct RDMA as a smaller inter-node hardware-control experiment | `TP_SIZE=8 PP_SIZE=2 NCCL_NET_GDR_LEVEL=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` |
| Longer output length | Test a more decode-heavy regime | `OUTPUT_TOKENS=256` or `512` |
| Higher batch pressure | Optional stress point beyond the observed c=64 region | `TP_SIZE=8 PP_SIZE=2 MAX_NUM_SEQS=96 MAX_NUM_BATCHED_TOKENS=24576 CONCURRENCY=96`, only if latency blow-up is acceptable |
| Non-Qwen larger model | Extend the 100B-class Command R+ batch curve | `sbatch --export=ALL,EXPERIMENT_NAME=command-r-plus-batch-c32,CONCURRENCY=32,MAX_NUM_SEQS=32,MAX_NUM_BATCHED_TOKENS=8192,RUNS=128 slurm/vllm_command_r_plus_16v100.slurm` |
| 405B longer decode | Test whether 405B throughput changes in a more decode-heavy regime | `OUTPUT_TOKENS=128 CONCURRENCY=16 RUNS=64 MAX_NUM_SEQS=16` |

Metrics to compare:

- `aggregate_output_tokens_per_s`
- mean/p50/p95 request latency
- mean/p50/p95 TTFT
- vLLM startup and model load time
- NCCL selected network path from `vllm-server.log`, `ray-head.log`, and `ray-worker-*.log`
