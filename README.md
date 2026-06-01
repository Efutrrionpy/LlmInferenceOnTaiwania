# LLM Inference Batch Optimization on Taiwania 2

這個 repository 記錄在 Taiwania 2 V100 GPU 節點上做 LLM inference throughput optimization 的實驗。核心問題是：在一次最長一小時、最多 2 nodes / 16 x V100 的 HPC allocation 內，怎麼提高大型語言模型的總輸出 token throughput。

本實驗使用 vLLM `0.7.0`。因為 V100 是 Volta 架構，compute capability 是 `7.0`，新版 vLLM 對 GPU 架構支援較不適合，因此固定使用仍支援 V100 的版本。

## TL;DR

- 主要有效的優化是 continuous batching，也就是同時讓多個 request 進入 vLLM scheduler。
- 72B GPTQ 模型在 16 x V100 上，concurrency 從 `1` 提高到 `64`，aggregate throughput 從 `41.92 tok/s` 提升到 `481.82 tok/s`，為 `11.49x`。
- 104B GPTQ 模型在 concurrency `16` 時達到 `201.72 tok/s`，相對 c=1 是 `9.21x`。
- 405B GPTQ INT4 模型 cache 約 `205G`，可以在 16 x V100 上成功載入；concurrency `64` 時達到 `103.85 tok/s`，相對 c=1 是 `15.12x`。
- 2 nodes 有最高總吞吐量，但 1 node 有更好的 per-GPU efficiency。這表示多用 GPU 不一定線性變快，跨節點與 pipeline parallelism 會帶來額外成本。
- FlashAttention-2 不是這組 V100/vLLM stack 的可行優化路徑，實際 backend 會 fallback 到 XFormers。

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

- `Concurrency`: 同時進入系統的 in-flight requests 數量。
- `Requests`: 該次 benchmark 實際測量的總 request 數量。
- `Aggregate tok/s`: 整個服務在測試期間每秒輸出的總 token 數，是本實驗最重要的 throughput 指標。
- `Decode tok/s`: 單一 request 在拿到第一個 token 之後的 decoding 速度。batch 變大時，aggregate tok/s 會上升，但每個 request 的 decode tok/s 通常會下降。

## Reproduce

提交前先把 Slurm 檔案裡的 `YOUR_ACCOUNT` 換成自己的 project allocation account。若使用 gated model，例如 Llama，提交前也要先設定 `HF_TOKEN`。

先在 compute node 上建立環境，不要在 login node 做安裝或推論：

```bash
cd /work/$USER/llm
sbatch slurm/setup_vllm_env.slurm
```

提交 72B baseline：

```bash
sbatch slurm/vllm_70b_16v100.slurm
```

提交 104B 和 405B：

```bash
sbatch slurm/vllm_command_r_plus_16v100.slurm
sbatch slurm/vllm_llama31_405b_gptq_16v100.slurm
```

Batch throughput example：

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

每個 job 會輸出到：

```text
runs/<slurm-job-id>/
```

其中 `summary.json` 是主要統計結果，`experiment.env` 記錄 Slurm、模型、parallelism 和 benchmark 參數，`vllm-server.log` 可確認 backend、NCCL 和模型載入狀態。

## 72B Batch Throughput

2-node batch scaling 使用 16 x V100，`TP_SIZE=8`、`PP_SIZE=2`。

| Job ID | Concurrency | Requests | Aggregate tok/s | Speedup | Mean latency s | Mean TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 930301 | 1 | 8 | 41.92 | 1.00x | 3.050 | 0.259 | 45.56 |
| 930314 | 4 | 16 | 121.50 | 2.90x | 4.205 | 0.480 | 34.14 |
| 930321 | 8 | 32 | 219.81 | 5.24x | 4.651 | 0.648 | 31.81 |
| 930349 | 16 | 64 | 322.95 | 7.70x | 6.334 | 1.210 | 24.91 |
| 930811 | 24 | 96 | 350.08 | 8.35x | 8.761 | 1.668 | 18.00 |
| 930833 | 32 | 128 | 406.27 | 9.69x | 10.068 | 2.150 | 16.14 |
| 930859 | 48 | 192 | 453.71 | 10.82x | 13.521 | 3.197 | 12.43 |
| 930884 | 64 | 256 | 481.82 | 11.49x | 16.979 | 4.181 | 10.04 |

結果顯示 batching 可以明顯提升系統總吞吐量，但代價是 latency 上升。c=64 是最高 throughput；c=48 比較像 throughput 和 latency 的折衷點。從 c=48 到 c=64，throughput 只增加約 `6.2%`，但 mean latency 增加約 `25.6%`。

## 1-Node vs 2-Node

同樣做 batch inference 時，2 nodes 有最高總 throughput，但 1 node 的 per-GPU efficiency 較好。

| Job ID | Nodes | GPUs | Concurrency | Aggregate tok/s | Tok/s/GPU | Mean latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 930833 | 2 | 16 | 32 | 406.27 | 25.39 | 10.068 |
| 930912 | 1 | 8 | 32 | 299.35 | 37.42 | 13.666 |
| 930884 | 2 | 16 | 64 | 481.82 | 30.11 | 16.979 |
| 930937 | 1 | 8 | 64 | 334.06 | 41.76 | 24.498 |

同樣 c=32 時，1 node 只用一半 GPU，但達到 2-node throughput 的 `73.7%`。這說明跨節點 parallelism 可以提高總吞吐量，但 scaling 不會完全線性。

## Larger Models

104B 和 405B 測試用來確認 batch throughput 是否只對某個模型有效。結果顯示模型變大後單流速度變慢，但 continuous batching 仍然帶來明顯提升。

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

405B 的結果也回答了容量問題：理論上 16 x V100 有 512GB GPU memory，而 GPTQ INT4 checkpoint 約 205G cache，在這個 setup 下可以成功載入並完成 batch benchmark。

## Hardware Notes

除了 batch，本實驗也做過幾個偏硬體與 topology 的控制實驗：

| Setting | Aggregate tok/s | Mean latency s | Observation |
| --- | ---: | ---: | --- |
| 1 node, `TP=8`, `PP=1`, multiprocessing | 47.47 | 2.694 | 最好的 single-request 設定，避免跨節點 pipeline overhead |
| 2 nodes, `TP=8`, `PP=2` | 42.60 | 3.002 | 預設 16-GPU layout，總資源較多但單流較慢 |
| 2 nodes, `TP=4`, `PP=4` | 37.17 | 3.440 | 更多 pipeline stages 對 single request 不利 |
| 2 nodes, `NCCL_P2P_DISABLE=1` | 29.25 | 4.374 | 關閉 intra-node GPU P2P 後明顯變慢 |
| 1 node, forced `NCCL_PROTO=LL128` | 42.81 | 2.988 | 比 NCCL auto 慢 |
| 1 node, forced `NCCL_ALGO=Tree` | 47.12 | 2.714 | 接近 auto，但仍略慢 |

這些結果說明：對 single-request inference 而言，更多 GPU 不一定更快。比較好的 HPC 故事是把 continuous batching 和 topology-aware parallelism 放在一起看：batching 提高系統吞吐量，topology 解釋為什麼新增節點後效率不會線性提升。

## Conclusion

本實驗最重要的優化是 continuous batching。在固定 16 x V100 的條件下，72B GPTQ 模型從 c=1 到 c=64 得到 `11.49x` aggregate throughput gain。104B 和 405B 模型也有同樣趨勢，代表這不是單一模型的特殊現象。

從 HPC 角度看，最佳化目標不是讓單一 request 最快，而是在固定 allocation 內處理最多 output tokens。若目標是最高總 throughput，2 nodes 較好；若目標是 GPU 使用效率，1 node 反而更有優勢。V100 上 FlashAttention-2 不可用，因此本實驗中真正有效且可展示的方向是 vLLM continuous batching、parallelism topology 和 NCCL/P2P 控制實驗。
