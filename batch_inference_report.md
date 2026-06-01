# LLM Inference Batch Optimization on Taiwania 2

## 摘要

本實驗在 Taiwania 2 的 V100 GPU 節點上測試 LLM inference throughput。主要模型是 `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`，推論框架是 vLLM `0.7.0`。實驗重點是 batch inference：在固定 2 nodes、16 x V100 的資源下，提高 concurrent requests，觀察 aggregate output token throughput 的變化。

最主要的結果是：在 70B 模型上，concurrency 從 1 提高到 64，aggregate throughput 從 `41.92 tok/s` 提升到 `481.82 tok/s`，達到 `11.49x`。代價是 mean latency 從 `3.05 s` 增加到 `16.98 s`。

另外補了一個非 Qwen 的 100B-class 對比：`alpindale/c4ai-command-r-plus-GPTQ`。這個 104B GPTQ 模型在 c=16 時達到 `201.72 tok/s`，相對 c=1 的 `21.90 tok/s` 是 `9.21x`。這表示更大的 dense model 雖然單流較慢，但仍然能明顯受益於 continuous batching。

最後也做了一個更大的 405B GPTQ INT4 對比：`hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4`。這個 checkpoint cache 約 `205G`，可以在 16 x V100 上成功載入。batch concurrency 從 1 提高到 64 時，aggregate throughput 從 `6.87 tok/s` 提升到 `103.85 tok/s`，達到 `15.12x`。

## 實驗環境

| Item | Setting |
| --- | --- |
| Cluster | Taiwania 2 |
| GPU | V100-SXM2-32GB |
| Main allocation | 2 nodes, 16 GPUs, 1 hour |
| Framework | vLLM `0.7.0` |
| Quantization | GPTQ |
| Main model | `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` |
| Comparison model | `alpindale/c4ai-command-r-plus-GPTQ` |
| Large batch model | `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4` |
| Parallelism | `TP_SIZE=8`, `PP_SIZE=2` |
| Input / output | about 500 input tokens, 128 output tokens |

## 70B Batch Scaling

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

Batching 大幅提升總吞吐量，但會增加等待時間。c=64 是最高 throughput；c=48 則比較像 throughput 和 latency 的折衷點。從 c=48 到 c=64，throughput 只增加 `6.2%`，但 latency 增加 `25.6%`。

## 1-Node vs 2-Node

同樣做 batch inference 時，2-node 有最高總 throughput，但 1-node 的 per-GPU efficiency 較好。

| Job ID | Nodes | GPUs | Concurrency | Aggregate tok/s | Tok/s/GPU | Mean latency s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 930833 | 2 | 16 | 32 | 406.27 | 25.39 | 10.068 |
| 930912 | 1 | 8 | 32 | 299.35 | 37.42 | 13.666 |
| 930884 | 2 | 16 | 64 | 481.82 | 30.11 | 16.979 |
| 930937 | 1 | 8 | 64 | 334.06 | 41.76 | 24.498 |

同樣 c=32 時，1-node 只用一半 GPU，但達到 2-node throughput 的 `73.7%`。這代表多一個 node 可以提高總吞吐量，但跨節點和平行化成本會讓 scaling 不完全線性。

## 104B Model Comparison

為了加入非 Qwen 的大模型對比，測試了 `alpindale/c4ai-command-r-plus-GPTQ`。這是 104B dense GPTQ 模型，cache 約 `55G`。

| Job ID | Model | Concurrency | Requests | Aggregate tok/s | Speedup | Mean latency s | Mean TTFT s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 933548 | Command R+ 104B GPTQ | 1 | 8 | 21.90 | 1.00x | 5.840 | 0.338 | 23.08 |
| 933549 | Command R+ 104B GPTQ | 16 | 64 | 201.72 | 9.21x | 10.138 | 1.604 | 14.93 |

Command R+ 104B 比 72B baseline 慢，但 batch scaling 仍然明顯。c=16 時，104B 模型的 throughput 約為 72B 模型 c=16 的 `62.5%`。這個結果可以用來說明：模型變大會降低單位 token 速度，但 continuous batching 仍然是有效的 HPC inference 優化。

## 405B Batch Throughput

另外測試 `hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4`，確認 200GB 級 checkpoint 在 16 x V100 上的 batch throughput。這組使用約 125 input tokens、32 output tokens，cache 約 `205G`。

| Job ID | Concurrency | Requests | Aggregate tok/s | Speedup | Mean latency s | Mean TTFT s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 933580 | 1 | 4 | 6.87 | 1.00x | 4.655 | 0.569 | 7.59 |
| 933581 | 2 | 8 | 12.12 | 1.77x | 5.249 | 0.900 | 7.15 |
| 933582 | 4 | 16 | 22.00 | 3.20x | 5.785 | 1.127 | 6.69 |
| 933591 | 8 | 32 | 39.50 | 5.75x | 6.464 | 1.357 | 6.11 |
| 933592 | 16 | 64 | 60.46 | 8.80x | 8.450 | 1.975 | 4.83 |
| 933593 | 32 | 128 | 77.90 | 11.34x | 13.114 | 3.266 | 3.18 |
| 933601 | 64 | 256 | 103.85 | 15.12x | 19.678 | 5.717 | 2.25 |

405B 的單一 request 很慢，但 batch scaling 很明顯。c=64 時總吞吐量達到 `103.85 tok/s`，是 c=1 的 `15.12x`；代價是 mean latency 從 `4.66 s` 增加到 `19.68 s`。

## 結論

本實驗最主要的優化是 continuous batching。對 70B GPTQ 模型，batch concurrency 從 1 增加到 64，使 aggregate throughput 從 `41.92 tok/s` 提升到 `481.82 tok/s`，達到 `11.49x`。

HPC 角度下，重點不是單一 request 最快，而是在固定 GPU allocation 內處理最多 token。2-node 設定提供最高總吞吐量；1-node 設定提供較好的 per-GPU efficiency。對更大的 104B 和 405B 模型，batching 仍然帶來明顯 throughput gain，證明這個方法不只適用於單一 Qwen 模型。

405B GPTQ INT4 也補上容量面結論：在 `/work` cache 約 `205G` 的情況下，16 x V100 可以完成載入、短輸出和 batch throughput 測試。

建議報告圖表：

1. Concurrency vs aggregate output tok/s
2. Concurrency vs mean latency
3. 1-node vs 2-node tok/s/GPU
4. 70B vs 104B batch throughput comparison
5. 405B concurrency vs aggregate output tok/s
