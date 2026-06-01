# Model Scaling Summary

## Model

這份筆記整理 104B 和 405B 模型的 batch throughput 結果，用來補強 72B baseline 之外的模型規模比較。

| Item | Setting |
| --- | --- |
| Model family | Cohere Command R+ |
| Size | 104B |
| Quantization | GPTQ |
| Cache size | about 55G |
| Hardware | 2 nodes, 16 x V100 |
| Parallelism | `TP_SIZE=8`, `PP_SIZE=2` |

## Result

| Job ID | Model | Concurrency | Output tokens | Requests | Aggregate tok/s | Mean latency s | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 933548 | Command R+ 104B GPTQ | 1 | 128 | 8 | 21.90 | 5.840 | 23.08 |
| 933549 | Command R+ 104B GPTQ | 16 | 128 | 64 | 201.72 | 10.138 | 14.93 |
| 933580 | Llama 3.1 405B GPTQ INT4 | 1 | 32 | 4 | 6.87 | 4.655 | 7.59 |
| 933592 | Llama 3.1 405B GPTQ INT4 | 16 | 32 | 64 | 60.46 | 8.450 | 4.83 |
| 933601 | Llama 3.1 405B GPTQ INT4 | 64 | 32 | 256 | 103.85 | 19.678 | 2.25 |

c=16 相對 c=1 的 throughput speedup 是 `9.21x`。

Llama 3.1 405B GPTQ INT4 的 cache 約 `205G`。它不只成功載入，也完成 batch throughput：c=64 達到 `103.85 tok/s`，相對 c=1 是 `15.12x`。

## Interpretation

Command R+ 104B 比 72B baseline 慢，這符合模型較大的預期。不過 batching 仍然有效，代表這個 HPC optimization 不是只對單一模型規模有效。

405B GPTQ INT4 更慢，但 batch scaling 更明顯：c=1 只有 `6.87 tok/s`，c=64 提升到 `103.85 tok/s`。這可以作為報告裡的「容量 + batching」示範。

和 72B baseline 比較：

| Setting | 72B model tok/s | 104B model tok/s | Ratio |
| --- | ---: | ---: | ---: |
| c=1 | 41.92 | 21.90 | 52.3% |
| c=16 | 322.95 | 201.72 | 62.5% |

## Next Run

如果還要補一個點，建議在已 cache 的模型上跑 Command R+ c=32，讓 104B 對比曲線更完整：

```bash
sbatch --export=ALL,EXPERIMENT_NAME=command-r-plus-batch-c32,OUTPUT_TOKENS=128,WARMUP=2,RUNS=128,CONCURRENCY=32,MAX_NUM_SEQS=32,MAX_NUM_BATCHED_TOKENS=8192 slurm/vllm_command_r_plus_16v100.slurm
```
