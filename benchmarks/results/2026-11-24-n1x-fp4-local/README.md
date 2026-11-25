# Flash-Next FP4-Local on Basalt RTX Spark

The Windows Arm64 release delivered **84.41 tok/s** weighted single-stream decode and **94.13 tok/s** aggregate throughput at four streams on the 128 GiB RTX Spark N1X. Short-prompt median time to first token was **369 ms**. All 392 measured requests completed with zero reported cache hits. [Per-request timing records](results.json) · [BetterBench summary](summary.json)

| Same laptop and BetterBench preset | Basalt FP4-Local | Basalt UD-IQ3_XXS | Native vLLM FP4-Local |
| --- | ---: | ---: | ---: |
| Weighted single-stream decode (tok/s) | 84.41 | 92.76 | 54.19 |
| Short-prompt median TTFT (ms) | 369 | 280 | 271 |
| Four-stream mixed throughput (tok/s) | 94.13 | 112.10 | 100.75 |
| Four-stream median TTFT (ms) | 525 | 396 | 455 |

All three runs used BetterBench 0.6.1 with the same corpus hash, 160 measured single-stream prompt IDs in the same order, 40 cold-prefill requests across five depths, and 48 mixed requests at each of C1 through C4. The preset adds a unique nonce per request and uses temperature 0.7 with no fixed generation seed. Basalt ran both packs on the same release executable. Native vLLM for RTX Spark 0.2.0 used the FP4-Local checkpoint with FP8 target KV, BF16 draft KV and MTP 7. The three measurements came from separate runs at different power levels. [Preset](betterbench-config.json) · [Basalt IQ3 result](../2026-11-24-n1x-cooled-release) · [Native vLLM result](../2026-11-24-n1x-verified-default/vllm)

## Cold prompts and mixed concurrency

| Nominal input | Median actual input tokens | Median prefill (tok/s) | Median TTFT (s) |
| ---: | ---: | ---: | ---: |
| 2K | 1,560.5 | 1,606 | 0.98 |
| 8K | 5,975.5 | 2,080 | 2.87 |
| 16K | 11,839.5 | 2,303 | 5.15 |
| 32K | 23,604.5 | 1,961 | 12.06 |
| 64K | 47,087.0 | 2,522 | 18.69 |

| Concurrent requests | Aggregate output (tok/s) | Median TTFT (ms) |
| ---: | ---: | ---: |
| 1 | 78.18 | 368 |
| 2 | 86.55 | 478 |
| 3 | 91.32 | 504 |
| 4 | 94.13 | 525 |

## Model placement and power

The released Arm64 binary used the `Qwen3.8-FN-FP4-Local.basalt` conversion (122.64 GiB). At startup, the 96 GiB GPU allocation held all 24,576 expert slots with 3.11 GiB of host expert backing. The nominal 32K input ran slower per token than the 64K input. At roughly 23.5K actual prompt tokens, the engine staged expert blobs and used zero CPU experts. At roughly 47K actual tokens it used 15,360-token chunks with buffers outside the full expert cache. Logged full chunks staged zero blobs. The server selected INT8 KV, MTP 8 and four slots with 262,144 tokens each. The auxiliary n-gram table used the Windows file cache because only 8.0 GiB of disk space was free for a 50.7 GiB direct-read sidecar. Cold prompt timings include that choice. [Effective settings and binary hash](run-config.json) · [Host memory](host-memory.json) · [Release download](https://github.com/SuperDistance/basalt-rtx-spark/releases/tag/v0.4.1-rtx-spark.2)

The HP OmniBook X ran Windows 11 Arm64, NVIDIA 621.24, the Balanced Windows base scheme and the HP Performance thermal profile. After model load, the 60-second idle window had a median GPU temperature of 48°C and SoC temperature of 52.9°C. SparkHelper's power-source samples during the timed run were 10 on battery (15:37:09 to 15:37:27), then 1,046 on AC (15:37:29 to 16:12:27), local time. The switch to AC came during the opening chat workload. The driver reported no wattage limit or fan RPM. [Cooldown](precondition.json) · [Timed rail samples](spark.csv) · [NVIDIA samples](gpu.csv)

| Busy phase | Mean GPU / CPU (W) | Mean CPU + GPU (W) | Median SM clock (MHz) | Mean GPU temperature (°C) |
| --- | ---: | ---: | ---: | ---: |
| Single-stream decode | 44.55 / 4.78 | 49.32 | 2,275 | 72.3 |
| Cold prefill | 46.42 / 8.94 | 55.36 | 1,982 | 70.9 |
| Mixed C4 | 46.04 / 3.62 | 49.66 | 2,275 | 72.2 |

Busy samples have at least 80% GPU utilization. The earlier vLLM run averaged 35.02 W GPU draw in single-stream decode and 34.46 W at C4. The rail averages describe the observed operating points. [Per-workload power](power.json) · [vLLM power](../2026-11-24-n1x-verified-default/vllm/sparkhelper-power-by-phase.json)

## Serving while another request is active

Both arrival checks passed on fresh FP4-Local servers with four full-context slots, MTP 8, INT8 KV, 24,576 GPU expert slots and a pinned 1 GiB host expert budget. These checks run separately from the 392 BetterBench requests.

- A fourth request with **16,672 cold tokens** received its first token in **42.34 s**. The three incumbent chats emitted **430–486 tokens each** during the read. The longest gap between their tokens was **279 ms**. The engine logged 175 overlapped windows and 66 chunk jobs. Mean GPU draw during the join was 34.37 W across 21 samples. [Arrival trace and power](cold-join.json)
- A short request arrived 1,450 tokens into a **49,966-token cold prefill**. The long read yielded at token 4,096, the short request reached its first token in **1.68 s**, and its 16-token reply completed before the long request's first token. The long request resumed at token 4,096 and reached its first token in **73.24 s**. [Yield and resume trace](short-during-prefill.json) · [Fixture and reproduction](../../feature-coverage.md)
