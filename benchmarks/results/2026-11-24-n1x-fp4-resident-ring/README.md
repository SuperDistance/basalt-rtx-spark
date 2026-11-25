# FP4 resident staging ring on RTX Spark N1X

At the nominal 32K cold input, Basalt processed **1,951 tok/s** with a median **12.08 s** to first token. The preceding FP4-Local release run measured 1,961 tok/s and 12.06 s at the same input depth. The new binary completed 392/392 measured requests with zero reported cache hits. [Per-request results](results.json) · [Previous FP4 run](../2026-11-24-n1x-fp4-local)

| Same FP4-Local pack and BetterBench preset | Resident ring | Prior release | Change |
| --- | ---: | ---: | ---: |
| Weighted single-stream decode (tok/s) | 85.28 | 84.41 | +1.0% |
| Median short-prompt TTFT (ms) | 364.74 | 369.49 | -1.3% |
| Mixed C4 aggregate (tok/s) | 95.88 | 94.13 | +1.9% |
| Decode busy CPU + GPU (W) | 52.45 | 49.32 | +6.3% |
| Mixed C4 busy CPU + GPU (W) | 50.21 | 49.66 | +1.1% |

## Cold prefill

| Nominal input | Median actual tokens | Resident ring tok/s | Release tok/s | Change | Median TTFT (s) | Busy GPU W, ring / release |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1,541.0 | 1,583 | 1,606 | -1.4% | 0.98 | 40.46 / 50.53 |
| 8K | 5,946.5 | 2,076 | 2,080 | -0.2% | 2.88 | 40.69 / 45.95 |
| 16K | 11,804.5 | 2,258 | 2,303 | -2.0% | 5.23 | 39.16 / 39.71 |
| 32K | 23,626.0 | 1,951 | 1,961 | -0.5% | 12.08 | 32.71 / 34.89 |
| 64K | 47,189.5 | 2,353 | 2,522 | -6.7% | 20.04 | 47.03 / 54.14 |

The engine reduced the staging ring once, from 256 slots to 64, on a 15,360-token chunk of a 47,104-token cold request. At 23K–24.5K native rows, 3 chunks staged 1507 blobs, 7 chunks staged 1528 blobs. This includes warmups. The automatic run retained the file-staging fallback at this depth. Its change in throughput cannot be attributed to the reduced ring. The reduced ring allocates prompt scratch separately when the full expert cache and original chunk can fit with fewer staging slots. The targeted alternating experiment used pinned placement, fixed tokens and a fixed memory reserve. [Effective settings and native trace](run-config.json) · [Controlled experiment and row parity](../2026-11-24-n1x-fp4-prefill-chunks)

## Serving and measurement conditions

The HP OmniBook X had 128 GiB of unified memory, a 96 GiB GPU allocation and NVIDIA driver 621.24. Both FP4 runs started with all 24,576 GPU expert slots, four 262,144-token contexts, INT8 KV and MTP 8. Automatic host backing was 3.26 GiB in the new run and 3.11 GiB previously. Both used the Windows file cache for the n-gram table because the optional 50.7 GiB direct-read sidecar exceeded free space. BetterBench 0.6.1 used the same config, corpus hash and single-stream prompt IDs as the earlier FP4 run. The full suites ran sequentially with stochastic generation. [Configuration](betterbench-config.json) · [Per-workload summary](summary.json)

Windows used the Balanced base power scheme with the HP Performance thermal profile. The 60-second idle baseline had a 48.0°C median GPU temperature. Power-source samples during the run were 1,061 on AC. At ≥80% GPU utilization, single-stream decode averaged 46.00 W GPU at a median 2,288 MHz and 73.1°C mean. Mixed C4 averaged 46.62 W GPU at a median 2,281 MHz and 73.0°C mean. The driver did not expose a wattage limit. [Power by phase](power.json) · [Recorded rails](spark.csv) · [Cooldown](precondition.json)

An earlier automatic attempt allocated 4.61 GiB of host expert backing. After 115 successful single-stream measurements, external memory pressure brought physical headroom below the 2 GiB guard. Three consecutive prose requests returned no tokens and dispatch stopped during a fourth. That attempt reached neither cold prefill nor concurrency. [Partial attempt and native guard readings](guard-attempt.json)

## Arrival controls and output identity

The cold 16,672-token fourth prompt reached its first token in **40.17 s**. Its three incumbents generated 483, 479, 426 tokens during the join. Their largest delivery gap was **0.267 s**. The native trace counted 173 overlapping windows and 66 chunk jobs. Busy GPU draw averaged 45.26 W over the fixture. The cold join check passed. [Cold join](cold-join.json)

With the reverse arrival, the short request reached its first token in **1.70 s** during a 49,966-token cold read. The long request yielded at 4,096 tokens. The reverse-arrival check passed with 56.32 W mean busy GPU draw. Both controls pinned 24,576 expert slots and 1 GiB of host backing. [Short arrival](short-during-prefill.json)

All **900/900 native greedy output IDs** matched the released executable across conversation, structured output and prose. Each stock fixture generated 300 tokens at temperature zero with pinned expert placement and MTP 8. [Token identity](greedy-identity.json)
