# Flash-Next NVFP4-DQ6_K on RTX Spark N1X

Basalt reached **85.22 tok/s** weighted single-stream decode and **95.84 tok/s** mixed four-stream throughput on the 128 GiB laptop. The median short-prompt TTFT was **365 ms**. All 392 measured requests completed with zero reported cache hits. [Raw result](results.json) · [Summary](summary.json)

| BetterBench on the same laptop | NVFP4-DQ6_K | FP4-Local | UD-IQ3_XXS |
| --- | ---: | ---: | ---: |
| Weighted single-stream decode (tok/s) | 85.22 | 85.28 | 92.76 |
| Median short-prompt TTFT (ms) | 365.07 | 364.74 | 280.06 |
| Mixed C4 aggregate output (tok/s) | 95.84 | 95.88 | 112.10 |
| Mixed C4 median TTFT (ms) | 516.97 | 512.06 | 396.47 |
| Nominal 2K cold prefill (tok/s) | 1,598.56 | 1,583.07 | 1,767.44 |
| Nominal 64K cold prefill (tok/s) | 2,438.52 | 2,352.92 | 2,288.18 |

The DQ6_K and FP4-Local measurements used the same native binary. The IQ3 measurement used release v0.4.1-rtx-spark.2. Each used BetterBench 0.6.1, the same preset and corpus hash, 160 single-stream prompts, 40 cold-prefill measurements and 192 mixed requests. Temperature was 0.7 without a fixed generation seed. Runs were sequential with different power and memory conditions. BetterBench measures serving performance. [FP4-Local](../2026-11-24-n1x-fp4-resident-ring) · [IQ3 release](../2026-11-24-n1x-cooled-release)

## Placement, cold prompts and clocks

The exact published DQ6_K pack was 131,681,979,197 bytes (122.64 GiB) with SHA-256 `177f9b70c63ecd7efd17fc705ad85a8c8253d59e4f04dd10a001b556d15fad33`. Windows 11 Arm64 used driver 621.24 with a 96 GiB GPU allocation on the HP OmniBook X. The server selected four 262,144-token contexts, INT8 KV and MTP 8. It backed 4.38 GiB of experts in host memory. NVIDIA's whole-GPU memory reading peaked at 76,830 MiB (75.03 GiB) across 2,092 samples during the timed suite, including other GPU processes. Native allocation was `basalt generate: expert cache 24576 slots, 63.29 GiB of VRAM; policy is`. The engine read the n-gram table through the Windows file cache. [Pack provenance](model-provenance.json) · [Native settings and trace](run-config.json)

| Nominal input | Median actual tokens | Prefill (tok/s) | Median TTFT (s) |
| ---: | ---: | ---: | ---: |
| 2K | 1,562.5 | 1,599 | 0.98 |
| 8K | 5,951.0 | 2,149 | 2.77 |
| 16K | 11,827.0 | 2,351 | 5.02 |
| 32K | 23,633.0 | 2,037 | 11.56 |
| 64K | 47,091.5 | 2,439 | 19.30 |

At the nominal 32K depth, DQ6_K staged 1,072–1,093 expert blobs per native chunk and FP4-Local staged 1,507–1,528. Automatic host backing differed (4.38 versus 3.26 GiB), which affects how many expert blobs need file staging.

Mixed throughput at C1 / C2 / C3 / C4 was 77.43 / 85.86 / 90.98 / 95.84 tok/s. Windows used the Balanced base scheme and HP Performance thermal profile. The selected rail samples had ≥80% GPU utilization. Single-stream decode averaged 48.39 W GPU and 52.97 W CPU + GPU, at median 2,314 MHz and mean 73.8°C GPU temperature. Mixed C4 averaged 50.50 W GPU and 55.21 W CPU + GPU. Whole-run power-source samples were 1,047 on AC. The driver did not expose a power limit. [Power and clocks](power.json) · [Recorded rails](spark.csv)

## Arrival controls

The 16,672-token cold fourth prompt reached its first token in **41.79 s** with a maximum incumbent delivery gap of **0.259 s**. The engine recorded 178 overlap windows and 66 chunk jobs. Busy GPU draw averaged 46.73 W. The cold join check passed. [Cold join](cold-join.json)

In the reverse test, a short request arrived during a 49,966-token cold read and reached its first token in **1.77 s**. The long request yielded at 4,096 tokens. The reverse-arrival check passed with 54.46 W mean busy GPU draw. Both controls pinned 24,576 expert slots and 1 GiB of host backing. [Short arrival](short-during-prefill.json)
