# Cooled RTX Spark N1X comparison · November 24, 2026

The published Windows Arm64 ZIP reached **92.76 tok/s** single-stream and **112.10 tok/s** across four streams. The original Basalt Arm64 port reached **84.05** and **85.23 tok/s** on the same laptop. Both engines completed all 392 measured BetterBench requests with zero reported cache hits. [Release](https://github.com/SuperDistance/basalt-rtx-spark/releases/tag/v0.4.1-rtx-spark.2) · [Binary hashes and native smoke test](release-package.json)

| BetterBench metric | Release | Arm64 port | Change versus port |
| --- | ---: | ---: | ---: |
| Weighted single-stream decode | 92.76 tok/s | 84.05 tok/s | +10.4% |
| Short-prompt median time to first token | 280 ms | 619 ms | −54.8% |
| Four-stream aggregate throughput | 112.10 tok/s | 85.23 tok/s | +31.5% |
| Four-stream median time to first token | 396 ms | 747 ms | −46.9% |
| Cold prefill at nominal 2K input | 1,767 tok/s | 967 tok/s | +82.7% |

## Single-stream decode

Each engine saw the same eight categories in the same order, with 20 measured requests per category. BetterBench weights six category medians. Weights, including the two zeros, come from the [run configuration](release/betterbench-config.json). The release was faster in all eight categories.

| Category | Weight | Release tok/s | Arm64 port tok/s |
| --- | ---: | ---: | ---: |
| Chat | 0% | 83.80 | 71.84 |
| Code | 30% | 96.18 | 88.05 |
| File edit | 10% | 99.29 | 83.88 |
| Json | 15% | 113.04 | 111.43 |
| Math | 0% | 106.86 | 103.09 |
| Prose | 15% | 77.34 | 67.81 |
| Reasoning | 20% | 75.93 | 69.12 |
| Summarization | 10% | 102.40 | 85.35 |

JSON, math and summarization exceeded 100 tok/s on the release. Both engines already benefit from speculative decoding on structured text. On JSON they produced a median 4.25 and 4.45 tokens per window respectively. The release produced 2.30 on prose and 2.50 on reasoning. Those two categories carry 35% of the headline score. These are sampled responses, and the suite uses no fixed generation seed. [Release speculation measurements](release/speculation.json) · [Port measurements](arm-port/speculation.json)

## Cooling and power

An HP OmniBook X with 128 GiB of unified memory and a 96 GiB GPU allocation ran the release first, then the port. It used Windows 11 Arm64, NVIDIA driver 621.24, AC power and the HP Performance thermal profile. The owner’s physical cooling setup was in place before either full run. A 60-second idle window preceded each test. The GPU median was 47.5°C for the release and 46°C for the port. Both engines used UD-IQ3_XXS, INT8 KV, four 262,144-token slots and MTP 8. Both allocated all 24,576 GPU expert slots. Automatic host expert backing differed (about 3.0 GB and 4.0 GB respectively).

| Busy measurements | Release | Arm64 port |
| --- | ---: | ---: |
| Decode GPU / CPU draw | 47.50 / 3.84 W | 49.20 / 4.89 W |
| Decode CPU + GPU draw | 51.34 W | 54.09 W |
| Decode median SM clock | 2,320 MHz | 2,366 MHz |
| C4 CPU + GPU draw | 55.62 W | 52.77 W |

Busy samples had at least 80% GPU utilization. Component power is measured by SparkHelper. The firmware power limit and fan RPM were unavailable. The ZIP upload finished 16.3 seconds into the opening chat category. The full runs were sequential with stochastic sampling and one run per engine. [Release telemetry](release/power.json) · [Port telemetry](arm-port/power.json) · [Release provenance](release/release-provenance.json)

A separate cooled decode replay of the previously measured production executable completed 160/160 requests at **93.71 tok/s** and **53.75 W CPU + GPU**. Physical cooling changed during its opening category. Earlier complete runs recorded **97.88 tok/s** on November 23 and **90.79 tok/s** at **45.28 W CPU + GPU** on the morning of November 24. Their decode-phase GPU power was 42.42 W and 35.83 W. Code prompts accounted for 4.42 tok/s of the 7.09 tok/s cross-day gap. [Cooled replay](production-decode/summary.json) · [Historical phase comparison](../2026-11-24-n1x-verified-default/power-method.md#why-the-single-stream-score-changed-between-runs)

## Serving behavior under load

In separate controls with pinned expert placement, four simultaneous 1,024-token structured replies reached **165.51 tok/s on the release** and **110.66 tok/s on the port**. The release needed 367 verification windows to the port's 522, and drafting took 5.42 versus 21.53 ms per window. It committed 2.88 versus 1.99 tokens per slot per window. Busy GPU draw was 62.06 W and 58.02 W. This fixture exercises the batched drafter directly. [Release native timings](release/structured-c4.json) · [Port native timings](arm-port/structured-c4.json)

A cold 16,672-token fourth prompt reached its first token in **26.17 s on the release** and **61.36 s on the port**. All four requests completed on each engine. Release incumbents continued generating, with a worst gap of 0.389 s. The port kept incumbent gaps down to 0.126 s but finished all three replies before the newcomer emitted a token, so it missed the fixture's overlap-at-first-token criterion. [Release cold join](release/cold-join.json) · [Port cold join](arm-port/cold-join.json)

In the reverse test, a short request interrupted a cold 49,966-token read. Both engines yielded and resumed at token 4,096. The short request reached its first token in **1.95 s on the release** and **4.01 s on the port**. Its greedy output matched on both engines. The long replies differed (30 versus 32 output tokens), so this fixture establishes scheduling behavior only. [Release arrival record](release/short-during-prefill.json) · [Port arrival record](arm-port/short-during-prefill.json)

## Reproduction and raw results

Use BetterBench 0.6.1 with the archived preset. The full runs each included 160 single-stream requests, 40 cold-prefill requests and 192 mixed requests at concurrency one through four. Both servers also answered the same greedy arithmetic smoke prompt with `7`. [Benchmark instructions](../../README.md#running-it) · [Release raw results](release/results.json) · [Arm64 port raw results](arm-port/results.json)
