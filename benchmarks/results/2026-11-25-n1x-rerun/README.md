# Basalt RTX Spark at full GPU power · November 25, 2026

These runs refresh the Basalt rows of the [results table](../../../README.md#results). They use the same RTX Spark N1X
laptop (128 GB, 96 GB GPU split, physical cooling) and the same BetterBench 0.6.1 preset as the
[November 24 runs](../2026-11-24-n1x-cooled-release). The difference is power. On November 24 the GPU averaged
46–49 W in busy decode. Here every kept piece ran in P0 at about 54 W.

| Run | Build | Pack | Weighted decode (tok/s) | 1 / 2 / 3 / 4 chats (tok/s) | Median first token | Prompt read at 2K / 8K / 16K / 32K / 64K (tok/s) | Busy decode GPU W |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [merged-iq3](merged-iq3) | source build `209a14bb` | `UD-IQ3_XXS` | **98.92** | 92.3 / 96.2 / 115.2 / **120.6** | **266 ms** | **1,929** / 2,262 / 2,423 / 2,385 / 2,431 | 54.6 |
| [release-iq3](release-iq3) | release ZIP v0.4.1-rtx-spark.2 | `UD-IQ3_XXS` | 97.10 | 91.2 / 103.9 / 116.0 / 119.9 | 270 ms | 1,883 / **2,272** / 2,386 / 2,438 / 2,429 | 54.3 |
| [merged-fp4-local](merged-fp4-local) | source build `209a14bb` | NVIDIA `FP4-Local` | 91.85 | 83.1 / 91.2 / 98.6 / 101.0 | 346 ms | 1,741 / 2,241 / 2,521 / 2,607 / 2,655 | 53.2 |
| [merged-dq6k](merged-dq6k) | source build `209a14bb` | `NVFP4-DQ6_K` | 87.94 | 82.9 / 89.9 / 99.4 / 100.3 | 352 ms | 1,716 / 2,251 / **2,524** / **2,609** / **2,657** | 53.7 |

Source build `209a14bb` is the engine on `main`. `main` adds only lint changes to it (a shared fp16 helper that
compiles to the same instruction on Arm64 MSVC, and an import move in `serve`). All four servers started with four
262,144-token slots, INT8 KV and MTP window 8 (`batch=4 ctx=262144 kv=int8 spec=8` in each `server.log`). The
exact commands are in each run's `server-command.txt`.

## Five decode runs of the source build on IQ3

BetterBench's weighted decode varies from run to run even at full power, so the IQ3 source build ran the eight decode
scenarios four more times, each in its own cooled, P0-checked pieces.

| Run | Weighted decode (tok/s) | Lowest P0 share | Busy decode GPU W |
| --- | ---: | ---: | ---: |
| [merged-iq3](merged-iq3) (full run) | **98.92** | 0.990 | 54.6 |
| [r1](merged-iq3-decode-repeats/r1) | 98.67 | 0.990 | 54.6 |
| [r2](merged-iq3-decode-repeats/r2) | 97.29 | 0.981 | 54.7 |
| [r3](merged-iq3-decode-repeats/r3) | 93.99 | 0.943 | 54.6 |
| [r4](merged-iq3-decode-repeats/r4) | 96.94 | 1.000 | 54.1 |

The best is 98.92 tok/s and the median of the five is 97.29. They range from 93.99 to 98.92. The README headline
uses the best run, which is also the one with the full prefill and concurrency sweep. A fifth repeat was stopped
partway to free the laptop and is not included.

## How the runs were assembled

Each run splits BetterBench into seventeen pieces (one per decode scenario, prompt depth and concurrency level) against
one server process. Before each piece, the runner waits for an idle, cool GPU (120 s under 52 °C and 15 W). A piece is
kept when the GPU was in P0 for at least 90% of its busy samples. Otherwise it is retried up to three times and the try
with the most P0 is kept. `results.json` is the kept pieces merged into one BetterBench result. Its `assembly` key
lists them, and each piece's own result and output are in `pieces`.
[Per-piece power](merged-iq3/power.json) · [GPU samples, one per second](merged-iq3/gpu.csv) · [Piece log](merged-iq3/pieces.log)

- **FP4-Local and DQ6_K.** Every piece held P0 on the first try. The lowest shares were 0.954 and 0.963, at 49–56 W.
- **IQ3, deep prompt reads.** At the 16K, 32K and 64K targets the IQ3 builds never reached 90% P0. The driver labels
  part of heavy IQ3 prefill as P4 while the SM clock stays near 1,900 MHz at about 57 W, with only the software power
  cap as the clock reason. So the 0.675 (release) and 0.749 (source) shares there are a P-state label, not a power
  drop. The best of three tries is kept. Decode and concurrency pieces held P0 (0.976 or higher).

## Greedy identity

Speed changes in this build are kept only when they produce the same greedy output as the build before them. The source
build here matched the previous engine for 768 single-request tokens. With `--slots --kv int8`, it also matched in 24
batched runs, four requests at once. Every check reported `GREEDY IDENTICAL`.
