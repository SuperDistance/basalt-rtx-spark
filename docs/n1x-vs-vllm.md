# Basalt against vLLM on the N1X

BetterBench 0.6.1 with one config for both (temperature 0.7, cold prefix cache, concurrency 1-4) on the 96 GB GPU split.
Basalt serves UD-IQ3_XXS; vLLM for RTX Spark 0.2.0 (vLLM 0.32.0) serves FP4-Local with its default recipe (MTP 7 with
block verification, Marlin weight-only NVFP4 MoE, b12x MXFP8 dense; its kernels are sm_121a code, and its FP4 GEMMs need
sm_100, so the experts run weight-only).

| | This project · Basalt `2dacc65` | This project · Basalt `a67b9d4` | This project · Basalt `08f5791` | This project · Basalt `c4ca7d2` | Reference engine · vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single-stream decode, weighted median (tok/s) | 91.9 | 88.8 | 80.9 | 88.2 | 55.1 |
| First token, short prompts, p50 (ms) | 375 | 363 | 420 | 324 | 279 |
| Aggregate output at 1 / 2 / 3 / 4 streams (tok/s) | 77.6 / 80.9 / 91.4 / 93.9 | 77.5 / 85.1 / 96.4 / 99.4 | 75.8 / 82.8 / 91.4 / 94.4 | 77.2 / 86.6 / 94.5 / 98.3 | 53.0 / 81.2 / 97.8 / 104.4 |
| Prefill at 2K / 8K / 16K / 32K / 64K (tok/s) | 1,218 / 1,590 / 1,768 / 1,840 / 1,845 | 1,070 / 1,588 / 1,760 / 1,699 / 1,729 | 1,278 / 1,665 / 1,752 / 1,780 / 1,767 | 1,347 / 1,700 / 1,777 / 1,786 / 1,781 | 1,370 / 1,773 / 1,815 / 2,035 / 2,182 |
| Prefill sweep run alone (tok/s) | | | 1,704 / 1,970 / 2,037 / 1,979 / 1,766 | | |

A full run takes about 45 minutes, so every Basalt run after its first eight minutes or so is on the firmware's
sustained floor ([GPU power](n1x-gpu-power.md)): 08f5791's decode phase drew about 22 W at 1.6 GHz. The full runs'
differences are within what the power state moves, and the controlled comparisons say so: decode alternated between
a67b9d4 and 08f5791 at the same floor (ABBA, sampled, thinking on) gave 138.7 and 136.2 tok/s on long generations and
82.1 and 80.6 on short ones, with the 08f5791 arms about 1 W lower; on the CLI the same 1,500-token generation ran at 85.0
and 84.6 tok/s while the host waited 275 and 7.6 ms on n-gram rows. The prefill sweep run alone started cooler: its
2K-16K figures are above vLLM's.

The 2dacc65 build on the 64 GB split had measured 74.0 tok/s decode and 697 tok/s at 2K: there the prompt path read most
of its experts from RAM. vLLM gains on concurrency because it draws its MTP drafts at every concurrency; Basalt's batch
windows took argmax drafts until 798e173 (a67b9d4 and later).

c4ca7d2 adds the faster long-context prompt scores, the millisecond pipe wait (about 45 ms less per request) and
the dense projections' FP16 copies ([prompt path](n1x-prompt-path.md)); its first token on short prompts is the
clearest change (324 ms p50 against 420 and 363 in the earlier full runs), the rest is within the power state's spread.

## Against Basalt with the Arm64 port

The same benchmark and config, both builds on the same pack and settings, run back to back on 2026-11-23 with the
laptop's sustained-power controller on (it re-sends the thermal profile when the GPU's power budget starts to decline,
so neither run sits on the 22-28 W floor). The busy GPU drew 42.3 W at 2,063 MHz in the first run and 40.8 W at
2,072 MHz in the second. These figures are not comparable with the table above, which was measured without the
controller.

| | Basalt + Arm64 port (2ec7a84) | This fork (004d287) |
| --- | ---: | ---: |
| Single-stream decode, weighted median (tok/s) | 87.5 | 97.9 |
| First token, short prompts, p50 (ms) | 622 | 284 |
| Aggregate output at 1 / 2 / 3 / 4 streams (tok/s) | 74.6 / 81.1 / 85.7 / 85.3 | 86.5 / 99.4 / 110.8 / 113.1 |
| First token at 4 streams, p50 / p99 (ms) | 727 / 2,351 | 430 / 1,326 |
| Prefill at 2K / 8K / 16K / 32K / 64K (tok/s) | 983 / 1,506 / 2,055 / 1,951 / 2,095 | 1,724 / 2,077 / 2,149 / 2,295 / 2,342 |

Both builds placed every routed expert on the GPU at load. A 20-second fp16 GEMM from another program ran during the
second run (2ec7a84) about 20 minutes in; at most a few of its requests overlapped it.
