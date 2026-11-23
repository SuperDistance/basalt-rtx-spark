# Basalt against vLLM on the N1X

BetterBench 0.6.1 with one config for both (temperature 0.7, cold prefix cache, concurrency 1-4) on the 96 GB GPU split.
Basalt serves UD-IQ3_XXS; vLLM for RTX Spark 0.2.0 (vLLM 0.32.0) serves FP4-Local with its default recipe (MTP 7 with
block verification, Marlin weight-only NVFP4 MoE, b12x MXFP8 dense; its kernels are sm_121a code, and its FP4 GEMMs need
sm_100, so the experts run weight-only).

| | Basalt 2dacc65 | Basalt a67b9d4 | Basalt 08f5791 | Basalt c4ca7d2 | vLLM |
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
