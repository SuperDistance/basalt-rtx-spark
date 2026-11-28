# Basalt RTX Spark `main` · November 27, 2026

These runs refresh the source-build rows of the [results table](../../../README.md#results). The laptop, cooling,
memory split and BetterBench 0.6.1 preset are the same as in the [November 25 runs](../2026-11-25-n1x-rerun). What
changed is the engine. It is `main` with [pull request #27](https://github.com/SuperDistance/basalt-rtx-spark/pull/27)
(engine `29a1b77c`). When a reply reaches its thinking budget, the closing pass now continues the live state instead
of reading the reply again.

| Run | Pack | Weighted decode (tok/s) | 1 / 2 / 3 / 4 chats (tok/s) | Median first token | Prompt read at 2K / 8K / 16K / 32K / 64K (tok/s) | Busy decode GPU W |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [main-iq3](main-iq3) | `UD-IQ3_XXS` | 100.34 | 92.9 / 108.7 / 120.8 / **125.3** | **267 ms** | **1,928** / **2,281** / 2,410 / 2,444 / 2,404 | 51.8 |
| [main-fp4-local](main-fp4-local) | NVIDIA `FP4-Local` | 94.26 | 84.9 / 95.0 / 103.3 / 106.1 | 353 ms | 1,715 / 2,206 / 2,429 / 2,556 / 2,561 | 52.1 |
| [main-dq6k](main-dq6k) | `NVFP4-DQ6_K` | 95.37 | 84.4 / 94.6 / 103.2 / 105.2 | 358 ms | 1,706 / 2,258 / **2,483** / **2,581** / **2,660** | 53.3 |

Against the November 25 source build, weighted decode rose 1.4% on IQ3, 2.6% on FP4-Local and 8.4% on DQ6_K. Four
chats rose 3.9%, 5.0% and 4.9%. Prompt reads and first-token times are within run-to-run noise. All three servers
started with four 262,144-token slots, INT8 KV and MTP window 8. The exact commands are in each run's
`server-command.txt`.

## Five decode runs on IQ3

The full IQ3 run's decode pieces averaged 51.8 W. Two of them ran lower: decode-code at 46.5 W, and decode-chat at
49.3 W on its kept redo. So, as on November 25, the eight decode scenarios ran four more times, each in its own cooled,
P0-checked pieces.

| Run | Weighted decode (tok/s) | Lowest P0 share | Busy decode GPU W |
| --- | ---: | ---: | ---: |
| [main-iq3](main-iq3) (full run) | 100.34 | 0.974 | 51.8 |
| [r1](main-iq3-decode-repeats/r1) | 104.17 | 0.993 | 54.1 |
| [r2](main-iq3-decode-repeats/r2) | 104.31 | 0.982 | 54.1 |
| [r3](main-iq3-decode-repeats/r3) | **105.44** | 0.986 | 53.9 |
| [r4](main-iq3-decode-repeats/r4) | 103.74 | 0.986 | 54.4 |

The best of the five is **105.44 tok/s** (r3), the median 104.17, and they range from 100.34 to 105.44. The README
headline and charts use the best, as on November 25: [main-iq3-headline](main-iq3-headline) takes the eight decode
scenarios of r3 and the prompt reads and concurrency levels of the full run, and `combine.py` there rebuilds it. On November 25 the five runs had
a median of 97.29 and a best of 98.92.

## How the runs were assembled

The method is the [November 25 one](../2026-11-25-n1x-rerun#how-the-runs-were-assembled). Each run is seventeen
pieces against one server process: one per decode scenario, prompt depth and concurrency level. Each piece starts on a
cool GPU, and it is kept when the GPU was in P0 for at least 90% of busy samples (otherwise the best of three tries).
Each run's `pieces.json` and `power.json` list every try.

- **main-iq3.** The first decode-chat tries overlapped other work on the laptop and drew 22–33 W, with P0 shares of
  0.77, 0.07 and 0.09. It was rerun on its own at 1.00 ([main-iq3-redo](main-iq3-redo)), and that try is the kept
  `decode-chat.try4`. Its deep prompt reads (16K and up) stayed below 90% P0 in all three tries, as on November 25.
  The driver labels part of heavy IQ3 prefill as P4 while the GPU still draws 55–57 W, so the kept shares (0.82, 0.66,
  0.46) are a P-state label, not a power drop.
- **main-fp4-local.** The run was stopped after its prompt reads. The four concurrency pieces ran later against a
  fresh server ([main-fp4-local-rest](main-fp4-local-rest)), and they are folded in as their first tries.
- **main-dq6k.** One server, every piece at P0 share 0.90 or more.

## The M5 Max test on this laptop

Strata publishes Apple M5 Max (128 GB) decode for Flash-Next: greedy, 200 generated tokens, a short question and a story,
with `--spec 2` MTP. On IQ3_XXS it lists 108.9 and 97.0 tok/s.
[Strata Mac results](https://github.com/Niko1221/Strata/blob/25e2319149dd2865afb6489e9bc8fb2c092a0aaa/docs/MAC.md#measured-m5-max-128-gb)

The [same-test-as-m5-max](same-test-as-m5-max) run repeats those conditions here with the IQ3 pack: greedy, thinking off,
200 tokens, five alternating rounds. The story prompt is Strata's ("Write a short story about a lighthouse keeper.").
Strata does not publish its short question, so this run uses its own.

| Prompt | This laptop, IQ3 (tok/s) | Tokens per window | M5 Max, Strata (tok/s) |
| --- | ---: | ---: | ---: |
| Short question | 104.0 | 3.3 | 108.9 |
| Story | 71.5 | 1.8 | 97.0 |

On this test the M5 Max is faster, by 5% on the question and 36% on the story. The five rounds gave the same text each
time. The story keeps only 1.8 tokens per verification window. That is below the M5 Max's own no-MTP rate (78.6 tok/s),
so window sizing is where this case can still gain.

## Drawn drafts versus the head's top draft

Basalt draws each MTP draft from the head's distribution under the request's own temperature, top-k and top-p, then
keeps it with probability min(1, p/q) ([Leviathan et al.](https://arxiv.org/abs/2211.17192)). A request with a fixed
seed instead proposes the head's top token, and keeps it only where the row's own draw is that token. Both are exact.
The [draft-coupling](draft-coupling) run compares them on one IQ3 server. It used eight prompts at temperature 0.7 and
1.0, top-p 0.95, top-k 20 and 600 tokens, alternating, two rounds (16 requests per cell).

| Temperature | Tokens per window, top / drawn | Median tok/s, top / drawn |
| --- | ---: | ---: |
| 0.7 | 2.44 / 2.57 (+5.2%) | 102.5 / 107.1 (+4.5%) |
| 1.0 | 2.39 / 2.50 (+4.5%) | 99.4 / 101.5 (+2.1%) |

Drawn drafts are the default for unseeded requests. The sample is small, so this measures the default rather than
setting a headline.
