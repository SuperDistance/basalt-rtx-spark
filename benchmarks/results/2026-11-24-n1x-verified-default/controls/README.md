# Scheduler and arrival controls

These fixed-output controls are separate from the official replay of the former local default `0cb0b207`. The four fixed-output controls alternated baseline A (`bd70ac59`) and candidate B (`3b27d149`) in A1/B1/B2/A2 order. Each run served four concurrent structured requests with a 1,024-token output cap. Throughput is aggregate output tokens per second over each wave.

| Arm | Build | Aggregate tok/s | Busy GPU draw | Median busy SM clock | Measured SoC rail aggregate |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | bd70ac59 | 142.49 | 43.9 W | 2,190 MHz | 52.9 W |
| B1 | 3b27d149 | 157.15 | 42.4 W | 2,106 MHz | 48.4 W |
| B2 | 3b27d149 | 147.21 | 36.3 W | 1,995 MHz | 41.9 W |
| A2 | bd70ac59 | 135.44 | 29.9 W | 1,891 MHz | 36.3 W |

The two-run means were **138.97 tok/s** for A and **152.18 tok/s** for B (**+9.5%**). Clock and power fell through the A/B/B/A sequence, so compare the individual arms and their conditions; the throughput delta alone does not isolate an engine-only gain. The per-arm response files and GPU/SoC telemetry are retained beside this summary.

With four slots and per-slot `spec=8`, the verifier can process up to 16 aggregate rows in a window. The ordinary mixed C4 phase exercised a narrower pattern: its 2,498 batch windows averaged 3.83 active slots and 7.89 actual rows, with 1.89 committed tokens per window per slot; it accepted 8,504 of 10,151 drafts. In the structured fixed-output B1/B2 controls, the engine log reports 11.98 and 11.06 rows respectively, and 2.89 and 2.63 committed tokens per slot-window. This workload difference helps explain why an optimization visible in fixed-output controls need not add the same percentage to broad BetterBench. The two mixed-C4 engine summary lines are preserved in [`mixed-c4-batch-summary.txt`](mixed-c4-batch-summary.txt).

The row trace contains 95 eight-row windows and 185 windows with a sixteen-row budget; 183 of those used 16 actual rows and two used 15 ([row trace](row-trace/batch-rows.txt)). Mean committed rows were 7.74 and 15.32 per window at budgets 8 and 16. `completed.json` records that adaptive widths were observed. This confirms that the default four-slot, `spec=8` configuration used wider aggregate verifier windows during the run.

The cold-join control passed with a 16,672-token uncached fourth prompt. It reached its first token in **26.96 s**; mean busy NVIDIA GPU draw was **43.48 W** over the whole control and **44.50 W** from join submission through first token. All three incumbent streams kept generating through prefill, producing 252, 249 and 214 tokens during the join, with a longest observed gap of about **383 ms**. The reverse-arrival check also passed: a 49,966-token cold prompt yielded at 4,096 tokens, the short request completed first at **1.66 s** at **41.68 W** mean busy NVIDIA GPU draw over the whole control, and the long request resumed from exactly 4,096 cached tokens.

The detailed outputs, verification flags, phase telemetry, and adaptive-width trace are in the sibling files and subdirectories. See [power method](../power-method.md) for sampling rules.
