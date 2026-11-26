# A long prompt arriving during three chats, and sparkDash · November 26, 2026

Same RTX Spark N1X laptop (128 GB, 96 GB GPU split, physical cooling) as the [November 25 runs](../2026-11-25-n1x-rerun).
Three engines, each on its own fresh server:

| Engine | Pack | Server settings |
| --- | --- | --- |
| Basalt RTX Spark, source build `209a14bb` (the engine on `main`) | `UD-IQ3_XXS` | defaults: four 262,144-token slots, INT8 KV, MTP 8 |
| Basalt + Arm64 port `2ec7a84` | `UD-IQ3_XXS` | the November 24 config: four 262,144-token slots, INT8 KV, MTP 8 |
| llama.cpp b12359 | `UD-IQ4_XS` | four slots over 262,144 tokens, flash attention, MTP 3 (as in the [llama.cpp run](../2026-11-25-n1x-llama-cpp)) |

Exact commands are in each engine's `server-command.txt`.

## A new long prompt while three chats decode

[`benchmarks/join_bench.py`](../../join_bench.py) works against any OpenAI-compatible server. Three chats stream (thinking
off, temperature 0). Once each has produced output, a long prompt arrives that no cache can match, because a fresh id
leads it. The test records when every streamed update arrived. Each depth ran twice; the table gives the medians.

| New prompt (actual tokens) | | Basalt RTX Spark | Arm64 port | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| 4,270 | new prompt's first token | **2.71 s** | 3.66 s | 5.39 s |
| | longest pause in a running chat | **0.48 s** | 3.55 s | 2.25 s |
| 17,360 | new prompt's first token | **8.49 s** | 86.35 s | 22.08 s |
| | longest pause in a running chat | 0.67 s | **0.17 s** | 2.65 s |
| 34,790 | new prompt's first token | **16.94 s** | 131.70 s | 45.87 s |
| | longest pause in a running chat | 0.75 s | **0.20 s** | 2.90 s |

- **Basalt RTX Spark** reads the new prompt alongside the running chats. Each chat keeps moving, with no pause over
  0.8 s, and the new prompt reads at 1,600–2,050 tok/s. While the prompt is read, the chats slow to about 5–9 streamed
  updates a second, against 29–37 before it arrived.
- **The Arm64 port** reads prompts under 16,384 tokens on their own, so a 4K prompt freezes all three chats for as long
  as it takes to read. Longer prompts read alongside decode but get little GPU time: the chats keep 10–15 updates a
  second and the new prompt waits 86–132 s.
- **llama.cpp** pauses the chats for 2.2–2.9 s at every size and reads the new prompt at about 750–800 tok/s.

Streamed updates are not tokens: Basalt and the port send several drafted tokens in one update. The per-chat update
rates are in each `join.json` (`updates_per_s_before`, `updates_per_s_during`).

What this fork changed: upstream Basalt already read prompts of 16,384 tokens or more alongside decode. This build
starts doing that from 2,048 tokens and spaces the decode windows so the prompt gets GPU time
([optimizations](../../../docs/rtx-spark-optimizations.md)).

## sparkDash on Basalt RTX Spark

sparkDash is the benchmark the TensorFold RTX Spark port publishes. It ran here with its own decode and prefill code,
called headlessly. The runner and its notes are outside this repository.
- **Decode:** forced 400-token replies, client-side timing, and the aggregate counted from the first to the last token.
- **Prefill:** a fresh id leads every prompt, and the rate uses the server's own prompt timing.

| Decode, tok/s | 1 chat | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose, thinking off, temp 0 | 93.3 | 109.8 | 116.9 | 112.5 | 130.4 |
| Prose, thinking on, temp 0.7 | 87.3 | 107.8 | 134.1 | 134.8 | 124.8 |
| Code, thinking off, temp 0 | 148.0 | 134.9 | 146.5 | 148.5 | 142.5 |

The server has four slots, so the fifth stream waits for one (TTFT 2.6–2.8 s on prose). Per-stream rates and TTFT are
in [sparkdash.md](basalt-rtx-spark-source/sparkdash.md).

| Prompt read (cold) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actual tokens | 1,064 | 2,087 | 4,134 | 8,231 | 16,420 | 32,806 | 65,572 | 131,111 |
| tok/s | 1,968 | 2,154 | 2,259 | 2,350 | 2,406 | 2,408 | 2,343 | 2,178 |

The TensorFold port's published sparkDash table comes from an ASUS ProArt GR1X, a desktop with the same GB10 chip. It
reports 68.2 / 58.2 / 98.6 tok/s for one prose, prose-with-thinking and code stream, 137.0 / 108.9 / 194.1 at four
streams, and 3,198 / 2,778 / 2,627 / 2,504 tok/s prompt reads at 2K / 16K / 64K / 128K. Different hardware and power
make that a reference, not a matched comparison.

## Power

Mean GPU power over samples at 80% utilization or more, for each engine's whole run:

| Engine | Mean GPU power | P0 share |
| --- | ---: | ---: |
| Basalt RTX Spark (sparkDash and join test) | 47.9 W | 0.61 |
| Arm64 port | 49.8 W | 1.00 |
| llama.cpp | 51.4 W | 1.00 |

Basalt's P0 share includes sparkDash's long prompt reads. The driver labels part of heavy IQ3 prefill as P4 at
unchanged clocks, as in the November 25 runs. [GPU samples](basalt-rtx-spark-source/gpu.csv)
