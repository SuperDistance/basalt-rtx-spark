# Native llama.cpp on the RTX Spark N1X · November 25, 2026

A reference row for the [results table](../../../README.md#results): llama.cpp b12359 (`b1b770d90`, the Windows Arm64
CUDA build) serving Unsloth's `UD-IQ4_XS` GGUF of Qwen3.8-Flash-Next with its Q8_0 MTP head, on the same laptop, 96 GB
GPU split and BetterBench 0.6.1 preset as the Basalt rows.

| BetterBench metric | llama.cpp · UD-IQ4_XS |
| --- | ---: |
| Weighted single-stream decode | 45.08 tok/s |
| Aggregate decode, 1 / 2 / 3 / 4 streams | 41.5 / 45.9 / 60.7 / 67.2 tok/s |
| Short-prompt median time to first token | 437 ms |
| Cold prompt read at the 2K / 8K / 16K / 32K / 64K targets | 828 / 912 / 915 / 863 / 778 tok/s |

| Scenario | Weight | tok/s |
| --- | ---: | ---: |
| Chat | 0% | 39.4 |
| Code | 30% | 43.0 |
| File edit | 10% | 49.2 |
| Json | 15% | 55.6 |
| Math | 0% | 55.9 |
| Prose | 15% | 40.4 |
| Reasoning | 20% | 38.7 |
| Summarization | 10% | 51.3 |

## Server settings

Four parallel slots over a 262,144-token context (65,536 per slot, which BetterBench was told with
`--max-model-len 65536`; the deepest prompt, the 64K target, is about 47,100 tokens), flash attention, MTP drafting
with up to three tokens, and Qwen's recommended sampling defaults. The exact command is in
[server-command.txt](server-command.txt).

## Power and how the run was assembled

The first full run (kept in [first-run](first-run)) lost GPU power after its opening scenario: the GPU dropped from P0
to P4 at about 22 W and stayed there, and it scored 41.83 tok/s weighted and 57.9 tok/s at four streams. This run
splits BetterBench into seventeen pieces (one per decode scenario, prompt depth and concurrency level) against one
server process. Each piece waits for an idle, cool GPU (120 s under 52 °C and 15 W) and is kept when the GPU was in P0
for at least 90% of its busy samples. All seventeen held P0 on the first try (lowest share 0.915), at 37.2 to 58.9 W
busy GPU draw. [Per-piece power](power.json) · [GPU samples, one per second](gpu.csv) · [Piece log](pieces.log)

`results.json` is the kept pieces merged into one BetterBench result; its `assembly` key lists them. Each piece's own
result, configuration and output are in [pieces](pieces).
