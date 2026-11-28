# Basalt RTX Spark

### A 120B-class model at 105 tokens/s on your RTX Spark laptop. Native Windows on Arm, no WSL.

Run **Qwen3.8-Flash-Next**, a ~120B-parameter mixture-of-experts model, entirely on an NVIDIA RTX Spark N1X laptop:
**105 tok/s** for one chat, **125 tok/s** across four, a **0.27 s** first token, at about **58 W** for CPU and GPU
together. That is faster than a desktop RTX 4080 SUPER on this model (90.2 tok/s with Strata), and nothing leaves the
machine.

**The goal:** the fastest Qwen3.8-Flash-Next on RTX Spark devices running Windows on Arm, while keeping decent decode
speed with four concurrent requests, and no request freezing while a cold prompt joins.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Windows on Arm64](https://img.shields.io/badge/Windows-Arm64-0078D4)
![RTX Spark N1X](https://img.shields.io/badge/NVIDIA-RTX%20Spark%20N1X-76B900)
![CUDA sm_121](https://img.shields.io/badge/CUDA-sm__121-426417)

**[⬇ Download for Windows Arm64](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest)** ·
[Quick start](#quick-start) · [Results](#results) · [Which quant?](#which-quant) · [How it gets its speed](#how-it-gets-its-speed)

<p>
  <a href="docs/showcase/realtime-chat"><img src="docs/showcase/realtime-chat/chat.webp" width="49%" alt="One chat streaming in real time on the laptop: 970 tokens in 9.29 s, 106 tok/s decode, first token 0.12 s."></a>
  <a href="docs/showcase/orbital-transfer"><img src="docs/showcase/orbital-transfer/preview.webp" width="49%" alt="The Long Way Up, an animated orbital transfer drawn by Flash-Next on RTX Spark."></a>
</p>

*Left: [one chat in real time](docs/showcase/realtime-chat), 106 tok/s. Right: an [animated orbital transfer](docs/showcase/orbital-transfer) Flash-Next wrote from one prompt in 123 s.*

![Single-chat decode on November 27: 105.44 output tok/s on IQ3 (the best of five decode runs; their median is 104.17), 94.26 on NVIDIA FP4-Local and 95.37 on NVFP4-DQ6_K. The Arm64 port and native vLLM, measured on November 24, reach 84.05 and 54.19, and native llama.cpp with UD-IQ4_XS reaches 45.08. The second column shows four-chat output: 125.3, 106.1 and 105.2 for this project against 85.2, 100.7 and 67.2. The third shows cold prompt reads at the 2K target.](benchmarks/img/engines-overview.svg)

- **2.3× native llama.cpp and 1.9× native vLLM** for one chat on the same laptop. [Results](#results)
- **No frozen chats.** A 35K-token prompt joining three running chats starts answering in 17 s, and no chat pauses for
  more than 0.8 s. llama.cpp paused the chats for 2.9 s and took 46 s. [Join test](benchmarks/results/2026-11-26-n1x-join-and-sparkdash)
- **A native Windows app.** No Linux, WSL or Docker: unzip, run `setup.cmd`, run `start.cmd`.
- **Made for unified memory.** It places the model across the GPU's share and system RAM by itself, at whatever memory
  split your laptop uses.
- **Faster, same output.** Every engine change is checked to give the same greedy text as before, one request at a
  time and batched.
- **Drop-in API.** OpenAI- and Anthropic-compatible endpoints, so chat apps and coding agents connect unchanged.

## Quick start

You need an RTX Spark N1X laptop (128 GB measured; [smaller machines](docs/rtx-spark-setup.md#choose-a-model-and-memory-split)),
a current NVIDIA driver, the ARM64 Visual C++ runtime, Python 3.10+, and about 120 GB of free disk.

1. Download the Windows Arm64 ZIP from [Releases](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest) and
   extract it. The CUDA libraries are inside.
2. Run `setup.cmd`. It creates the Python environment and offers to download the default quant, `UD-IQ3_XXS` (about
   82 GiB).
3. Run `start.cmd`, then open `http://127.0.0.1:8080/` for the web UI, or point any OpenAI client at
   `http://127.0.0.1:8080/v1` (model `flash-next`, any API key).

The first start writes the model's n-gram table beside it (about 29 GB, once). Flash-Next gets four slots of up to
262,144 tokens each and an INT8 KV cache that grows on demand.
[Setup guide and building from source](docs/rtx-spark-setup.md) ·
[Engine changes](docs/rtx-spark-setup.md#what-changed-from-basalt) · [Quality measurements](docs/precision/)

## Results

One 128 GB RTX Spark N1X with a 96 GB GPU allocation and physical cooling. Every row ran the same 392-request
[BetterBench](benchmarks) preset.

| | Engine | Quant | Decode, 1 chat | Decode, 4 chats | First token | Prompt read, 2K / 64K |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **This project** | **Basalt RTX Spark** · source build | `UD-IQ3_XXS` | **105.44** | **125.3** | **267 ms** | **1,928** / 2,404 |
| **This project** | **Basalt RTX Spark** · release ZIP, Nov 25 | `UD-IQ3_XXS` | 97.10 | 119.9 | 270 ms | 1,883 / 2,429 |
| **This project** | **Basalt RTX Spark** · source build | `NVIDIA FP4-Local` | 94.26 | 106.1 | 353 ms | 1,715 / 2,561 |
| **This project** | **Basalt RTX Spark** · source build | `NVFP4-DQ6_K` | 95.37 | 105.2 | 358 ms | 1,706 / **2,660** |
| Reference · Nov 24 | Basalt + Arm64 port | `UD-IQ3_XXS` | 84.05 | 85.2 | 619 ms | 967 / 1,999 |
| Reference · Nov 24 | Native vLLM for RTX Spark 0.2.0 | `NVIDIA FP4-Local` | 54.19 | 100.8 | 271 ms | 1,511 / 2,035 |
| Reference | Native llama.cpp b12359 | `UD-IQ4_XS` | 45.08 | 67.2 | 437 ms | 828 / 778 |

Decode and prompt reads are in tokens per second, and first token is for a short prompt.

- **Decode** is BetterBench's weighted result over eight scenarios. For the IQ3 source build it is the best of five
  decode runs (median 104.17, range 100.34–105.44); its other columns come from the one full run.
- **Dates.** The source builds ran on November 27, and the release ZIP and llama.cpp on November 25, all with the GPU
  in P0. The Arm64 port and vLLM rows are from November 24 and drew less GPU power (49 W and 35 W). Basalt also drew
  less that day (46–48 W), and it still led the port by 10% and vLLM by 57%.
- **Prompt reads** are cold, at the 2K and 64K targets (about 1,550 and 47,100 actual tokens).

[November 27 runs](benchmarks/results/2026-11-27-n1x-main-rerun) ·
[November 25 runs](benchmarks/results/2026-11-25-n1x-rerun) ·
[llama.cpp run](benchmarks/results/2026-11-25-n1x-llama-cpp) ·
[November 24 runs](benchmarks/results/2026-11-24-n1x-cooled-release)

<details>
<summary>Charts by scenario, number of chats and prompt depth; power, settings and other comparisons</summary>

![Median cold input tokens per second at five target prompt depths. This project has three columns, IQ3, NVIDIA FP4-Local and NVFP4-DQ6_K, from November 27. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-prefill.svg)

![Median output tokens per second in eight decode scenarios. This project has three columns, IQ3 (the best of five decode runs), NVIDIA FP4-Local and NVFP4-DQ6_K, from November 27. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-decode.svg)

![Aggregate output tokens per second at one through four chats. This project has three columns, IQ3, NVIDIA FP4-Local and NVFP4-DQ6_K, from November 27. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-concurrency.svg)

- **Settings.** Basalt used four 262,144-token slots, INT8 KV and MTP 8. Native vLLM ran with FP8 target KV and MTP 7.
  llama.cpp used four slots over a 262,144-token context, flash attention and MTP 3. The release ZIP row is
  v0.4.1-rtx-spark.2. The source build is `main`.
- **Power.** Busy decode averaged 53.9 W GPU in the best IQ3 decode run, 51.8 W in the full IQ3 run, 52.1 W on
  FP4-Local, 53.3 W on NVFP4-DQ6_K, 54.3 W for the ZIP and 45.4 W for llama.cpp. These runs logged the GPU only. The
  headline's 58 W CPU+GPU adds the CPU cores' 3.8 W from the November 24 IQ3 decode, which logged both. Each piece of
  each run started on a cool GPU and was kept only if the GPU stayed in P0.
  [How the runs were assembled](benchmarks/results/2026-11-27-n1x-main-rerun#how-the-runs-were-assembled)
- **Memory (November 24).** Both FP4 quants kept all 24,576 GPU expert slots. Whole-GPU memory had nearly identical
  medians (72.62 GiB DQ6_K, 72.77 GiB FP4-Local) and peaks of 75.03 and 78.23 GiB.
  [Measured source](https://github.com/SuperDistance/basalt-rtx-spark/commit/1bc565f277a3fb02c6182ea5ba0e9086b3f637e0) ·
  [DQ6_K telemetry](benchmarks/results/2026-11-24-n1x-nvfp4-dq6-k) · [vLLM run](benchmarks/results/2026-11-24-n1x-verified-default/vllm)
- **Desktop GPU.** The Strata figure (90.2 tok/s, IQ3_S) is from a desktop RTX 4080 SUPER.
  [Desktop benchmark](benchmarks/desktop-decode.md)
- **Apple M5 Max.** Strata lists 108.9 and 97.0 tok/s for IQ3_XXS on a 128 GB M5 Max (greedy, 200 tokens, a short
  question and a story). The same test on this laptop gave 104.0 and 71.5, so the M5 Max is faster there, most of all on
  the story, where the drafts keep only 1.8 tokens per window.
  [Same test here](benchmarks/results/2026-11-27-n1x-main-rerun#the-m5-max-test-on-this-laptop) ·
  [Strata Mac results](https://github.com/Niko1221/Strata/blob/25e2319149dd2865afb6489e9bc8fb2c092a0aaa/docs/MAC.md#measured-m5-max-128-gb)
- **SPEED-Bench coding** (80 Flash-Next conversations, each team's own client): NVIDIA reports 36.58 tok/s for
  llama.cpp with UD-IQ4_XS, and the vLLM for RTX Spark project 70.44 tok/s with FP4-Local. This project has not run
  SPEED-Bench.
  [NVIDIA result](https://github.com/NVIDIA/local-ai-inference-recipes/blob/bb515086841e7a81e7d51925561338e7dee35023/models/qwen/qwen3.8-flash-next/rtx-spark/windows/llama-cpp/78gb/iq4-xs/vision-mtp/performance.yaml) ·
  [vLLM result](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md#nvidia-reference)

</details>

**Have an RTX Spark laptop or desktop?** Run the [benchmark](benchmarks) and send your numbers.

## Which quant?

| Quant | Made by | Pick it for | Top-token agreement with BF16 |
| --- | --- | --- | ---: |
| `UD-IQ3_XXS` | Unsloth | **speed**: the default, fastest decode in the [results](#results) | 88.67% |
| `NVFP4-DQ6_K` | Basalt | closest to BF16, ~10% slower decode | **91.65%** |
| `NVIDIA FP4-Local` | NVIDIA (NVFP4, quantization-aware distilled) | NVIDIA's own weights, as native vLLM serves them | not measured on the same test |

Agreement is the quant publisher's measurement on the same 100 WikiText-2 chunks.
[Publisher’s quality test](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/README.md#quality)
NVIDIA publishes its checkpoint as `Qwen3.8-Flash-Next-FP4-Local`; this project converts it to `.basalt`.

<details>
<summary>All supported quants and per-prompt-type speeds</summary>

Basalt documents these Flash-Next quants: `NVFP4-DQ6_K`, `NVFP4-DQ8`, `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-IQ3_XXS`,
`UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`, `Q8_0`, plus NVIDIA FP4-Local converted to `.basalt`.
[Basalt quant catalog](docs/models/qwen38-flash-next.md#pack) ·
[vLLM release recipes](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md)

Earlier N1X single-request tests, output tokens per second by prompt type (ranges span two runs per quant):

| | Engine / model | Quant | Conversation | Structured output | Prose |
| --- | --- | --- | ---: | ---: | ---: |
| **This project** | Basalt RTX Spark / Flash-Next | `UD-IQ3_XXS` | 79.0–84.1 | 141.3–147.0 | 73.3–75.4 |
| **This project** | Basalt RTX Spark / Flash-Next | `NVIDIA FP4-Local` | 81.6–82.7 | 133.5–136.1 | 68.3–69.2 |
| **This project** | Basalt RTX Spark / Qwen3.8-27B | `QUASAR-NVFP4` | 48.6 | 143.7 | 33.1 |
| Reference | Basalt + Arm64 port / Flash-Next | `NVFP4-DQ6_K` | 80.8 | 140.3 | 67.8 |

The Flash-Next IQ3 and FP4-Local runs alternated, with INT8 KV, MTP 8 and fully GPU-resident experts; busy GPU draw
was 36.8–42.4 W and 38.4–41.4 W. The Arm64 port used FP16 KV for its NVFP4-DQ6_K test. The 27B run used INT8 KV and a
16-token block drafter. [Per-run measurements](benchmarks/results/2026-11-23-n1x-quantization) ·
[Arm64 port fixtures](docs/arm64.md#speed-on-the-rtx-spark-n1x)

</details>

## How it gets its speed

Every change is lossless: greedy text matches the code before it token for token, batched requests match solo ones,
and sampling keeps the model's own distribution. Each item names the published idea it builds on.

- **The model drafts its own next tokens.** Flash-Next's multi-token-prediction (MTP) head proposes up to seven tokens,
  and one verification pass keeps exactly the ones plain decoding would have produced
  ([Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318); MTP as in
  [Gloeckle et al.](https://arxiv.org/abs/2404.19737) and [DeepSeek-V3](https://arxiv.org/abs/2412.19437)).
  - Every active chat drafts in one GPU pass: **5.42 ms** per window for four structured replies, against **21.53 ms**
    on the Arm64 port.
  - Drafts are drawn at the request's own sampling settings, so a sampled reply keeps more of them (**+4.5%** at
    temperature 0.7).
  - A scheduler sizes each window from the time spent verifying and the tokens kept (**+8.1%** at four chats).

  [Release and port measurements](benchmarks/results/2026-11-24-n1x-cooled-release#serving-behavior-under-load) ·
  [Scheduler experiment](benchmarks/results/2026-11-24-n1x-four-stream-scheduler#scheduler-alone) ·
  [Drawn drafts](benchmarks/results/2026-11-27-n1x-main-rerun#drawn-drafts-versus-the-heads-top-draft)
- **New prompts are read beside running chats.** As in the chunked prefills of
  [Sarathi-Serve](https://arxiv.org/abs/2403.02310), a cold prompt is read in chunks on its own thread while the other
  chats keep decoding. It started as Strata's opt-in `STRATA_PIPED_CONCURRENT=1`, contributed by this project's author
  ([Strata #2326](https://github.com/Niko1221/Strata/pull/2326): the longest stall went from 11.9 s to 0.8 s). Here it
  is on by default for prompts over 2,048 tokens.
  [Strata design notes](https://github.com/Niko1221/Strata/blob/c234774038d1c9249cb33637d27b895d47b39786/docs/BATCHING.md#reading-a-prompt-beside-pipelined-groups-experimental-opt-in) ·
  [Join test](benchmarks/results/2026-11-26-n1x-join-and-sparkdash)
- **Experts are placed across unified memory.** MoE offloading usually copies experts to the GPU on demand
  ([Eliseev and Mazur](https://arxiv.org/abs/2312.17238)). The N1X's GPU reads pinned RAM directly, so the planner
  budgets GPU memory, pinned RAM and Windows commit together and serves RAM-resident experts in place. At the 96 GB
  split, all 24,576 of the IQ3 quant's experts fit in the GPU's share. [Placement design](docs/n1x-gpu-first.md)
- **FP4 prompts run on BF16 activations.** The kernel widens 4-bit weights in registers for BF16 tensor-core products,
  the W4A16 approach of [Marlin](https://arxiv.org/abs/2408.11743): a 716-token prompt's first token in **0.66 s**
  instead of **1.37 s**. A smaller staging ring also keeps long FP4 prompts in one chunk, with no expert reads from
  file: first token **27% sooner** on a 23,633-token prompt.
  [Kernel and quality](docs/n1x-gpu-first.md#nvfp4-prompts-on-bf16-activations-w4a16) ·
  [Controlled experiment](benchmarks/results/2026-11-24-n1x-fp4-prefill-chunks)
- **Faster prompt kernels.** FP16 copies of 300 dense projections stay resident, with faster kernels for the
  [Gated DeltaNet](https://arxiv.org/abs/2412.06464) recurrence, the long-context indexer scores and int8 sparse
  attention. A ~110-token prompt reads in 158–176 ms instead of 194–215 ms. [Prompt path](docs/n1x-prompt-path.md)
- **Conversations resume from what the engine holds.** As in the prefix reuse of
  [SGLang's RadixAttention](https://arxiv.org/abs/2312.07104), a follow-up request continues the engine's live state
  instead of reading the conversation again: a resumed thinking turn restores in 8–13 ms. When a reply reaches its
  thinking budget, the closing pass reads 4 tokens instead of 183–358 (**+2.9%** single-chat decode, **+4.8%** at four
  chats). [Pull request](https://github.com/SuperDistance/basalt-rtx-spark/pull/27)

**Where decode stands.** A single-chat verification window moves about 85% of the bytes per second the N1X's memory
can deliver, so faster kernels can buy little more. The remaining gains are in keeping more drafted tokens per window
and in removing serving stalls like the one above.

<details>
<summary>Tested and not shipped</summary>

- **Block verification** ([Sun et al.](https://arxiv.org/abs/2403.10444)) accepts a run of drafts jointly instead of
  one token at a time. It is exact, but on BetterBench at temperature 0.7 it kept 3.20 tokens per window, the same as
  token-by-token verification: the MTP head's kept drafts already agree with the model.
- **The N1X NPU.** Running experts on it cut serving throughput by 24.5%. It shares the GPU's memory bandwidth, runs an
  expert 6–8× slower and computes only in FP16, so its output would not match.
- **CPU expert workers.** The GPU already reads RAM-resident experts in place, so CPU workers would have nothing left
  to compute (0.00 ms per window).

</details>

## Credits

- [Basalt](https://github.com/jesdga95/basalt) by Jesus Gulfo and contributors (MIT): the engine this fork builds on.
- [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) (MIT): the Arm64 and compute capability 12.1 port it
  builds on.
- The projects and models Basalt credits in [README.basalt.md](README.basalt.md) and
  [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): Qwen's model, Unsloth's UD-IQ3_XXS quantization, and more.
- NVIDIA's `Qwen3.8-Flash-Next-FP4-Local` checkpoint (NVIDIA Open Model Agreement), converted here to `.basalt`.

## License

MIT, see [LICENSE](LICENSE).
