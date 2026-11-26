# Basalt RTX Spark

### A 120B-class model at 99 tokens/s on your RTX Spark laptop. Native Windows on Arm, no WSL.

Run **Qwen3.8-Flash-Next**, a ~120B-parameter mixture-of-experts model, entirely on an NVIDIA RTX Spark N1X laptop:
**98.9 tok/s** for one chat, **121 tok/s** across four, a **0.27 s** first token, at about **58 W** for CPU and GPU together.
That beats a desktop RTX 4080 SUPER on this model (90.2 tok/s with Strata), and nothing leaves the machine.

**The goal:** the fastest single-chat decode for Qwen3.8-Flash-Next and Qwen 4 Flash, plus up to four chats at once
with no frozen chat: a new long prompt is read alongside the chats already decoding.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Windows on Arm64](https://img.shields.io/badge/Windows-Arm64-0078D4)
![RTX Spark N1X](https://img.shields.io/badge/NVIDIA-RTX%20Spark%20N1X-76B900)
![CUDA sm_121](https://img.shields.io/badge/CUDA-sm__121-426417)

**[⬇ Download for Windows Arm64](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest)** ·
[Quick start](#quick-start) · [Results](#results) · [Which model pack?](#which-model-pack)

<p>
  <a href="docs/showcase/realtime-chat"><img src="docs/showcase/realtime-chat/chat.webp" width="49%" alt="One chat streaming in real time on the laptop: 970 tokens in 9.29 s, 106 tok/s decode, first token 0.12 s."></a>
  <a href="docs/showcase/orbital-transfer"><img src="docs/showcase/orbital-transfer/preview.webp" width="49%" alt="The Long Way Up, an animated orbital transfer drawn by Flash-Next on RTX Spark."></a>
</p>

*Left: [one chat in real time](docs/showcase/realtime-chat), 106 tok/s. Right: an [animated orbital transfer](docs/showcase/orbital-transfer) Flash-Next wrote from one prompt in 123 s.*

![This project’s weighted single-stream decode is 98.92 output tok/s on IQ3, 91.85 on NVIDIA FP4-Local and 87.94 on NVFP4-DQ6_K. The reference Arm64 port and native vLLM, measured on November 24, reach 84.05 and 54.19, and native llama.cpp with UD-IQ4_XS reaches 45.08. The second column shows cold prefill at the 2K target.](benchmarks/img/engines-overview.svg)

- **2.2× native llama.cpp, 1.6× native vLLM** on the same laptop. [Results](#results)
- **A native Windows app.** No Linux, no WSL, no Docker: unzip, run `setup.cmd`, run `start.cmd`.
- **Made for unified memory.** It places the model across the GPU's share and system RAM by itself, at whatever
  memory split your laptop uses, and reads RAM-resident experts in place.
- **Speedups that keep the output.** Each engine change is checked to produce the same greedy text as before it, one
  request at a time and four at once. Each model pack's quality is measured separately and published below.
- **Drop-in API.** OpenAI- and Anthropic-compatible endpoints, so chat apps and coding agents connect unchanged.

## Quick start

You need an RTX Spark N1X laptop (128 GB measured; [smaller machines](docs/rtx-spark-setup.md#choose-a-model-and-memory-split)),
a current NVIDIA driver, the ARM64 Visual C++ runtime, Python 3.10+, and about 120 GB of free disk.

1. Download the Windows Arm64 ZIP from [Releases](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest) and
   extract it. The CUDA libraries are inside.
2. Run `setup.cmd`. It creates the Python environment and offers to download the default model pack (about 82 GiB).
3. Run `start.cmd`, then open `http://127.0.0.1:8080/` for the web UI, or point any OpenAI client at
   `http://127.0.0.1:8080/v1` (model `flash-next`, any API key).

The first start writes a copy of the model's n-gram table beside it (about 29 GB, once). Building from source and
the server options are in the [setup guide](docs/rtx-spark-setup.md).

## Results

One 128 GB RTX Spark N1X with a 96 GB GPU allocation, physical cooling, the same 392-request
[BetterBench](benchmarks) preset for every row.

| | Engine | Model pack | Decode, 1 chat (tok/s) | Decode, 4 chats (tok/s) | First token, short prompt | Prompt read, 2K / 64K (tok/s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **This project** | **Basalt RTX Spark** · source build | `UD-IQ3_XXS` | **98.92** | **120.6** | **266 ms** | **1,929** / 2,431 |
| **This project** | **Basalt RTX Spark** · release ZIP | `UD-IQ3_XXS` | 97.10 | 119.9 | 270 ms | 1,883 / 2,429 |
| **This project** | **Basalt RTX Spark** · source build | `NVIDIA FP4-Local` | 91.85 | 101.0 | 346 ms | 1,741 / 2,655 |
| **This project** | **Basalt RTX Spark** · source build | `NVFP4-DQ6_K` | 87.94 | 100.3 | 352 ms | 1,716 / **2,657** |
| Reference · Nov 24 | Basalt + Arm64 port | `UD-IQ3_XXS` | 84.05 | 85.2 | 619 ms | 967 / 1,999 |
| Reference · Nov 24 | Native vLLM for RTX Spark 0.2.0 | `NVIDIA FP4-Local` | 54.19 | 100.8 | 271 ms | 1,511 / 2,035 |
| Reference | Native llama.cpp b12359 | `UD-IQ4_XS` | 45.08 | 67.2 | 437 ms | 828 / 778 |

Decode is BetterBench's weighted median over eight scenarios. Prompt reads are cold, at the 2K and 64K targets (about
1,550 and 47,100 actual tokens). The Basalt and llama.cpp rows ran on November 25 with the GPU in P0. The two November 24
rows were not rerun; that day's runs drew less GPU power in busy decode (49 W for the port, 35 W for vLLM, 46–48 W for
Basalt). On that day the release ZIP scored 92.76, 10% ahead of the port, and Basalt on FP4-Local scored 85.28, 57%
ahead of vLLM.
[November 25 runs](benchmarks/results/2026-11-25-n1x-rerun) ·
[llama.cpp run](benchmarks/results/2026-11-25-n1x-llama-cpp) ·
[November 24 runs](benchmarks/results/2026-11-24-n1x-cooled-release)

<details>
<summary>Charts by scenario, number of chats and prompt depth; power and test settings</summary>

![Median cold input tokens per second at five target prompt depths. This project has three columns, IQ3, NVIDIA FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-prefill.svg)

![Median output tokens per second in eight decode scenarios. This project has three columns, IQ3, NVIDIA FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-decode.svg)

![Aggregate output tokens per second at one through four streams. This project has three columns, IQ3, NVIDIA FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local, both from November 24, and native llama.cpp with UD-IQ4_XS.](benchmarks/img/engines-concurrency.svg)

- **Settings.** Basalt used four 262,144-token slots, INT8 KV and MTP 8. Native vLLM ran with FP8 target KV and MTP 7.
  llama.cpp used four slots over a 262,144-token context, flash attention and MTP 3. The release ZIP row is
  v0.4.1-rtx-spark.2. The source build is `main`.
- **Power.** Busy decode averaged 54.6 W GPU for the source build on IQ3, 54.3 W for the ZIP, 53.2 W on FP4-Local,
  53.7 W on NVFP4-DQ6_K and 45.4 W for llama.cpp. These runs logged the GPU only. The headline's 58 W CPU+GPU adds the
  CPU cores' 3.8 W from the November 24 IQ3 decode, which logged both rails (47.5 W GPU, 51.3 W together). Each piece of each run started on a cool GPU and was kept only if the
  GPU stayed in P0. [How the runs were assembled](benchmarks/results/2026-11-25-n1x-rerun#how-the-runs-were-assembled)
- **Memory (November 24).** Both FP4 packs kept all 24,576 GPU expert slots. Whole-GPU memory had nearly identical
  medians (72.62 GiB DQ6_K, 72.77 GiB FP4-Local) and peaks of 75.03 and 78.23 GiB.
  [Measured source](https://github.com/SuperDistance/basalt-rtx-spark/commit/1bc565f277a3fb02c6182ea5ba0e9086b3f637e0) ·
  [DQ6_K telemetry](benchmarks/results/2026-11-24-n1x-nvfp4-dq6-k) · [vLLM run](benchmarks/results/2026-11-24-n1x-verified-default/vllm)
- **Desktop comparison.** The Strata figure (90.2 tok/s, IQ3_S) is from a desktop RTX 4080 SUPER.
  [Desktop benchmark](benchmarks/desktop-decode.md)
- **Other published coding numbers.** On SPEED-Bench coding (80 Flash-Next conversations, each team's own client),
  NVIDIA reports 36.58 tok/s for llama.cpp with UD-IQ4_XS and the vLLM for RTX Spark project 70.44 tok/s with
  FP4-Local. This project has not run SPEED-Bench.
  [NVIDIA result](https://github.com/NVIDIA/local-ai-inference-recipes/blob/bb515086841e7a81e7d51925561338e7dee35023/models/qwen/qwen3.8-flash-next/rtx-spark/windows/llama-cpp/78gb/iq4-xs/vision-mtp/performance.yaml) ·
  [vLLM result](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md#nvidia-reference)

</details>

**Have an RTX Spark (or a DGX Spark)?** Run the [benchmark](benchmarks) and send your numbers.

## Which model pack?

| Pack | Made by | Pick it for | Top-token agreement with BF16 |
| --- | --- | --- | ---: |
| `UD-IQ3_XXS` | Unsloth | **speed**: the default, fastest decode in the [results](#results) | 88.67% |
| `NVFP4-DQ6_K` | Basalt | closest to BF16, ~11% slower decode | **91.65%** |
| `NVIDIA FP4-Local` | NVIDIA (NVFP4, quantization-aware distilled) | NVIDIA's own weights, as native vLLM serves them | not measured on the same test |

Agreement is the pack publisher's measurement on the same 100 WikiText-2 chunks.
[Publisher’s quality test](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/README.md#quality)
NVIDIA publishes its checkpoint as `Qwen3.8-Flash-Next-FP4-Local`; this project converts it to `.basalt`.

<details>
<summary>All supported packs and per-prompt-type speeds</summary>

Basalt documents these Flash-Next packs: `NVFP4-DQ6_K`, `NVFP4-DQ8`, `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-IQ3_XXS`,
`UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`, `Q8_0`, plus NVIDIA FP4-Local converted to `.basalt`.
[Basalt pack catalog](docs/models/qwen38-flash-next.md#pack) ·
[vLLM release recipes](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md)

Earlier N1X single-request tests, output tokens per second by prompt type (ranges span two runs per pack):

| | Engine / model | Model pack | Conversation | Structured output | Prose |
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

## What this fork changes

- **Faster prompt reads.** Short NVFP4 matrix multiplies take a BF16 activation path: a 716-token prompt reached its
  first token in 0.66 s instead of 1.37 s in the kernel comparison.
- **FP4 experts stay on the GPU.** A smaller staging buffer kept a 23,633-token prompt in one chunk without loading
  1,586 expert blobs from file: first token in 10.4 s instead of 14.2 s (**27% less wait**).
  [Controlled experiment](benchmarks/results/2026-11-24-n1x-fp4-prefill-chunks)
- **Four chats share one GPU pass.** Drafting for several requests at once took 5.42 ms per verification window
  instead of 21.53 ms on the Arm64 port (four-stream structured output).
- **Joining a busy server.** A new 16,672-token prompt got its first token in 26.2 s while three chats kept decoding;
  their longest pause was 389 ms. [Implementation and measurements](docs/rtx-spark-optimizations.md)

Basalt places model weights across GPU memory and system RAM automatically. Flash-Next defaults to four slots of up to
262,144 tokens each and an INT8 KV cache that grows on demand. [Setup and client guide](docs/rtx-spark-setup.md) ·
[Engine changes](docs/rtx-spark-setup.md#what-changed-from-basalt) · [Quality measurements](docs/precision/)

## Credits

- [Basalt](https://github.com/jesdga95/basalt) by Jesus Gulfo and contributors (MIT): the engine this fork builds on.
- [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) (MIT): the Arm64 and compute capability 12.1 port it
  builds on.
- The projects and models Basalt credits in [README.basalt.md](README.basalt.md) and
  [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): Qwen's model, Unsloth's UD-IQ3_XXS quantization, and more.
- NVIDIA's `Qwen3.8-Flash-Next-FP4-Local` checkpoint (NVIDIA Open Model Agreement), converted here to `.basalt`.

## License

MIT, see [LICENSE](LICENSE).
