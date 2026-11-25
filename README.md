# Basalt RTX Spark

## 120B-class Qwen3.8-Flash-Next at 92.8 tokens/s on your RTX Spark laptop

With UD-IQ3_XXS weights, the N1X averaged **51.3 W across CPU and GPU** during busy decode on Windows 11 Arm64. That puts its single-stream speed in the range of a desktop RTX 4080 SUPER running Strata (90.2 tok/s on IQ3_S). [Power measurements](benchmarks/results/2026-11-24-n1x-cooled-release/release/power.json) · [Desktop benchmark](benchmarks/desktop-decode.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Windows on Arm64](https://img.shields.io/badge/Windows-Arm64-0078D4)
![RTX Spark N1X](https://img.shields.io/badge/NVIDIA-RTX%20Spark%20N1X-76B900)
![CUDA sm_121](https://img.shields.io/badge/CUDA-sm__121-426417)

![This project’s weighted single-stream decode is 92.76 output tok/s on IQ3, 85.28 on FP4-Local and 85.22 on NVFP4-DQ6_K. The reference Arm64 port and native vLLM reach 84.05 and 54.19. The second column shows cold prefill at the 2K target.](benchmarks/img/engines-overview.svg)

**SPEED-Bench coding on Windows Arm64** · llama.cpp **36.58 tok/s** (NVIDIA, UD-IQ4_XS) · native vLLM **70.44 tok/s** (FP4-Local). Both teams report 80 Flash-Next coding conversations using their own clients. [NVIDIA result](https://github.com/NVIDIA/local-ai-inference-recipes/blob/bb515086841e7a81e7d51925561338e7dee35023/models/qwen/qwen3.8-flash-next/rtx-spark/windows/llama-cpp/78gb/iq4-xs/vision-mtp/performance.yaml) · [vLLM result](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md#nvidia-reference)

[Download for Windows Arm64](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest) · [Setup guide](docs/rtx-spark-setup.md) · [Benchmark results](#results) · [Model packs and speeds](#quantization) · [Connect a client](docs/rtx-spark-setup.md#use-it-from-your-tools) · [One-shot showcase](#one-prompt-one-orbital-transfer)

## Results

With the same IQ3 weights and physical cooling, Basalt RTX Spark decoded **10.4% faster** than the Arm64 port and served four streams at **31.5% higher aggregate throughput**. Every row below uses the same 392-request BetterBench preset on one 128 GB N1X with a 96 GB GPU allocation.

| Group | Engine | Weight pack tested | Weighted single-stream decode (tok/s) | Short-prompt time to first token (ms) | Four-stream mixed throughput (tok/s) |
| --- | --- | --- | ---: | ---: | ---: |
| **This project** | **Basalt RTX Spark** · release ZIP | `UD-IQ3_XXS` | **92.76** | 280 | **112.10** |
| **This project** | **Basalt RTX Spark** · source build | `FP4-Local` | 85.28 | 365 | 95.88 |
| **This project** | **Basalt RTX Spark** · source build | `NVFP4-DQ6_K` | 85.22 | 365 | 95.84 |
| **Reference** | Basalt + Arm64 port | `UD-IQ3_XXS` | 84.05 | 619 | 85.23 |
| **Reference** | Native vLLM for RTX Spark 0.2.0 | `FP4-Local` (NVFP4 / MXFP8) | 54.19 | **271** | 100.75 |

**Cold prompts.** At the 2K target, Basalt read **1,767 input tok/s**, up **82.7%** from the port’s 967. The first token arrived in **0.88 s** versus **1.59 s** (44.8% sooner). The eight prompts per engine contained 1,522–1,593 tokens. Basalt led the port at every tested depth. At the 64K target, roughly 47,100 actual tokens, the lead was **14.5%** (2,288 vs 1,999 tok/s). Both runs used IQ3. [Raw samples and power](benchmarks/results/2026-11-24-n1x-cooled-release)

On FP4-Local, Basalt read prompts faster than native vLLM at four of five depths. At the 32K target, Basalt processed **1,951 input tok/s** versus **2,123** for vLLM. At that depth Basalt still loaded some expert weights from file. vLLM led in four-stream mixed throughput. Busy decode averaged 46.00 W GPU for Basalt and 35.02 W for vLLM. The engines used different KV formats. [FP4-Local measurements](benchmarks/results/2026-11-24-n1x-fp4-resident-ring)

![Median cold input tokens per second at five target prompt depths. This project has three columns, IQ3, FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local.](benchmarks/img/engines-prefill.svg)

<details>
<summary>Decode by scenario, throughput by stream count and test settings</summary>

![Median output tokens per second in eight decode scenarios. This project has three columns, IQ3, FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local.](benchmarks/img/engines-decode.svg)

![Aggregate output tokens per second at one through four streams. This project has three columns, IQ3, FP4-Local and NVFP4-DQ6_K. Reference columns are the Arm64 port with IQ3 and native vLLM with FP4-Local.](benchmarks/img/engines-concurrency.svg)

Basalt used four slots, INT8 KV and MTP 8. The IQ3 runs used the v0.4.1-rtx-spark.2 ZIP and the Arm64 port, averaging 47.5 and 49.2 W GPU respectively during busy decode. Native vLLM ran earlier with FP8 target KV and MTP 7. [IQ3 runs](benchmarks/results/2026-11-24-n1x-cooled-release) · [vLLM](benchmarks/results/2026-11-24-n1x-verified-default/vllm)

Both FP4 packs used the same native source build with all 24,576 GPU expert slots and Windows file caching for the n-gram table. Whole-GPU memory sampled each second during BetterBench had nearly identical medians (72.62 GiB DQ6_K, 72.77 GiB FP4-Local). The sampled peaks were **75.03 GiB** and **78.23 GiB**, respectively. Busy decode averaged 48.39 W GPU for DQ6_K and 46.00 W for FP4-Local. [Measured source](https://github.com/SuperDistance/basalt-rtx-spark/commit/1bc565f277a3fb02c6182ea5ba0e9086b3f637e0) · [DQ6_K telemetry](benchmarks/results/2026-11-24-n1x-nvfp4-dq6-k) · [FP4-Local telemetry](benchmarks/results/2026-11-24-n1x-fp4-resident-ring)

</details>

## One prompt, one orbital transfer

In **123 seconds**, Flash-Next FP4-Local turned one prompt into this animated SVG of a probe transferring between two Earth orbits, complete with burn values and flight time. [Exact prompt, original SVG and physics checks](docs/showcase/orbital-transfer)

[![The Long Way Up, an animated orbital transfer drawn by Flash-Next on RTX Spark.](docs/showcase/orbital-transfer/preview.webp)](docs/showcase/orbital-transfer)

## What the N1X fork speeds up

- **Reading a prompt.** Flash-Next selects a BF16 activation path for short NVFP4 matrix multiplies. A 716-token prompt reached its first token in 0.66 s, down from 1.37 s in the kernel comparison.
- **Keeping FP4 experts on the GPU.** With 2,560 MiB reserved, a smaller staging buffer let a 23,633-token prompt stay in one chunk and avoided loading 1,586 expert blobs from file. The first token arrived in 10.38–10.41 s, down from 14.16–14.20 s (**26.7% less wait**). [Controlled experiment](benchmarks/results/2026-11-24-n1x-fp4-prefill-chunks)
- **Serving four chats.** Flash-Next drafts for several requests in one GPU pass. Drafting took 5.42 ms per verification window versus 21.53 ms on the Arm64 port in the four-stream structured-output test.
- **Joining an active server.** A new 16,672-token prompt produced its first token in 26.17 s while all three ongoing chats were still decoding. Their longest gap was 389 ms. [Implementation and measurements](docs/rtx-spark-optimizations.md)

## Quantization

<details>
<summary>Supported weight packs and measured speeds by prompt type</summary>

The documented Flash-Next weight packs for both Basalt engines include `NVFP4-DQ6_K`, `NVFP4-DQ8`, `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-IQ3_XXS`, `UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`, `Q8_0`. Basalt RTX Spark also completed BetterBench with NVIDIA `FP4-Local` converted to `.basalt`. The vLLM for RTX Spark 0.2.0 recipe for Flash-Next uses `FP4-Local`. The table above gives BetterBench throughput for the packs tested. [Basalt pack catalog](docs/models/qwen38-flash-next.md#pack) · [vLLM release recipes](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md)

The two FP4 packs finished within **0.1 tok/s** of each other for weighted decode in these separate runs. Of these three packs, **UD-IQ3_XXS** had the highest weighted decode rate on this laptop. For closer agreement with BF16, the pack publisher measured **91.65%** top-token agreement for DQ6_K versus **88.67%** for IQ3 on the same 100 WikiText-2 chunks. [Publisher’s quality test](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/README.md#quality)

Earlier N1X single-request tests measured output tokens per second for three prompt types. The ranges span two runs per pack.

| Group | Engine / model | Weight pack | Conversation | Structured output | Prose |
| --- | --- | --- | ---: | ---: | ---: |
| **This project** | Basalt RTX Spark / Flash-Next | `UD-IQ3_XXS` | 79.0–84.1 | 141.3–147.0 | 73.3–75.4 |
| **This project** | Basalt RTX Spark / Flash-Next | `FP4-Local` | 81.6–82.7 | 133.5–136.1 | 68.3–69.2 |
| **This project** | Basalt RTX Spark / Qwen3.8-27B | `QUASAR-NVFP4` | 48.6 | 143.7 | 33.1 |
| **Reference** | Basalt + Arm64 port / Flash-Next | `NVFP4-DQ6_K` | 80.8 | 140.3 | 67.8 |

The alternating Flash-Next IQ3 and FP4-Local runs used INT8 KV, MTP 8 and fully GPU-resident experts. Busy GPU draw was 36.8–42.4 W for IQ3 and 38.4–41.4 W for FP4-Local. The Arm64 port used FP16 KV for its NVFP4-DQ6_K test. The 27B run used INT8 KV and a 16-token block drafter. The other listed packs have no comparable N1X throughput measurement in these runs. [Per-run measurements](benchmarks/results/2026-11-23-n1x-quantization) · [Arm64 port fixtures](docs/arm64.md#speed-on-the-rtx-spark-n1x)

</details>

## Running locally

Basalt places model weights across GPU memory and system RAM automatically. Flash-Next defaults to four slots with up to 262,144 tokens each and an INT8 KV cache that grows on demand. A new cold prompt can join while another conversation is decoding. Chat clients and coding agents can connect through the OpenAI- and Anthropic-compatible APIs. [Setup and client guide](docs/rtx-spark-setup.md) · [Engine changes](docs/rtx-spark-setup.md#what-changed-from-basalt) · [Quality measurements](docs/precision/)

## Credits

- [Basalt](https://github.com/jesdga95/basalt) by Jesus Gulfo and contributors (MIT): the engine this fork builds on.
- [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) (MIT): the Arm64 and compute capability 12.1 port it
  builds on.
- The projects and models Basalt credits in [README.basalt.md](README.basalt.md) and
  [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): Qwen's model, Unsloth's UD-IQ3_XXS quantization, and more.

## License

MIT, see [LICENSE](LICENSE).
