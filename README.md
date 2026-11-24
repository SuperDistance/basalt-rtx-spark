# basalt-rtx-spark

### A 120B-class model at 90.79 tokens/s on your RTX Spark laptop.

Native on Windows on Arm. Lossless speedups. Ready for your chat app and your coding agent.

[Install](#install) · [Use it](#use-it-from-your-tools) · [Hardware](#hardware) · [Results](#results) · [Benchmarks](benchmarks) · [Docs](#documentation)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Windows on Arm64](https://img.shields.io/badge/Windows-Arm64-0078D4)
![RTX Spark N1X](https://img.shields.io/badge/NVIDIA-RTX%20Spark%20N1X-76B900)

**Run Qwen3.8-Flash-Next, a ~120B-parameter mixture-of-experts model, locally on an NVIDIA RTX Spark laptop** with
**90.79 tok/s** single-stream decode, a **289 ms** first token and **110.50 tok/s** across four parallel requests
in the November 24 [BetterBench run](#results). The goal: a capable large model fast enough at a laptop's power draw that
local LLM workflows (chat, documents, coding agents) feel seamless, with nothing leaving the machine.

It is a fork of the [Basalt](https://github.com/jesdga95/basalt) inference engine, built on its Arm64 port and tuned
for the RTX Spark:

- **Native on Windows on Arm64.** No Linux, no WSL: a Windows process, with the Windows-specific work done (memory
  placement that respects Windows commit, fast reads of the model's tables on NTFS, millisecond command handling).
- **Made for unified memory.** It spreads the model across the GPU's share of memory and pinned system RAM by itself,
  at whatever split your laptop has, and the GPU reads RAM-resident experts in place instead of copying them.
- **Lossless speedups.** Every speedup keeps the output: the same greedy text as before it (checked by hashes), the KLD
  gate against the BF16 model where rounding changes, and speculative decoding that keeps the sampling distribution.
  The quantization is a separate choice: the UD-IQ3_XXS pack measured here has a mean KLD of 0.118 against the BF16
  model (88.7% top-1 agreement); NVIDIA's NVFP4 packs 0.06-0.09.
- **Long context, many requests.** Four 262,144-token conversations at once, with an int8 KV cache that grows on demand.
- **Drop-in API.** OpenAI- and Anthropic-compatible endpoints, so existing clients and agents work unchanged.

## Install

With a source checkout, serving Flash-Next on one RTX Spark N1X automatically configures four full-context slots, 262,144 tokens per slot and INT8 KV. A long cold prompt can join an active decode. Use `--parallel 1` for single-request serving or pass your own slot settings.

**Prebuilt (easiest).** Download `basalt-...-windows-arm64.zip` from [Releases](https://github.com/SuperDistance/basalt-rtx-spark/releases),
unzip it, run `setup.cmd` (it makes the Python environment and offers to download the model), then `start.cmd`. The
CUDA libraries the engine needs are in the zip; a current NVIDIA driver is enough.

The release ZIP may predate the engine used for the performance results below. To reproduce them, build from this
source using the instructions below. The [benchmark record](benchmarks/results/2026-11-24-n1x-verified-default) identifies the tested binary.

**Docker (Linux hosts: DGX Spark, RTX Spark under Linux).** `docker compose up -d` builds the image for the host's
architecture (Arm64: sm_121) and serves with the Spark recipes' flags; the mounts and the unified-memory notes are in
[docs/docker.md](docs/docker.md).

**From source.** You need the hardware and software in [Hardware](#hardware). Then, from a Developer PowerShell for
Visual Studio:

**1. Get the CUDA toolkit and the source**

```powershell
git clone https://github.com/SuperDistance/basalt-rtx-spark
cd basalt-rtx-spark
python tools\port\fetch_cuda.py windows-arm64 C:\cuda-13.5
```

`fetch_cuda.py` assembles CUDA 13.5 from NVIDIA's redistributable archives (checked against NVIDIA's SHA-256), with no
installer and no driver change.

**2. Build**

```powershell
tools\port\build-win-arm64.cmd build C:\cuda-13.5
pip install -r requirements.txt
```

**3. Download the model** (about 82 GiB)

```powershell
pip install -U huggingface_hub
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-IQ3_XXS.basalt --local-dir models
```

**Run it.** Save this as `config.json`:

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-UD-IQ3_XXS.basalt", "alias": "flash-next", "kv": "int8", "parallel": 4,
               "max-context": 262144, "slot-full-context": true, "gpu": "0" }],
  "engine": { "spec": 8 },
  "server": { "port": 8080 }
}
```

```powershell
$env:PATH = "C:\cuda-13.5\bin\arm64;C:\cuda-13.5\bin;$env:PATH"
bin\basalt.exe serve --config config.json
```

The first start writes a copy of the model's n-gram table next to it (about 29 GB, once). The server is ready when it
prints `weights loaded`; open `http://127.0.0.1:8080/` for its web UI and live stats. Build details and tests:
[Arm64 build guide](docs/arm64.md#building).

## Use it from your tools

| Client speaks | Point it at |
| --- | --- |
| OpenAI API (`/v1/chat/completions`, `/v1/responses`) | base URL `http://127.0.0.1:8080/v1`, model `flash-next`, any API key |
| Anthropic API (`/v1/messages`) | base URL `http://127.0.0.1:8080`, model `flash-next` |

That covers most chat front ends, IDE assistants and coding agents that let you set a base URL. Thinking, tool calls
and structured output (JSON schema) work as in Basalt; images and documents with its `--vision` option
([running](docs/running.md)).

## Hardware

| | Tested | Notes |
| --- | --- | --- |
| Machine | HP OmniBook X with RTX Spark N1X, 128 GB | other N1X laptops should work; results welcome in [benchmarks](benchmarks) |
| GPU memory split | 96 GB | automatic expert placement adapts to the available GPU memory |
| OS | Windows 11 on Arm64 (build 28000) | Linux Arm64 builds too (from the port), untested here |
| Driver | NVIDIA 621.24 | |
| Disk | ~120 GB free | the pack (88 GB) plus its n-gram table copy (29 GB) |
| Build tools | Visual Studio 2026 (or Build Tools) with ARM64 C++, Python 3.10+ | CMake and Ninja come with Visual Studio |

GPU clocks and power vary during sustained workloads. The comparison below gives the observed draw for each run. [GPU power measurements](docs/n1x-gpu-power.md)

## Results

All three engines ran Qwen3.8-Flash-Next on the same RTX Spark N1X laptop with 128 GB of unified memory and a 96 GB GPU allocation. BetterBench 0.6.1 used the same 392-request workload in each run.

| Engine | Weighted single-stream decode (tok/s) | Short-prompt time to first token (ms) | Four-stream mixed throughput (tok/s) |
| --- | ---: | ---: | ---: |
| Basalt RTX Spark | **90.79** | 289 | **110.50** |
| Basalt + Arm64 port | 87.52 | 622 | 85.32 |
| vLLM for RTX Spark 0.2.0 | 54.19 | **271** | 100.75 |

Both Basalt engines used UD-IQ3_XXS weights, INT8 KV and MTP 8. vLLM used FP4-Local weights, FP8 target KV, BF16 draft KV and MTP 7. On November 24, busy GPU draw averaged 39.3 W for Basalt RTX Spark and 34.5 W for vLLM during the four-stream test. The Arm64 port ran separately on November 23 and averaged 40.8 W across busy samples over its full run. The driver did not report a GPU power limit. [November 24 measurements](benchmarks/results/2026-11-24-n1x-verified-default) · [Arm64 port measurements](benchmarks/results/2026-11-23-hp-omnibook-x-n1x-128gb/basalt-arm-sm121-2ec7a84.json)

![Single-stream decode by scenario · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](benchmarks/img/engines-decode.svg)

![Aggregate throughput at one to four streams · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](benchmarks/img/engines-concurrency.svg)

![Cold prefill throughput by prompt depth · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](benchmarks/img/engines-prefill.svg)

**Have an RTX Spark (or a DGX Spark)?** Run the [benchmark](benchmarks) and send your numbers.

## What changed from Basalt

- **Memory:** expert placement budgets the GPU, pinned RAM and Windows commit together (counting the page file's room
  to grow), so every routed expert of the IQ3 pack gets a GPU home at the 96 GB split; RAM-pinned experts are read in
  place in prompts and in decode.
- **Decode:** batch slots draft in one pass and draw their MTP drafts from the draft head at every concurrency; a
  slot's GDN state rows are committed by its next window; the CPU pool sleeps when idle.
- **Prompts:** the n-gram table read from its own unbuffered copy; FP16 copies of the dense projections in VRAM;
  faster long-context indexer scores, int8 sparse attention and GDN recurrence kernels; cold prompts read beside
  decoding requests; the prompt's memory loan sized to what it uses.
- **Requests:** commands answered within a millisecond; conversation checkpoints copied in 4 ms instead of 30, so a
  resumed thinking turn restores in 8-13 ms; batch admission moves checkpoints instead of copying them.

The measurements behind each: [performance notes](docs/n1x-performance.md), [GPU-first placement](docs/n1x-gpu-first.md),
[prompt path](docs/n1x-prompt-path.md).

## Documentation

- **This fork:** [Arm64 build guide](docs/arm64.md), [benchmarks](benchmarks), [GPU power](docs/n1x-gpu-power.md),
  [against vLLM](docs/n1x-vs-vllm.md), [performance notes](docs/n1x-performance.md).
- **Basalt's:** [README](README.basalt.md), [running](docs/running.md), [configuration](docs/configuration/index.md),
  [Flash-Next](docs/models/qwen38-flash-next.md), [precision](docs/precision/index.md), [the `.basalt` file](docs/model-file.md),
  [contributing](CONTRIBUTING.md).

## Credits

- [Basalt](https://github.com/jesdga95/basalt) by Jesus Gulfo and contributors (MIT): the engine this fork builds on.
- [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) (MIT): the Arm64 and compute capability 12.1 port it
  builds on.
- The projects and models Basalt credits in [README.basalt.md](README.basalt.md) and
  [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES): Qwen's model, Unsloth's UD-IQ3_XXS quantization, and more.

## License

MIT, see [LICENSE](LICENSE).
