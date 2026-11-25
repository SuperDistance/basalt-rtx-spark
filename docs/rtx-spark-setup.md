# Set up Basalt on an RTX Spark N1X laptop

The Windows release runs natively on Arm64 and targets the N1X integrated GPU at CUDA compute capability 12.1 (`sm_121`).

## Install

With a source checkout, serving Flash-Next on one RTX Spark N1X automatically configures four full-context slots, 262,144 tokens per slot and INT8 KV. A long cold prompt can join an active decode. Use `--parallel 1` for single-request serving or pass your own slot settings.

**Windows Arm64 release.** With an up-to-date NVIDIA driver, the ARM64 Visual C++ runtime and Python 3.10 or later installed, download the Windows Arm64 ZIP and extract it. Run `setup.cmd` to create the Python environment and download the Flash-Next pack, then `start.cmd` to serve it. The ZIP includes the CUDA runtime libraries. [Latest release](https://github.com/SuperDistance/basalt-rtx-spark/releases/latest)

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
[Arm64 build guide](arm64.md#building).

## Use it from your tools

| Client speaks | Point it at |
| --- | --- |
| OpenAI API (`/v1/chat/completions`, `/v1/responses`) | base URL `http://127.0.0.1:8080/v1`, model `flash-next`, any API key |
| Anthropic API (`/v1/messages`) | base URL `http://127.0.0.1:8080`, model `flash-next` |

That covers most chat front ends, IDE assistants and coding agents that let you set a base URL. Thinking, tool calls
and structured output (JSON schema) work as in Basalt; images and documents with its `--vision` option
([running](running.md)).

## Hardware

| | Tested | Notes |
| --- | --- | --- |
| Machine | HP OmniBook X with RTX Spark N1X, 128 GB | other N1X laptops should work; results welcome in [benchmarks](../benchmarks) |
| GPU memory split | 96 GB | automatic expert placement adapts to the available GPU memory |
| OS | Windows 11 on Arm64 (build 28000) | native Arm64 build and release |
| Driver | NVIDIA 621.24 | |
| Disk | ~120 GB free | the pack (88 GB) plus its n-gram table copy (29 GB) |
| Build tools | Visual Studio 2026 (or Build Tools) with ARM64 C++, Python 3.10+ | CMake and Ninja come with Visual Studio |

GPU clocks and power vary during sustained workloads. The [benchmark comparison](../README.md#results) gives the observed draw for each run. [GPU power measurements](n1x-gpu-power.md)

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

The measurements behind each: [performance notes](n1x-performance.md), [GPU-first placement](n1x-gpu-first.md),
[prompt path](n1x-prompt-path.md).

## Choosing a memory split on the 128 GiB laptop

For mixed Windows and inference use, try **80 GiB GPU / 48 GiB host** if your laptop offers it. With 96 GiB assigned to the GPU, device-wide usage peaked at **68.0 GiB for IQ3**. FP4-Local peaked at **79.3 GiB** on the release and **78.2 GiB** on the newer build. An 80 GiB split leaves about 12 GiB above the IQ3 peak and only 0.7–1.8 GiB above those FP4 peaks. Throughput at 80 GiB and four near-full contexts at that split still need measurement. Choose 96/32 for FP4 benchmark runs or 64/64 when Windows needs more than 48 GiB. [IQ3 telemetry](../benchmarks/results/2026-11-24-n1x-cooled-release/release/gpu.csv) · [FP4 release](../benchmarks/results/2026-11-24-n1x-fp4-local/gpu.csv) · [FP4 source build](../benchmarks/results/2026-11-24-n1x-fp4-resident-ring/gpu.csv)

On the same earlier IQ3 build, 64 vs 96 GiB for the GPU delivered **74.0 vs 91.9 tok/s** single-stream decode and **697 vs 1,218 input tok/s** at the 2K cold-prefill target. Power varied between the runs. Windows manages dedicated and shared memory as separate budgets even though the CPU and GPU share physical memory. [Split measurements](n1x-vs-vllm.md) · [NVIDIA UMA guide](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

## Documentation

[Contributing to Basalt RTX Spark](../CONTRIBUTING.md) · [Optional Linux container](docker.md)

- **This fork:** [Arm64 build guide](arm64.md), [benchmarks](../benchmarks), [GPU power](n1x-gpu-power.md),
  [against vLLM](n1x-vs-vllm.md), [performance notes](n1x-performance.md).
- **Upstream Basalt:** [README](../README.basalt.md), [running](running.md), [configuration](configuration/index.md),
  [Flash-Next](models/qwen38-flash-next.md), [precision](precision/index.md), [the `.basalt` file](model-file.md).
