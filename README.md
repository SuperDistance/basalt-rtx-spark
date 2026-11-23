# basalt-rtx-spark

A fork of the [Basalt](https://github.com/jesdga95/basalt) inference engine tuned for the **NVIDIA RTX Spark** (N1X:
Arm64 CPU with an integrated Blackwell GPU, compute capability 12.1), running Qwen3.8-Flash-Next natively on Windows on
Arm64. It serves an OpenAI- and Anthropic-compatible API, like Basalt.

Changes are lossless: the same greedy text as before the change (checked by output hashes), the KLD gate against the
BF16 model for the few that change rounding, and speculative sampling that keeps the sampling distribution.

## Results

Same laptop, same model pack, same settings, run back to back with BetterBench 0.6.1 (the
config of the community runs: temperature 0.7, cold prefix cache, 20 passes per category, 48 requests per concurrency
level). Qwen3.8-Flash-Next UD-IQ3_XXS, int8 KV, four 262,144-token slots, MTP window 8, 96 GB GPU split. The GPU drew
41-42 W on average while busy in both runs, at about 2.07 GHz.

| | Basalt + Arm64 port (2ec7a84) | basalt-rtx-spark | |
| --- | ---: | ---: | ---: |
| Single-stream decode, weighted median (tok/s) | 87.5 | **97.9** | +12% |
| First token, short prompts, p50 (ms) | 622 | **284** | 2.2x faster |
| Aggregate output, 1 stream (tok/s) | 74.6 | **86.5** | +16% |
| Aggregate output, 2 streams (tok/s) | 81.1 | **99.4** | +23% |
| Aggregate output, 3 streams (tok/s) | 85.7 | **110.8** | +29% |
| Aggregate output, 4 streams (tok/s) | 85.3 | **113.1** | +33% |
| First token at 4 streams, p50 (ms) | 727 | **430** | 1.7x faster |
| Prefill at 2K tokens (tok/s) | 983 | **1,724** | +75% |
| Prefill at 8K tokens (tok/s) | 1,506 | **2,077** | +38% |
| Prefill at 16K / 32K / 64K tokens (tok/s) | 2,055 / 1,951 / 2,095 | **2,149 / 2,295 / 2,342** | +5% / +18% / +12% |

Against vLLM for RTX Spark (FP4-Local, MTP 7) on the same laptop and benchmark (an earlier build, both without the
sustained-power controller), single-stream decode was 60% faster (88.2 against 55.1 tok/s) and so was output at one
stream; vLLM led on short-prompt first token, at three and four streams, and on prefill. The details, and why the GPU's power state matters for any
comparison on this laptop, are in [docs/n1x-vs-vllm.md](docs/n1x-vs-vllm.md) and [docs/n1x-gpu-power.md](docs/n1x-gpu-power.md).

## What changed

On top of Basalt and the Arm64 / sm_121 port:

- **Memory placement on a unified-memory laptop.** Expert placement budgets the GPU, private RAM and Windows commit
  together (counting the page file's room to grow), so every routed expert of the IQ3 pack gets a GPU home; experts
  pinned in RAM are read in place by the GPU, in prompt reads and in decode windows.
- **Decode.** Batch slots draft in one pass and draw their MTP drafts from the draft head (speculative sampling, at
  every concurrency); a slot's GDN state rows are committed by its next window; the CPU pool sleeps when parked.
- **Prompt path.** The n-gram embedding table read from its own unbuffered copy; FP16 copies of the dense projections
  kept in VRAM; faster tensor-core indexer scores, int8 sparse attention (two blocks per SM) and GDN recurrence; a
  cold prompt read beside decoding slots; the prompt's expert loan sized to what it stages.
- **Requests.** The engine answers the server's commands within a millisecond on Windows; conversation checkpoints
  copy in 4 ms instead of 30 (a resumed thinking turn restores in 8-13 ms); batch admission moves checkpoints instead
  of copying them.
- **Conversion.** NVFP4 experts with CSF-encoded block scales, and FP4-Local's norms and NVFP4 n-gram table.

The measurements behind each change: [performance notes](docs/n1x-performance.md),
[GPU-first placement](docs/n1x-gpu-first.md), [prompt path](docs/n1x-prompt-path.md),
[GPU power](docs/n1x-gpu-power.md).

## Building and running

Tested on Windows 11 on Arm64 (RTX Spark N1X, 128 GB, 96 GB GPU split) with MSVC, CUDA 13.5 and `sm_121`. The Linux
Arm64 build of the port is kept but untested here. The build guide is [docs/arm64.md](docs/arm64.md#building)
(`tools/port/fetch_cuda.py` assembles the CUDA toolkit; `tools/port/build-win-arm64.cmd` builds).

A server config like the one measured above:

```json
{
  "version": 2,
  "models": [{ "model": "Qwen3.8-FN-UD-IQ3_XXS.basalt", "alias": "flash-next", "kv": "int8", "parallel": 4,
               "max-context": 262144, "slot-full-context": true, "gpu": "0" }],
  "engine": { "spec": 8 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

```
basalt serve --config config.json --python <python with requirements.txt>
```

The rest of the documentation is Basalt's: [README.basalt.md](README.basalt.md), [running](docs/running.md),
[configuration](docs/configuration), [model files](docs/model-file.md).

## Credits and license

- [Basalt](https://github.com/jesdga95/basalt) by Jesus Gulfo and contributors (MIT): the engine this fork is built on.
- [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) (MIT): the Arm64 and compute capability 12.1 port
  this work builds on.
- The projects Basalt credits in [README.basalt.md](README.basalt.md) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).

MIT License, see [LICENSE](LICENSE).
