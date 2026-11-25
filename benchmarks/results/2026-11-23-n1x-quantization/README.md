# Single-stream decode by weight pack

The November 23 N1X single-stream test (UD-IQ3_XXS, FP4-Local, FP4-Local, UD-IQ3_XXS) kept all 24,576 experts resident and 1 GiB of host copies in
each arm. IQ3 used 46.84 GiB for routed weights and FP4-Local used 63.29 GiB. INT8 KV, seed 17, MTP8,
context 16,720 and the fixtures were held fixed. The power limit was unavailable.

| This project · Basalt RTX Spark pack in run order | Conv / struct / prose tok/s | Conv / struct / prose ms per window | Cold 16K tok/s | Busy GPU W |
| --- | --- | --- | ---: | ---: |
| IQ3 | 79.0 / 141.3 / 75.4 | 33.54 / 51.89 / 29.06 | 1,789 | 36.8 |
| FP4-Local | 82.7 / 136.1 / 68.3 | 35.86 / 57.11 / 33.11 | 1,774 | 41.4 |
| FP4-Local | 81.6 / 133.5 / 69.2 | 36.35 / 58.23 / 32.67 | 1,844 | 38.4 |
| IQ3 | 84.1 / 147.0 / 73.3 | 31.52 / 49.87 / 29.90 | 1,627 | 42.4 |

FP4-Local's conversation window committed 2.99 tokens versus IQ3's 2.68, partly offsetting its longer
window. Both emitted 7,230 structured tokens. IQ3 used its existing PLE sidecar for direct reads. FP4-Local
used the OS file cache because creating its 50.7 GiB sidecar exceeded free disk space. This compares the
local packs in their attainable configurations.

### How the NVFP4-DQ6_K pack compares

The published DQ6_K pack and the FP4-Local conversion each store **67,948,118,016 bytes (63.28 GiB)** in the routed-expert section. Their `dense`, `gguf` and `mtp/dense` sections also have matching sizes. The expert and main dense payloads have different SHA-256 digests. Switching between these two packs offers little reason to expect a lower weight footprint. The IQ3 routed experts use **46.84 GiB**. Peak device use also includes runtime allocations. [Published DQ6_K pack](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/Qwen3.8-FN-NVFP4-DQ6_K.basalt)

On an earlier N1X build at a roughly 64 GiB GPU split, DQ6_K reached **70.9 and 64.3 tok/s** on the 717-token single-stream decode test. IQ3 reached **80.4 tok/s** with all experts resident. DQ6_K held 20.1 GiB of experts in host RAM. On a later 96 GiB GPU split, the full 392-request BetterBench run placed all 24,576 DQ6_K expert slots on the GPU and recorded 85.22 weighted single-stream tok/s. [Current-build DQ6_K measurements](../2026-11-24-n1x-nvfp4-dq6-k) [Earlier placement and timings](../../../docs/n1x-gpu-first.md#nvfp4-against-iq3_xxs)

DQ6_K has a measured advantage in fidelity. In the pack publisher’s 100-chunk WikiText-2 test, the top token agreed with BF16 **91.65%** of the time for DQ6_K and **88.67%** for IQ3. This is a reference for choosing a pack when fidelity matters more than memory headroom. [Publisher’s test method and results](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/README.md#quality)
