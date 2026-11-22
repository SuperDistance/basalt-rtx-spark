# Basalt on RTX Spark N1X

## Baseline and target

The local fork tracks `yanmindi/basalt-arm-sm121` at `2ec7a844f9d83cda0fa579801f85cd8898a12bec`. The baseline builds as Windows ARM64 with MSVC and CUDA 13.5 for `sm_121`. The test laptop is a 128 GiB HP OmniBook X with driver 621.24. Windows currently reports 64,000 MiB of dedicated GPU memory.

| Machine | Published memory bandwidth |
| --- | ---: |
| RTX Spark N1X | 300 GB/s |
| DGX Spark GB10 | 273 GB/s |

N1X has 9.9% more theoretical memory bandwidth. This is a useful target for kernels limited by sustained memory reads. Serving throughput also depends on inference kernels, speculative acceptance, prefix reuse, memory placement, the operating system, and sustained power. [NVIDIA N1X guide](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/overview.html) · [NVIDIA DGX Spark guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)

| Public reference | Single request | Concurrent |
| --- | ---: | ---: |
| N1X Basalt port, 27B QUASAR | 28.6 tok/s | 72.3 tok/s at C4 |
| N1X Basalt port, Flash-Next | 63.7 tok/s | 80.1 tok/s at C4 |
| One GB10, 27B vLLM DSpark | 38–43 tok/s on code and reasoning | 122.8 tok/s at C8 |
| One GB10, Flash-Next vLLM V2/b12x | 60.4 tok/s at defaults | 136.7 tok/s at C8 |

These results use different prompts, decode settings, speculative drafters, and measurement boundaries. The Flash-Next GB10 source also reports 192.0 tok/s at C8 with greedy coding prompts and thinking disabled. The N1X port documents an additional mixed decode run at about 116 tok/s. Use the published results as reference points. [N1X Basalt results](https://github.com/yanmindi/basalt-arm-sm121/blob/n1x/docs/arm64.md) · [27B GB10 results](https://github.com/kosztyua/Qwen3.8-27B-DGX-Spark#benchmarks) · [Flash-Next GB10 results](https://github.com/ursuciprian/qwen3.8-flash-next-1x-dgx-spark#performance)

An Ubuntu 26.04 N1X vLLM deployment reports about 60 tok/s for Flash-Next and a 445-to-146-second weight-load improvement from pinning the loader to the fast cores. Its Linux allocator and cache workarounds are specific to that stack. CPU placement during loading and model repack peak memory are useful measurements on Windows too. [N1X Linux serving report](https://github.com/LeonMarzollDev/local-llm-rtx-spark#the-fixes-that-matter-on-unified-memory)

A paired comparison needs one unit of each machine, identical Basalt revision and weights where available, matching prompt and output lengths, sampling and speculation, KV format, context and prefix-cache policy. Track median and p95 time to first token, end-to-end and decode-only aggregate output tokens per second, per-request throughput, and memory placement at C1/C2/C4/C8. Record power source, power mode and temperature. The initial local idle sweep uses a 32,768-token context budget, int8 KV, eight slots and at most 160 output tokens per request, with prompt caches disabled.

## Unified memory

The Flash-Next artifact holds 63.3 GiB of routed experts and about 50.7 GiB of Q8_0 per-layer token embeddings. Basalt can source those embeddings from SSD. At the current reservation, the experts need a placement policy that considers both WDDM budgets and Windows commit. Keep the GPU reservation in place during development and compare automatic placement at the same reservation.

A local test opened a 256 MiB read-only region of the 27B model using a Win32 file mapping and registered it as CUDA mapped read-only host memory. GPU copies from three sampled offsets matched the model bytes. System committed memory was 56,640.11 MiB before and after registration. This validates the API path. A timing probe using the same bytes compares sequential, shuffled and repeated reads from dedicated device memory, ordinary CUDA host allocations, private registered memory at two alignments, and the file-backed mapping. It checks the GPU checksum in each case. [NVIDIA CUDA UMA guidance](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

The CUDA guide describes different L2 behavior for CUDA host allocations and suitably aligned registered memory. Use the measured file-backed bandwidth and the size of the hot expert working set to choose between a direct GPU read, a staged expert, and CPU execution. For a first engine experiment, cap the amount registered, fall back to normal staging when registration is refused, and preserve placement generations and mapping lifetime across adaptive swaps and sleep.

## CPU experts and quantization

An affinity-pinned `SMMLA` probe executed on each of the 20 cores. Basalt detects Arm I8MM and its current ARM NVFP4/Q8_0 dot product uses SDOT. A two-activation, two-output-row I8MM probe matched the existing SDOT results bit for bit across 1,024 rows at widths 512, 2,048 and 3,072. The probe is independent of the engine. Time small expert batches next, then compare end-to-end prefill overlap across the CPU clusters. Avoid spending GPU bandwidth on activations for CPU work if measured gather cost and overlap favor a smaller transfer.

Radiance IQR-V4 is a reference for selecting per-band precision by measured quality loss, including the output head. Basalt already ships mixed precision and separate FP8 and NVFP4 head measurements. A new head experiment should retain the existing weights as a control, gate on KLD and top-1 agreement against the higher-precision reference, and time the fused kernels at C1 and C8. [Radiance IQR-V4 report](https://radiance.stilldeadco.de/whitepapers/iqr-v4.pdf) · [Basalt 27B precision](https://github.com/jesdga95/basalt/blob/main/docs/precision/qwen38-27b.md)

## NPU

Local cuDLA 1.5.0 reported one DLA, hardware version 2. Creating and destroying a hybrid device succeeded. Creating a standalone device returned `cudlaErrorUnsupportedOperation`. cuDLA reported no unified addressing for the DLA handle. This probe exercises the device API. Model execution and speed remain to be measured. [Community probe](https://github.com/roberthendriksen/rtx-spark-npu/blob/main/src/01_find_npu.py)

XDA ran FP16 convolutions, fixed-shape attention with RoPE, and concurrent GPU and DLA work. Its fixed-shape 1×1 projection ran well below cuBLAS throughput. NVIDIA documents FP16 computation, fixed dimensions and batch size one for the Windows ARM compiler. AMD’s Ryzen AI demonstrates a hybrid LLM deployment through its own ONNX Runtime GenAI stack. [XDA experiments](https://www.xda-developers.com/nvidia-rtx-spark-npu-barely-mentions-spent-week-figuring-out/) · [NVIDIA DLA operation support](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-windows-on-arm.html) · [AMD Ryzen AI LLM deployment](https://ryzenai.docs.amd.com/en/1.5/llm/overview.html)

A first workload would compare a known fixed-shape DLA graph with the equivalent GPU result, then measure it alongside the unchanged Basalt baseline. Record DLA execution, combined throughput, Qwen p95 latency, and energy per output token if system-level power is available. A small fixed-shape drafter can be tested if its operations compile and end-to-end speculative acceptance justifies the transfer. The Windows ARM TensorRT compiler download from NVIDIA requires license acceptance. [TensorRT download](https://developer.nvidia.com/tensorrt/download/11x)
