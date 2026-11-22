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

A paired comparison needs one unit of each machine, identical Basalt revision and weights where available, matching prompt and output lengths, sampling and speculation, KV format, context and prefix-cache policy. Track median and p95 time to first token, end-to-end and decode-only aggregate output tokens per second, per-request throughput, and memory placement at C1/C2/C4/C8. Record power source, power mode and temperature. Use `tools/port/spark_serve_bench.py --port PORT --out FILE` against the same local server configuration on each Spark. The replay client was checked with a local SSE endpoint at C1/C2/C4/C8. The initial 32,768-token/eight-slot configurations hit the Windows memory limits on the laptop. Windows later expanded its system-managed pagefile during the test session, so each result belongs to the available memory at its start.

## Local results

These are short AC-powered tests with normal desktop activity and the existing 62.5 GiB GPU reservation. The prompts used about 717 input tokens and generated 160 output tokens. Throughput is aggregate end-to-end output tokens per second, including time spent reading the prompt.

| Model and configuration | C1 | C2 | C4 | C8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B, upstream native build, eight 2,048-token slots | 27.48 | 40.96 | 61.70 | 71.84 |
| Flash-Next, upstream native build, mapped-file experts | 7.85 | 9.18 | RAM guard stopped requests | |
| Flash-Next, GPU-first under commit pressure | 9.77 | 10.17 | | |
| Flash-Next, GPU-first with bounded RAM residency and I8MM, four slots | 10.04 | 9.92 | 14.53 | |
| Flash-Next, same patches, eight slots with more Windows commit available | | | | 38.31 |
| Flash-Next, automatic placement with default memory settings | 18.72 | | | 26.43 |

Qwen3.8-27B median time to first token was 0.41, 0.93, 1.67 and 2.95 seconds from C1 to C8. At C8, p95 was 4.99 seconds over 24 requests. Flash-Next four-slot runs used 4,096 tokens per slot, int8 KV and two waves per measured concurrency. The patched Flash-Next C4 completed eight requests with median TTFT of 17.49 seconds and p95 of 25.32 seconds. At eight slots after pagefile growth, Flash-Next completed 16 requests with median TTFT of 10.38 seconds and p95 of 17.98 seconds. The server reported 47.5 GiB of loaded model data and a 20.9 GiB file-backed expert tier. The automatic build completed 16 eight-way requests with median TTFT of 14.50 seconds and p95 of 24.61 seconds. It loaded 29.1 GiB of model data at startup, versus 47.5 GiB in the earlier experimental eight-slot run. Windows commit, available RAM and the file cache varied between runs.

The first auto-placement attempt reserved almost 50 GiB in anticipated host mapping costs while the combined available budget was about 14 GiB. The `n1x_commit_cache` experiment prioritizes the GPU cache under Windows commit pressure. It started Flash-Next at the existing reservation. A subsequent fallback fix clamps the size of the ranked resident CPU complement to both RAM and commit. It successfully selected and loaded 0.64 GiB of hot experts when 0.65 GiB remained above the safety reserve. Commit a700674 made GPU-first placement automatic for the default single-GPU resident path when commit is tighter than device memory. The verification run used the existing GPU reservation and default Basalt memory settings.

## Unified memory

The Flash-Next artifact holds 63.3 GiB of routed experts and about 50.7 GiB of Q8_0 per-layer token embeddings. Basalt can source those embeddings from SSD. At the current reservation, the experts need a placement policy that considers both WDDM budgets and Windows commit. Keep the GPU reservation in place during development and compare automatic placement at the same reservation.

A local test opened a 256 MiB read-only region of the 27B model using a Win32 file mapping and registered it as CUDA mapped read-only host memory. GPU copies from three sampled offsets matched the model bytes. System committed memory was 56,640.11 MiB before and after registration. This validates the API path. A timing probe using the same bytes compares sequential, shuffled and repeated reads from dedicated device memory, ordinary CUDA host allocations, private registered memory at two alignments, and the file-backed mapping. GPU checksums matched across all placements. Large sequential and shuffled reads reached 262?268 GB/s on device allocations, 261?271 GB/s on CUDA mapped host allocations and 150?154 GB/s on the file-backed mapping. In these tests, mapped files bought capacity and avoided an anonymous copy, while CUDA mapped host allocations delivered streaming bandwidth close to device allocations. [NVIDIA CUDA UMA guidance](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

The CUDA guide describes different L2 behavior for CUDA host allocations and suitably aligned registered memory. Use the measured file-backed bandwidth and the size of the hot expert working set to choose between a direct GPU read, a staged expert, and CPU execution. For a first engine experiment, cap the amount registered, fall back to normal staging when registration is refused, and preserve placement generations and mapping lifetime across adaptive swaps and sleep.

## CPU experts and quantization

An affinity-pinned `SMMLA` probe executed on each of the 20 cores. Basalt detects Arm I8MM and its current ARM NVFP4/Q8_0 dot product uses SDOT. A two-activation, two-output-row I8MM probe matched the existing SDOT results bit for bit across 1,024 rows at widths 512, 2,048 and 3,072. The paired dot kernels ran 2.29?2.83? faster in the synthetic microbenchmark. The engine enables the gate/up and down variants through `--exp n1x_i8mm=1`. Native wrapper parity passed 40 cases covering odd token and row counts and both SwiGLU modes. Serving gains still depend on the fraction of experts that run on the CPU. Avoid spending GPU bandwidth on activations for CPU work if measured gather cost and overlap favor a smaller transfer.

Radiance IQR-V4 is a reference for selecting per-band precision by measured quality loss, including the output head. Basalt already ships mixed precision and separate FP8 and NVFP4 head measurements. For 27B, Basalt reports mean KLD of 0.0262 and 92.82% top-1 agreement for its FP8 head, versus 0.0312 and 91.83% for the NVFP4 head. Use those as quality anchors for frequency-weighted precision bands and time prefill and concurrent decode. For Flash-Next, run per-layer sensitivity and routing calibration first. Its 63.3 GiB routed-expert tier is the larger target, and the existing pack format supports different quantizers across layers. [Radiance IQR-V4 report](https://radiance.stilldeadco.de/whitepapers/iqr-v4.pdf) · [Basalt 27B precision](https://github.com/jesdga95/basalt/blob/main/docs/precision/qwen38-27b.md)

## NPU

Local cuDLA 1.5.0 reported one DLA, hardware version 2. Creating and destroying a hybrid device succeeded. Creating a standalone device returned `cudlaErrorUnsupportedOperation`. cuDLA reported no unified addressing for the DLA handle. A native CUDA allocation of 64 KiB also registered and unregistered successfully through cuDLA. Both probes exercise the device API. Model execution and speed remain to be measured. [Community probe](https://github.com/roberthendriksen/rtx-spark-npu/blob/main/src/01_find_npu.py)

XDA ran FP16 convolutions, fixed-shape attention with RoPE, and concurrent GPU and DLA work. Its fixed-shape 1×1 projection ran well below cuBLAS throughput. NVIDIA documents FP16 computation, fixed dimensions and batch size one for the Windows ARM compiler. AMD’s Ryzen AI demonstrates a hybrid LLM deployment through its own ONNX Runtime GenAI stack. [XDA experiments](https://www.xda-developers.com/nvidia-rtx-spark-npu-barely-mentions-spent-week-figuring-out/) · [NVIDIA DLA operation support](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-windows-on-arm.html) · [AMD Ryzen AI LLM deployment](https://ryzenai.docs.amd.com/en/1.5/llm/overview.html)

A first workload would compare a known fixed-shape DLA graph with the equivalent GPU result, then measure it alongside the unchanged Basalt baseline. Record DLA execution, combined throughput, Qwen p95 latency, and energy per output token if system-level power is available. A small fixed-shape drafter can be tested if its operations compile and end-to-end speculative acceptance justifies the transfer. The Windows ARM TensorRT compiler download from NVIDIA requires license acceptance. [TensorRT download](https://developer.nvidia.com/tensorrt/download/11x)
