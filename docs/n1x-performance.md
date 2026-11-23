# Basalt on RTX Spark N1X

## Laptop serving target

Run Qwen3.8-27B and Flash-Next natively on Windows ARM64, at the laptop's existing memory split. The application should choose its expert placement automatically from the GPU budget, free system RAM and Windows commit. Give frequently used weights a GPU home where the budget allows, and let eligible work access shared RAM directly. Measure the cost of each placement, since the memory segments differ in page size, caching and how the driver accounts for them. [NVIDIA CUDA UMA guidance](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

Prioritize C1 decode tokens/s, cold prefill tokens/s, and time to first token. Then measure C2, C3 and C4, reporting per-request latency as well as aggregate throughput. Use C8 as an optional capacity and stability check. Compare against GB10 with matched workloads and evaluate CPU or NPU offload by its effect on the full request.

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

A community native Windows ARM64 vLLM build reports 65.75 tok/s at C1 and 233.63 tok/s at C16 on its mixed Flash-Next workload. Its configuration specifies a 96 GB GPU split, a larger fixed minimum for the Windows page file, NVFP4 experts, and SSD-backed PLE. The throughput runner counts streamed tokens inside two 20-second windows after warm-up, with prefix caching enabled. The local Basalt fixtures use the existing 62.5 GiB reservation, different weights and end-to-end completion timing. [Native vLLM results and setup](https://github.com/devjeonghwan/vllm-for-rtx-spark#benchmarks) · [Workload method](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/main/benchmarks/README.md#workload-throughput)

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
| Flash-Next NVFP4-DQ6_K, automatic placement with default memory settings | 18.72 | | | 26.43 |
| Flash-Next UD-IQ3_XXS, automatic placement with default memory settings | 25.42 | | | 33.18 |

Qwen3.8-27B median time to first token was 0.41, 0.93, 1.67 and 2.95 seconds from C1 to C8. At C8, p95 was 4.99 seconds over 24 requests. Flash-Next four-slot runs used 4,096 tokens per slot, int8 KV and two waves per measured concurrency. The patched Flash-Next C4 completed eight requests with median TTFT of 17.49 seconds and p95 of 25.32 seconds. At eight slots after pagefile growth, Flash-Next completed 16 requests with median TTFT of 10.38 seconds and p95 of 17.98 seconds. The server reported 47.5 GiB of loaded model data and a 20.9 GiB file-backed expert tier. The automatic build completed 16 eight-way requests with median TTFT of 14.50 seconds and p95 of 24.61 seconds. It loaded 29.1 GiB of model data at startup, versus 47.5 GiB in the earlier experimental eight-slot run. Windows commit, available RAM and the file cache varied between runs.

The first auto-placement attempt reserved almost 50 GiB in anticipated host mapping costs while the combined available budget was about 14 GiB. The `n1x_commit_cache` experiment prioritizes the GPU cache under Windows commit pressure. It started Flash-Next at the existing reservation. A subsequent fallback fix clamps the size of the ranked resident CPU complement to both RAM and commit. It successfully selected and loaded 0.64 GiB of hot experts when 0.65 GiB remained above the safety reserve. Commit a700674 made GPU-first placement automatic for the default single-GPU resident path when commit is tighter than device memory. The verification run used the existing GPU reservation and default Basalt memory settings.

UD-IQ3_XXS completed two serial requests and 16 requests at C8, with p95 first-token latency of 2.31 and 19.83 seconds, respectively. It used eight 4,096-token slots and int8 KV. The engine initially kept 24.7 GiB of experts on the GPU and 0.5 GiB in RAM, then added 21.6 GiB to the RAM tier. All 46.8 GiB of routed experts had a home in memory before serving requests. Relative to the earlier automatic NVFP4 run, this was 26% higher C8 throughput and 19% lower p95 first-token latency. The two runs used different model quantization, and Windows memory availability can vary. Output quality and sustained throughput still need separate validation.

The partial-RAM swap-space experiment held the GPU cache at 8,000 slots and used a configured 0.625 GiB resident budget.
The two control runs reached 11.22 and 12.26 output tok/s at C8, and the experimental run reached 14.01 tok/s. The
experiment reserved 60 swap places (158 MiB), compared with 384 places (1,013 MiB) allocated above the ranked expert
payload by the controls. One experimental launch timed out before the server reported ready. These are short, exploratory
96-output-token runs. The control and experimental process footprints differed. The experiment is opt-in.

## Unified memory

The Flash-Next artifact holds 63.3 GiB of routed experts and about 50.7 GiB of Q8_0 per-layer token embeddings. Basalt can source those embeddings from SSD. At the current reservation, the experts need a placement policy that considers both WDDM budgets and Windows commit. Keep the GPU reservation in place during development and compare automatic placement at the same reservation.

A separate N1X CUDA allocation probe measured about 64 MiB of Windows commit for an empty 64 MiB device allocation before any kernel ran, with little change in the CPU working set. NVIDIA forum staff confirmed the WDDM behavior after checking with an RTX Spark engineer. On this laptop, a drop in the system-managed commit limit cut the practical model budget while CUDA still reported more than 53 GiB of free device capacity. Both counters belong in comparisons of automatic placement. [Allocation measurements and NVIDIA response](https://forums.developer.nvidia.com/t/rtx-spark-on-windows-cuda-allocations-consume-system-commit-even-with-96-gib-dedicated-gpu-memory/386453)

A local test opened a 256 MiB read-only region of the 27B model using a Win32 file mapping and registered it as CUDA mapped read-only host memory. GPU copies from three sampled offsets matched the model bytes. System committed memory was 56,640.11 MiB before and after registration. This validates the API path. A timing probe using the same bytes compares sequential, shuffled and repeated reads from dedicated device memory, ordinary CUDA host allocations, private registered memory at two alignments, and the file-backed mapping. GPU checksums matched across all placements. Large sequential and shuffled reads reached 262?268 GB/s on device allocations, 261?271 GB/s on CUDA mapped host allocations and 150?154 GB/s on the file-backed mapping. In these tests, mapped files bought capacity and avoided an anonymous copy, while CUDA mapped host allocations delivered streaming bandwidth close to device allocations. [NVIDIA CUDA UMA guidance](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

The CUDA guide describes different L2 behavior for CUDA host allocations and suitably aligned registered memory. The opt-in `--exp n1x_file_gpu_mib=1024` registered 1,021.9 MiB of file-backed expert pages and initialized the verifier's GPU staging path. It registers a separate read-only virtual view so the normal CUDA host-to-device refill can still use the original file view. Registering the original view caused a CUDA invalid-argument error during a prompt cache-slot refill. With the separate view and early registration, the engine completed a warm-up and both eight-request waves. That run used 32 output tokens and measured 5.85 output tokens/s, with p95 first-token latency of 40.22 seconds. The 96-output-token control serves as a routing comparison. A matched run with staged-expert counts is needed to measure the performance effect of staging.

### Registered read-only expert pages

The shared-memory path registers model pages through a second read-only mapping. The engine can count these weights as RAM-resident without giving the adaptive cache a writable home there. CPU readers keep the original view, and the GPU can gather straight from the registered addresses during eligible prefill. The two routes share immutable file data. The initial measurements enabled registration explicitly.

On this laptop, registration covered 8,191.7 MiB in 0.19 seconds after earlier serving runs. During registration, system commit rose by 500.1 MiB and free system RAM fell by 326.2 MiB. These are system-wide snapshots taken with a warm file cache. The engine reported 14.6 GiB of GPU experts plus 2.0 GiB of private RAM experts in both arms. Registration added 8.0 GiB of expert data to the RAM tier and reduced the file-only tier from 30.2 GiB to 22.2 GiB. The Windows page file setting and 62.5 GiB GPU reservation stayed as configured.

Flash-Next UD-IQ3_XXS ran on AC in Balanced mode, with four 4,096-token slots, int8 KV, prompt caching disabled, 7,677 GPU cache slots, about 717 input tokens and 128 output tokens. NVML did not report a GPU power limit. Both arms enabled the existing direct GPU read experiments. The control ran first, then the registered tier. Each arm ran two waves at each concurrency.

| Concurrency | Control output tok/s | Registered output tok/s | Control median / p95 TTFT | Registered median / p95 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.96 | 13.00 | 4.44 / 4.59 s | 4.09 / 4.23 s |
| 2 | 14.84 | 17.32 | 7.80 / 11.82 s | 5.33 / 9.98 s |
| 3 | 18.34 | 18.24 | 9.77 / 14.58 s | 10.48 / 14.37 s |
| 4 | 18.42 | 20.68 | 13.37 / 19.54 s | 11.88 / 17.18 s |

In the two uncached C1 requests per arm, median engine prefill was 165 versus 180 tok/s and median decode was 23.95 versus 22.45 tok/s. At C4, decode windows averaged 101.72 and 103.10 ms in the two control waves, versus 91.27 and 90.32 ms with registered pages. Time waiting for CPU experts fell from 35.50 and 32.74 ms per window to 24.39 and 22.54 ms. The registered arm read 27,223 eligible expert groups directly during prefill, versus 5,418 from private RAM in the control, including warm-up. These short runs with many experts still file-backed need an alternating repeat before changing automatic placement.

A separate greedy check held the GPU cache, private RAM allocation and 8,191.7 MiB of registered pages fixed in both arms. The only change was whether the GPU gather read eligible groups directly. It handled 1,628 groups and matched the staged arm's greedy response and all 34 accepted-row logit hashes.

The larger-budget A/B/B/A repeat kept the same 7,677 GPU cache slots and 2 GiB private expert tier. It used two C1 requests and two C4 waves per arm, with the same input and output lengths on AC in Balanced mode. The preceding registration implementation stopped at 19,732.4 MiB with a 24 GiB cap because experts can share boundary pages. The registered arms left 11.0 GiB of SSD-only weights, versus 30.2 GiB in the controls. Registration raised measured system commit by 1,320 and 1,364 MiB in the two runs.

| Arm | C1 prefill tok/s | C1 decode tok/s | C1 median TTFT | C1 output tok/s | C4 output tok/s | C4 p95 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| File staging 1 | 174.95 | 23.20 | 4.21 s | 13.16 | 17.98 | 19.04 s |
| Registered 1 | 251.90 | 26.45 | 2.94 s | 16.49 | 20.66 | 15.44 s |
| Registered 2 | 236.05 | 24.25 | 3.13 s | 15.30 | 19.30 | 16.30 s |
| File staging 2 | 149.00 | 15.05 | 4.96 s | 9.55 | 15.48 | 22.70 s |

Both registered arms beat the faster control. The two controls had matching placement and similar speculative counts. Windows commit and file cache conditions drifted, and the precise cause of the slower second control is unverified.

### Using the available memory at the existing split

The merged-registration planner selects the ranked experts, counts overlapping boundary pages once, and registers each contiguous interval. This removed the previous limit from skipping adjacent experts. It stops at the driver, physical RAM and Windows commit budgets. On Windows ARM64 the file expert source tests covered overlapping pages, joining two existing ranges, and falling back to a smaller range when a larger one would exceed the budget.

One automatic launch with no fixed GPU slot count or private RAM copy size held all 46.8 GiB of routed IQ3 experts in memory. Basalt placed 24.8 GiB on the GPU first and topped up the private RAM tier to 22.0 GiB after startup. The proposed read-only tier registered zero pages on this launch because commit headroom was tight at its earlier initialization step. The engine reported 16.1 GiB of free host RAM at ready. C1 measured 347.35 prefill tok/s, 35.10 decode tok/s, and 2.17 s median TTFT. C4 reached 26.91 aggregate output tok/s with a 10.75 s p95 TTFT.

The separate maximum-mapping check used a 2 GiB private expert budget. The inherited cache estimate priced every non-GPU expert as a private pinned copy, so the automatic GPU planner predicted zero slots. The private copy estimate now observes the configured cap and includes adaptive swap space. The same IQ3 configuration then started with 7,565 GPU cache slots (14.4 GiB), 2 GiB of private RAM experts, and 30.4 GiB of registered experts across 5,511 ranges. It had zero SSD-only routed experts and 4.99 GiB of free host RAM at ready. C1 measured 354.95 prefill tok/s, 25.00 decode tok/s, and 2.16 s median TTFT. C4 reached 19.64 aggregate output tok/s with a 13.70 s p95 TTFT. The heavier GPU placement in the preceding run delivered better decode throughput. Both launches used the same laptop memory split, AC power, Balanced mode, and the same fixture, with different serving reserves and expert placement. NVML did not report a GPU power limit. [Basalt estimate](https://github.com/jesdga95/basalt/blob/85f797f01fd1b698d8cce8d39f29f0adadd4d443/src/models/qwen4exp/load/cache_size.cpp) · [ARM port estimate](https://github.com/yanmindi/basalt-arm-sm121/blob/2ec7a844f9d83cda0fa579801f85cd8898a12bec/src/models/qwen4exp/load/cache_size.cpp)

With the same mapping policy, the larger NVFP4-DQ6_K pack started with 14.0 GiB of routed experts on the GPU, 2.0 GiB in private RAM and 33.1 GiB in registered pages. That left 14.2 GiB on SSD and 4.93 GiB of free host RAM at ready. C1 completed at 110.75 prefill tok/s and 9.10 decode tok/s. The RAM guard rejected C4 requests after the free host RAM dropped below 2 GiB. The mapped tier now applies the existing filesystem-cache reserve when the routed experts will still spill.

The retry left 8.46 GiB of host RAM at ready. It placed 13.4 GiB of experts on the GPU, 31.0 GiB in both RAM tiers and 18.9 GiB on SSD. C1 finished both requests at medians of 128.35 prefill tok/s and 14.60 decode tok/s. The file-cache change and the different GPU slot count make this a capacity check. The C4 trial was stopped to start the live IQ3 server, so steady C4 behavior for NVFP4 remains unverified.

The joint planner now reserves commit for the registered tier before finalizing GPU cache size in the opt-in automatic mapping mode. It keeps as many hot GPU experts as the remaining budget allows and holds a 1 GiB private payload for mutable cache swaps. A native planning check matched brute force across 2,500 cases. On the same live serving configuration, the earlier launch placed 24.1 GiB on the GPU, 2.7 GiB in RAM and 20.0 GiB on SSD. The new launch reported 21.0 GiB on the GPU, 25.9 GiB in RAM and zero SSD-only experts at ready. It passed a generation request and reported 11.79 GiB of free host RAM at ready. The system-managed page file and existing GPU split were used in both launches. Snapshot residency can differ from the reads a later request performs when it borrows cache slots.

With automatic sizing on a commit-constrained integrated GPU, Flash-Next selects read-only registration when the joint planner expects all experts to fit in GPU and host memory. Explicit placement and diagnostic settings take precedence. This automatic selection uses the same allocation used by the earlier opt-in serving runs.

### Four full-context serving slots

The live IQ3_XXS model pack declares a native context of 262,144 tokens. The deployment uses `--max-context 262144 --parallel 4 --slot-full-context --kv int8`. The engine backed 1.5 GiB of GPU KV at startup and reported 15.2 GiB at full allocation. Its elastic cache can grow by reclaiming 13.7 GiB from the GPU expert cache. The startup capacity estimate covered 1,048,576 positions across four sessions. Startup routed weights occupied 19.1 GiB on the GPU and 27.7 GiB in RAM, with zero SSD-only experts.

The API completed an uncached 9,312-token prompt with a 32,000-token output budget. It read the prompt at 773 tok/s, generated two tokens before its stop sequence and increased GPU KV backing. A separate short four-request wave used all four slots and returned 48 tokens per request at 17.0 aggregate tok/s including prefill. One short solo request read 719 tokens at 468.6 tok/s and decoded 128 at 44.4 tok/s. The largest test prompt was 9,312 tokens. The full-window capacity comes from the runtime estimate.

Passive telemetry subsequently recorded a completed request with 15,108 input and 3,668 output tokens at 815.0 prefill tok/s and 22.2 decode tok/s. Its time to first token was 18.71 seconds. The engine grew that slot's GPU KV backing to 20,480 positions. The surrounding batch trace averaged 113.82 ms per window, including 66.70 ms for CPU experts and 38.24 ms of GPU-reach wait. It accepted 68.7% of drafts and averaged 1.09 active slots per window. The earlier 97.1% GPU-hit and 1% SSD-read figures covered a one-token admission pass. The API now omits those counters when a request ran through batch decode, which has no request-level expert counters.

The long prefill had zero direct shared-memory expert groups. The short-prompt gather accepted registered source addresses, while the long fused path copied every nonresident expert through its staging ring. With direct prefill enabled, eligible fused layers now accept aligned GPU addresses from registered RAM too. Ineligible layers and unmapped experts keep their existing copy path. Test identical long prompts and expert placement before attributing a throughput change to this path.

Elastic KV and prefill loans can now keep recently routed experts on the GPU by trading with slots backed by either private RAM or registered read-only pages. GPU cache regrowth can also refill from both kinds of RAM. The source indexes the device address of each completely registered expert at load so placement changes can look it up directly. A native ARM64 test verified GPU reads from registered ranges in two layers with different expert sizes and checked that dropping the registrations cleared the aliases.

During verification, Windows committed 107.2 of 108.8 GiB while NVML showed approximately 29.7 GiB in use out of 62.5 GiB dedicated to the GPU. The system-managed page file had grown to 44.9 GiB. Sizing the page file explicitly for an 80 GiB initial size and a 96 GiB ceiling is a proposed comparison to leave roughly 35 GiB more commit available at the existing GPU split. Compare expert placements and throughput after restart to determine whether the extra headroom improves serving. [Microsoft commit sizing](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows#peak-system-commit-charge) · [NVIDIA CUDA memory on RTX Spark](https://docs.nvidia.com/rtx-spark/rtx-spark-porting-guide/latest/uma/umacuda.html)

### Separating prompt and decode costs

Flash-Next's PLE reader already deduplicates pages across a prompt chunk, uses bounded outstanding I/O, and caches individual rows. Its per-token read is issued ahead of the layer that needs it. NVIDIA's native Windows backend uses registered buffers with Windows I/O rings for embedding tables and reports plan, submit, wait and complete time separately. Basalt can distinguish this latency from expert reads by collecting PLE read latency and hit rate alongside its per-window GPU and CPU timings. An implementation change requires measurements with routed expert placement and the file-cache budget held fixed. [NVIDIA Windows backend](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/windows.md#serve-a-prepared-core)

### Direct GPU reads from mapped system memory

An opt-in path gives the CUDA grouped expert kernel addresses of selected mapped experts. It supports different expert sizes across layers and bypasses the copy to device scratch. The parity test used Flash-Next geometry, random NVFP4, IQ3_XXS and IQ2_S weights (the latter two with IQ4_NL down projections), and one-, three- and eight-row windows. The direct addresses and output matched the existing staged CUDA path bit for bit in every exercised case. At eight rows, the direct plan plus compute ran in 133–160 µs, about 64–104 GB/s of selected weights. Staging then computing those groups separately took 177–286 µs in the same test, before any overlap with other streams. The ARM64 Release rebuild of `dec_moe.cu` took 206 seconds under a six-job build.

The serving comparison used Flash-Next UD-IQ3_XXS, eight 4,096-token slots, int8 KV, about 717 input and 160 output tokens, and 16 requests per restart. The laptop was on AC with Windows Balanced, and NVML reported no power-limit value. Windows expanded its pagefile after the first control. These are the three runs at the resulting common placement of 15,014 cache slots (28.5 GiB GPU cache, 18.3 GiB RAM, zero file-backed experts at startup).

| Decode path | Output tok/s at C8 | p95 TTFT | Decode window time, two waves | CPU experts, two waves |
| --- | ---: | ---: | ---: | ---: |
| Direct, run 1 | 55.73 | 12.22 s | 68.17 / 68.77 ms | 4.31 / 2.68 ms |
| Direct, run 2 | 53.11 | 12.81 s | 73.23 / 70.90 ms | 6.81 / 2.07 ms |
| Control | 54.01 | 12.02 s | 69.96 / 76.60 ms | 11.40 / 5.23 ms |

The direct arms dispatched 21,739 routed entries from outside the device cache. They reduced CPU expert latency, and total throughput was within the spread of repeated runs. The path remains experimental. A control with a smaller 11,089-slot cache stopped during warm-up at a cuBLAS prefill call with status 14. The higher-cache runs and the initial 11,848-slot control completed. The cause of the smaller-cache failure is unverified.

On a single integrated GPU, the Experts card reports where the routed weights were placed at load. The same byte counts are in `/metrics`. A fresh automatic load with eight configured slots reported 21.2 GiB in the GPU cache, 25.6 GiB in RAM and zero file-backed experts. The three byte counts summed exactly to the pack's 50,292,326,400-byte routed-expert section. The load and a generation request completed under the laptop's existing reservation.

### Direct reads during prefill

An opt-in prefill experiment sends eligible pinned RAM expert addresses to the existing MMQ gather. It skips the host-to-device staging copy for those experts. The GPU cache, run-ahead copies, CPU-selected groups, file-backed experts and streamed long-prompt path use their existing routes. Flash-Next UD-IQ3_XXS was tested on AC, Windows Balanced, with eight 4,096-token slots, an 8 GiB device reserve, int8 KV, about 717 input tokens and 64 output tokens. Each arm served 16 requests. NVML did not report a GPU power limit.

| Prefill path | GPU cache slots | File-backed experts | C8 output tok/s | p95 TTFT | Decode window time, two waves |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staged, run 1 | 8,466 | 28.7 GiB | 13.33 | 31.01 s | 130.86 / 96.55 ms |
| Direct, run 1 | 8,466 | 28.7 GiB | 13.48 | 30.81 s | 116.36 / 92.80 ms |
| Direct, run 2 | 8,261 | 29.1 GiB | 13.52 | 30.76 s | 118.95 / 96.16 ms |
| Staged, run 2 | 8,189 | 29.2 GiB | 11.18 | 36.74 s | 157.13 / 112.88 ms |

Both direct arms exercised the path, using it for 3,089 and 2,771 routed groups respectively, including warm-up and rerouted prompt chunks. All arms allocated a 2.73 GiB pinned complement, including adaptive spare slots. The pair with the same GPU cache differed by 1.1% in aggregate throughput. The smaller cache and much longer CPU waits in the final control limit the comparison. These results leave staging as the default. When the full expert set was in memory in the earlier decode experiment, throughput was about 54 tok/s at C8 with 160 generated tokens, a different fixture.

A separate greedy test held the GPU cache at 7,677 slots and the pinned allocation at 2.73 GiB in both arms. The direct path handled 92 expert groups for the same 716-token prompt. All 34 reported accepted-row logit hashes matched, as did the 32-token response.

### Optional model parking

The `park-on-swap` model option keeps an idle Qwen3.8-27B engine alive while the router serves a different model. On Windows ARM64, Basalt saves the supported dense weight allocation to private system RAM, releases its device backing, and restores it at the same CUDA virtual addresses. The parked engine retains its KV cache and other allocations. Parking requires physical RAM and commit headroom. The router can unload a parked engine if the incoming model cannot load.

One on-device smoke test served Qwen3.8-27B, switched to Flash-Next UD-IQ3_XXS, then served the same greedy 27B request again. The 27B response matched before and after, and `/v1/models` reported 17,181,966,336 parked weight bytes. The copy to RAM took 3.1 seconds and weight restore took 3.2 seconds. The return request took 8.25 seconds including Flash-Next shutdown, restore and generation, compared with 5.91 seconds for the initial 27B request on a warm filesystem cache. Flash-Next served with 45.6 GiB of file-backed experts while 27B was parked. Parking remains opt-in, and the serving benchmarks use a single loaded model.

### The n-gram table's own copy

Flash-Next reads 16 rows of its 28.8 GB n-gram table from the SSD for every prompt token. The expert source maps the model
file, so Windows opens it through the file cache. That path ran about 100K row reads per second with a disk queue near 10
and most CPU cores busy: a 6.8K-token prompt's first chunk waited 0.90 s on its rows (later chunks hide the wait behind
the one before). The engine now reads an unbuffered copy of the table ([platforms](platforms.md#windows-limitations)).
The wait fell to 0.31-0.36 s. An alternating A/B against the previous build at the 96 GB split, with GPU power drifting
19-47 W, measured the following (means of four):

| Prompt tokens | Read before (ms) | Read with the copy (ms) |
| ---: | ---: | ---: |
| 2K | 1,342 | 1,101 |
| 6.8K | 4,405 | 3,308 |
| 13.6K | 9,296 | 6,470 |

At matched power (35-40 W) the 6.8K and 13.6K reads took 3,811 and 7,400 ms before and 3,413 and 6,521 ms after. The
previous build's arms also drew less GPU power, consistent with its CPU-heavy reads taking from a shared budget. Decode
(95.2 and 95.9 tok/s) and the greedy text are unchanged. A deeper queue (`--ple-inflight` 1024) or more issuing threads
did not change the cached path. The unbuffered path is at the drive's limit: a probe of random 4 KiB unbuffered reads
of the copy reached 133K reads/s from one thread (5.6 us of CPU per read) and 306-314K reads/s from four to eight
threads, and 512-byte reads were no faster.

### Where a prompt chunk goes

One 6.8K-token chunk with every stage synchronized (diagnostic build, about 2.1 GHz), 3.3 s in all:

| Stage | ms | Against the hardware |
| --- | ---: | --- |
| Routed experts | 825-857 | 51 TFLOP/s in the v2 kernel; 26 of 48 layers have IQ2_S gate/up rows, which v2 does not take |
| GDN | 716-744 | projections about 59 TFLOP/s; recurrence about 4.7 ms per layer |
| QSA | 400-409 | projections 161, attention 175 (a block per query and KV head), indexer 32, KV and norms 42 |
| Hyper-connection read and write | 640 | moves the FP32 residual of four streams, at the memory limit |
| PLE rows wait | 310-346 | first chunk only; the drive's random-read limit |
| Combine | 177 | reads ten FP32 expert rows per token |
| Shared expert, router, PLE compute | 120 | |

`mma.sync` peaks at 230 TFLOP/s for int8 m16n8k32 and FP8 m16n8k32 and at 115 TFLOP/s for int8 m16n8k16 and BF16 on
this GPU. The quantized products run at about a quarter of that: each 32-value block's scales are applied with two FP32
FMAs per output after every int8 product. The remaining prompt-speed work is in those kernels.

## CPU experts and quantization

An affinity-pinned `SMMLA` probe executed on each of the 20 cores. Basalt detects Arm I8MM and its current ARM NVFP4/Q8_0 dot product uses SDOT. A two-activation, two-output-row I8MM probe matched the existing SDOT results bit for bit across 1,024 rows at widths 512, 2,048 and 3,072. The paired dot kernels ran 2.29?2.83? faster in the synthetic microbenchmark. The engine enables the gate/up and down variants through `--exp n1x_i8mm=1`. Native wrapper parity passed 40 cases covering odd token and row counts and both SwiGLU modes. Serving gains still depend on the fraction of experts that run on the CPU. Avoid spending GPU bandwidth on activations for CPU work if measured gather cost and overlap favor a smaller transfer.

Radiance IQR-V4 is a reference for selecting per-band precision by measured quality loss, including the output head. Basalt already ships mixed precision and separate FP8 and NVFP4 head measurements. For 27B, Basalt reports mean KLD of 0.0262 and 92.82% top-1 agreement for its FP8 head, versus 0.0312 and 91.83% for the NVFP4 head. Use those as quality anchors for frequency-weighted precision bands and time prefill and concurrent decode. For Flash-Next, run per-layer sensitivity and routing calibration first. Its 63.3 GiB routed-expert tier is the larger target, and the existing pack format supports different quantizers across layers. [Radiance IQR-V4 report](https://radiance.stilldeadco.de/whitepapers/iqr-v4.pdf) · [Basalt 27B precision](https://github.com/jesdga95/basalt/blob/main/docs/precision/qwen38-27b.md)

Two published Unsloth-based Flash-Next packs offer a residency comparison. Their section sizes are from the packed model headers.

| Pack | Full file | Routed experts | PLE table |
| --- | ---: | ---: | ---: |
| NVFP4-DQ6_K | 122.64 GiB | 63.28 GiB | 50.66 GiB |
| UD-IQ4_XS | 90.94 GiB | 56.24 GiB | 26.82 GiB |
| UD-IQ3_XXS | 81.59 GiB | 46.84 GiB | 26.82 GiB |

IQ3_XXS saves 16.44 GiB in routed experts. It uses a different quantizer and GPU kernel path, so compare placement, throughput, and output quality on this laptop. [Flash-Next pack](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt/blob/fc721830e519ab52012d372e89b90d9a7df30ff2/Qwen3.8-FN-UD-IQ3_XXS.basalt) · [Basalt Flash-Next precision](https://github.com/jesdga95/basalt/blob/main/docs/precision/qwen38-flash-next.md)

An ARM64 CPU IQ3_XXS experiment used a 16 KiB signed lookup table to fold the per-vector sign multiplication into the lookup. It matched the Basalt NEON dot kernel bit for bit over 5,120 randomized rows at width 2,560. On this laptop it took 151.06 µs versus 126.59 µs for 640 rows, and 1,196.32 µs versus 1,026.07 µs for 5,120 rows. The lookup cost outweighed the saved arithmetic. Basalt keeps the existing kernel.

A worker-count sweep used a real Flash-Next NVFP4 expert with synthetic activations and the I8MM diagnostic enabled. It tested 9, 14 and 19 unpinned workers in forward and reverse order on AC in Balanced mode. At 64 rows, 19 workers took 1.02–1.10 ms median versus 1.26–1.29 ms for 14. At 32 rows, 19 workers shortened the median but increased p95 from 1.09–1.34 ms to 1.44–2.29 ms. At eight rows, 19 workers also had a higher p95 in both runs. All worker counts matched the single-worker outputs bit for bit. The sweep used a warm expert in isolation. Serving still uses the automatic 14-worker pool on this laptop.

### Flash-Next batched drafting

An initial C8 A/B/B/A run compared `--spec 8` with `--spec 0` using eight 4,096-token slots and 16 requests per restart. End-to-end throughput was 49.26 and 45.54 tok/s with drafting, and 47.53 and 45.28 tok/s without. One drafting run held 23.4 GiB of GPU experts. The other three held 28.5 GiB. Every arm kept the remaining experts in RAM. The drafting runs produced about 1.05–1.06 tokens per active slot per window, spending 8.2–9.7 ms/window on drafts. The draft setting also affects state commit overlap. These comparisons do not isolate a benefit from drafting.

The `n1x_batch_draft=0` diagnostic leaves the solo drafter and its allocations in place for an uncached serving benchmark. In the follow-up, Windows reported an 87.68 GiB commit limit at load, down from 112.41 GiB during the earlier runs. The first control had 23.7 GiB of experts on SSD and reached 26.31 tok/s at C8. The first diagnostic run and a control retry with a smaller cache and a 4 GiB device reserve both reached model ready, then returned cuBLAS status 14 from the first prompt. The cause is unverified. These runs leave the normal draft setting unchanged.

## NPU

Local cuDLA 1.5.0 reported one DLA, hardware version 2. Creating and destroying a hybrid device succeeded. Creating a standalone device returned `cudlaErrorUnsupportedOperation`. cuDLA reported no unified addressing for the DLA handle. A native CUDA allocation of 64 KiB also registered and unregistered successfully through cuDLA. The official Windows ARM64 package, TensorRT 11.5.0.63 for CUDA 13.5, compiled a static FP16 32-channel 1x1
projection with ReLU. The loadable ran through cuDLA hybrid mode. Outputs matched the NumPy reference exactly for both
test inputs. Over 128 synchronous runs on AC, median submit-to-completion was 0.314 ms and p95 was 0.476 ms. This
workload measures a single DLA graph including host submission and synchronization. [Community probe](https://github.com/roberthendriksen/rtx-spark-npu/blob/main/src/01_find_npu.py)

XDA ran FP16 convolutions, fixed-shape attention with RoPE, and concurrent GPU and DLA work. Its fixed-shape 1×1 projection ran well below cuBLAS throughput. NVIDIA documents FP16 computation, fixed dimensions and batch size one for the Windows ARM compiler. AMD’s Ryzen AI demonstrates a hybrid LLM deployment through its own ONNX Runtime GenAI stack. [XDA experiments](https://www.xda-developers.com/nvidia-rtx-spark-npu-barely-mentions-spent-week-figuring-out/) · [NVIDIA DLA operation support](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-windows-on-arm.html) · [AMD Ryzen AI LLM deployment](https://ryzenai.docs.amd.com/en/1.5/llm/overview.html)

Four alternating eight-way Qwen3.8-27B serving waves compared the idle DLA context with the same context submitting the 32-channel projection continuously in groups of eight. Throughput averaged 63.62 output tokens/s with the DLA idle and 48.04 with it busy (−24.5%). Per-wave p95 time to first token was 3.84 and 5.55 seconds with the DLA idle, and 7.12 and 7.44 seconds under DLA load. A standalone DLA run completed 2,044 graphs/s, versus 242 and 262 graphs/s while Qwen was running. The experiment used separate CUDA contexts, AC power, normal desktop activity, and a parallel model download. It establishes that these two workloads interfere on this system.

### Routed expert feasibility

TensorRT 11.5.0.63 compiled a full Flash-Next expert (`2560 → 640 → 2560`) at batch dimension one, with same-expert tokens packed along the spatial width. The graph uses three 1×1 convolutions and expresses SwiGLU as sigmoid and products. The probe used expert 0 from layer 0 of the NVFP4-DQ6_K pack, with synthetic input activations. Basalt used the packed NVFP4 weights with Q8_0 activations. The DLA used the same weights expanded to FP16. Against an FP32 reference, relative RMS error was 0.53–0.57% on DLA and 0.90–1.04% on CPU. The timing includes activation quantization for Basalt and the two CUDA transfers in the DLA host round trip. [NVIDIA Windows on ARM DLA operations](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/inference-library/dla-windows-on-arm.html)

| Tokens for one expert | Basalt one worker, median ms | Basalt 14 workers, median ms | DLA submit and wait, median ms | DLA host round trip, median ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.555 | 0.100 | 0.454 | 0.635 |
| 2 | 0.404 | 0.091 | 0.456 | 0.677 |
| 4 | 0.802 | 0.181 | 0.455 | 0.693 |
| 8 | 1.613 | 0.203 | 0.481 | 0.686 |
| 32 | 6.591 | 0.894 | 0.527 | 0.611 |
| 64 | 13.687 | 1.378 | 0.530 | 0.678 |

Each result is the median of 64 warm iterations. The 14-worker pool matched one-worker output bit for bit and used groups of up to eight rows per job. The probe used a warm expert and one graph per token width. Each loadable was about 12 MiB on disk, compared with 2.64 MiB for the packed weights. Measure resident memory separately before choosing a graph-cache size. Build `tools/port/n1x_expert_cpu_probe.cpp` against `kernels_cpu`, `platform` and `ggml_quants`, then pass the resulting binary and the CUDA, cuDLA and TensorRT directories to `tools/port/n1x_expert_dla_probe.py`. [Community cuDLA helpers](https://github.com/roberthendriksen/rtx-spark-npu)

In a separate run, a native cuBLAS reference held the same expert in FP16 in device memory and ran three projections with a CUDA SwiGLU kernel. The GPU and DLA could submit independently on two streams in the same CUDA primary context. Both used preallocated CUDA buffers. The CPU used the same 14-worker configuration as above.

| Tokens for one expert | CPU pool, median ms | DLA submit and wait, median ms | Resident cuBLAS submit and wait, median ms | One GPU and one DLA job together, median ms |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.219 | 0.493 | 0.081 | 0.550 |
| 32 | 1.089 | 0.492 | 0.066 | 0.583 |
| 64 | 1.428 | 0.609 | 0.080 | 0.700 |

The GPU path completed the single warm expert 6–8× faster than the DLA. Adding a DLA job beside a GPU job measured close to the sum of their individual times. Each GPU result was within 0.048% relative RMS of the FP32 reference. The TensorRT GPU builder failed to load `myelin_resource64_sm110_2.dll` from the local installation, so the GPU measurements use cuBLAS. To reproduce, compile `tools/port/n1x_expert_gpu_probe.cu` as an ARM64 CUDA DLL linked against cuBLAS and run the Python probe with `--gpu-library` and `--model`.

The same-context burst probe queued several resident GPU expert calls around one DLA job. Both used the same weights and synthetic inputs. Each case used 64 warm iterations, and both outputs passed validation. The CUDA-only control with one extra call does the same amount of expert work as the hybrid cases. These are median wall times in milliseconds from a run on AC with Windows Balanced. NVML did not expose a configurable GPU power limit for the laptop.

| Rows per expert | GPU jobs before adding one expert | GPU only | One extra GPU job | GPU then DLA, sequential | DLA submitted first | GPU submitted first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 0.540 | 0.582 | 1.004 | 0.637 | 0.953 |
| 32 | 16 | 1.093 | 1.108 | 1.603 | 1.099 | 1.466 |
| 32 | 32 | 1.951 | 2.193 | 2.809 | 2.108 | 2.527 |
| 64 | 8 | 0.658 | 0.719 | 1.278 | 0.826 | 1.041 |
| 64 | 16 | 1.254 | 1.392 | 1.935 | 1.399 | 1.922 |
| 64 | 32 | 2.698 | 2.781 | 3.240 | 2.779 | 2.740 |

Early DLA submission overlapped much of its execution with the GPU burst. Compared with the extra GPU call, the DLA-first median ranged from 0.085 ms faster to 0.107 ms slower. Its p95 improved in two of the six cases. This repeated-expert probe excludes model routing and staging, and the 12 MiB DLA loadable still competes for memory. Use `--gpu-burst 8 16 32 --tokens 32 64` to reproduce.

In 16 eight-way Flash-Next requests with an 8,000-slot cache and file-backed misses, 51,585 of 56,628 CPU expert groups carried a single token. Only 137 groups carried four or more, and the largest group carried six. `--exp n1x_cpu_group_stats=1` counts the groups the CPU actually receives in batched decode. A separate run with `--exp n1x_prefill_group_stats=1` observed 188,114 nonresident expert groups across 19 prefill invocations. Of those, 37,115 carried at least 32 rows, including 16,174 with at least 64. The prefill counter covers layers that group routed IDs on the host. The two traces favor testing NPU offload during prefill.

The DLA candidate is a bounded set of frequently used Flash-Next experts handling larger groups during prompt processing. The RTX GPU can keep attention, shared layers and its resident experts. The CPU pool can serve small routed groups and everything without a compiled DLA graph. cuDLA hybrid mode accepts registered CUDA buffers, allowing a same-context implementation to keep activation input and output on the CUDA side. Budget the expanded weights inside automatic memory placement and dispatch DLA only where the combined pipeline wins against the existing GPU and CPU pools. Measure p95 first-token latency, aggregate output tokens/s and GPU overlap. AMD's production hybrid deployment similarly targets compute-intensive prefill for its NPU and token generation for its GPU. [NVIDIA cuDLA hybrid mode](https://docs.nvidia.com/cuda/cudla-api/cudla_api/group__CUDLA__API.html) · [AMD hybrid implementation](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-llm-startup-on-amd-ryzen-ai.html)
