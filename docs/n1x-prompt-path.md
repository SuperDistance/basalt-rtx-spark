# The N1X prompt path

What a prompt costs on the RTX Spark N1X at the 96 GB split besides its layers' arithmetic, and what changed it. Speed figures elsewhere: [performance notes](n1x-performance.md), [GPU power](n1x-gpu-power.md).

## The n-gram table's own copy

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

## Request overhead

A request's first token also waits for the server to hand the engine its prompt, the prefix match and the run, one
command each over the engine's stdin. On Windows the engine polled that pipe at the 15.6 ms system tick, so a short
request spent 66-71 ms beyond its prompt read and generation (median, about 70-token prompts). The pipe is now polled
every millisecond within 2 s of the last command, at the tick when idle: 21-27 ms, and short requests' wall time fell
from about 400 to 360 ms. The idle engine's CPU did not rise (0.8% of a core against 2.0%).

## Where a prompt chunk goes

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

## The dense projections' copies

The prompt path multiplies every native dense projection (GDN qkv, gate and out; QSA q, k, v and out; the shared
expert) in FP16 and dequantized each one's GGUF blocks into a scratch for every chunk: about 0.45 ms of a GDN qkv
projection at a 110-token chunk, against 0.06 ms for its blocks' bytes. When the VRAM left after loading holds
FP16 copies of them beside every session's KV at full context, the reserve and 2 GiB, the server now makes them once
(300 matrices, 5.42 GiB) and the products read them: the same cuBLAS calls on the same rows, so the same bits (a
fixed 2K-token prompt's greedy reply hashed the same on both builds). A VRAM give-back drops them first. Alternating
A/B, prompt read in ms:

| Prompt tokens | Before | With the copies |
| ---: | ---: | ---: |
| ~110 | 194-215 | 158-176 |
| ~420 | 318-338 | 283-296 |
| ~2K | 1,034-1,063 | 985-999 |

BF16 copies of the hyper-connection matrices (held as 8-bit copies) did not change the read time at any of these
sizes and were left out.
