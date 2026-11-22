# GPU-first Flash-Next on the N1X

Every routed expert of the Flash-Next UD-IQ3_XXS pack now starts on the GPU at the laptop's existing 64,000 MiB split, with
four 262,144-token slots and int8 KV, and decode no longer hands work to the CPU. This page records what changed and what it
measured. The earlier placement experiments are in [N1X performance notes](n1x-performance.md).

## Why the ARM port measured faster

The public ARM port is upstream Basalt `63cb68a` plus one squashed porting commit, and its code has not moved since this fork
started from it. Upstream's later commits change other models and serving, not Flash-Next decode. The gap came from setup:

| | ARM port's published run | This fork before the changes below |
| --- | --- | --- |
| GPU split | 96 GB, set in firmware | 64,000 MiB (unchanged) |
| Page file | custom 80-96 GiB, set by hand | system managed (unchanged) |
| Pack and routed experts | NVFP4-DQ6_K, 63.3 GiB, all on the GPU | IQ3_XXS, 46.8 GiB, 19.1 GiB on the GPU |
| Prompt cache | default (6 checkpoints) | off |
| C4 figure | streamed tokens after the first minute (decode only) | output over wall time, prefill included |

With the cache off, every turn of a conversation is read from its first token, and a request left alone in a batch slot
never returns to the solo path. The C4 figures measure different things: per-request C4 decode here was 20.5 tok/s, about
82 aggregate, against the port's 80.1. NVFP4 also reads prompts on FP4 tensor cores in 32,768-token chunks, where IQ3 uses
the int8 fused kernels in 16,384-token chunks; quantization quality was not compared.

Measured with the port's own `tools/serve_speed.py long` against the GPU-first server: a 4,024-token prompt read at 1,247
tok/s and decoded at 74.0 tok/s, a 15,921-token prompt read at 2,015 tok/s and decoded at 71.4 (the port reports 1,473 and
63.6 tok/s at 4,128 tokens). The packs and prompts differ, so this is a reference, not parity.

## Commit counted against the page file's room to grow

Windows charges integrated-GPU allocations to the commit limit. The engine read only the current limit, and since every
planner stayed under it, a system-managed page file never grew: the same pinned configuration registered 12.0 GiB of shared
expert pages in two launches and none in the next two (13.6 GiB on the SSD) as the current limit moved between 75.3 and
85.0 GiB. The laptop's page file was 11.3 GiB and may grow to 119 GiB. `platform::commit_headroom` now adds each paging
file's room to grow, held to its volume's free space less a tenth.

In one A/B pair with automatic placement (not A/B/B/A), the growth-aware build placed every routed expert on the GPU:

| | Current limit only | With the room to grow |
| --- | ---: | ---: |
| C1 decode, 717-token prompt | 52.7 tok/s | 62.2 tok/s |
| C1 output, prefill included | 33.9 tok/s | 40.5 tok/s |
| C4 aggregate output | 41.5 tok/s | 48.6 tok/s |
| 8K / 16K cold prefill | 1,338 / 1,523 tok/s | 1,642 / 1,732 tok/s |

Both ran on AC in Balanced mode; NVML exposes no GPU power limit. Windows grew the page file during load, and system commit
reached about 109 of 110 GiB while serving. Physical RAM, not commit, then bounds the host tiers.

## Decode without the CPU

With every expert resident, the only misses are experts that elastic KV displaces at long contexts (98.5-99.2% hits at
30-50K). Three changes moved that work to the GPU:

- **Pool spin.** A parked worker spun 20 ms before sleeping, and a miss arrived every few milliseconds, so all 14 workers
  held a full core (an `isb` loop on Arm). On an integrated GPU the pool now sleeps after 200 us. Live traffic went from 15
  busy threads to about 4 cores in total; the GPU kept its decode clock (about 1.5 GHz, P4).
- **Direct share.** On an integrated GPU, CUDA0 computes up to 16 of a layer's misses in place from their mapped RAM
  addresses in every window; the CPU pool takes only the rest and no longer warms experts.
- **Device-planned tail beside the zero-doorbell graph.** An engine that started fully resident ran the host-planned graph
  once experts left, so the CPU computed them. The device-planned tail is now built for it and used only while experts are
  out; such an engine keeps the lazy GDN commit and the landed words off, as before.

A correctness run after these changes: the GPU read 95-248 displaced experts per request (0.1-0.2% of routed) and the CPU
none; greedy output repeated for a 30,105-token prompt read cold and from its checkpoint; decode at 30K context read
74.9-78.2 tok/s and cold prefill 1,963-1,977 tok/s. The kernels match the staged CUDA path bit for bit
(`dec_moe_parity --kq`, `--nvfp4`). Moving experts from the CPU to the GPU changes their rounding; the KLD gate is pending.

## NVFP4 against IQ3_XXS

The same build (`67328e0`) and serving settings (four 262,144-token slots, int8 KV, MTP 8, prompt cache on) ran both packs
A/B/B/A. IQ3_XXS held every routed expert on the GPU. NVFP4-DQ6_K placed 43.2 GiB on the GPU and 20.1 GiB in RAM, which the
GPU read in place, and left 3.8-6.0 GiB of physical RAM free against 14.1-15.8 GiB for IQ3. SparkHelper's Performance
profile was active. The fourth arm (IQ3) ran with the GPU clock capped near 1,470 MHz, against 1,960-2,130 MHz medians in
the other arms, while another tool tested GPU caps; it is left out below.

| | IQ3_XXS | NVFP4-DQ6_K (two arms) |
| --- | ---: | ---: |
| C1 decode, 717-token prompt, median | 80.4 tok/s | 70.9 / 64.3 tok/s |
| C1 output, prefill included | 52.7 tok/s | 45.6 / 41.3 tok/s |
| C4 aggregate output | 63.2 tok/s | 54.9 / 54.8 tok/s |
| Prefill 717 / 7,715 / 30,105 tokens | 725 / 2,143 / 2,227 tok/s | 603-542 / 2,054-2,228 / 2,154-2,423 tok/s |
| Decode after a 30K prompt | 81.2 tok/s | 75.1 / 80.0 tok/s |

GPU power medians were 34-46 W at full clocks and the SoC reached 88-93 °C. At this split IQ3_XXS decodes faster and keeps
far more RAM free; NVFP4 matches it on long prompts.

## Cold prompts joining a decoding stream

One stream decodes (717-token prompt, 1,500 tokens out) and a cold prompt joins it two seconds in. IQ3_XXS, the same
serving settings, `67328e0`:

| Joining prompt | Joiner's first token | The stream while the prompt is read | The stream with both decoding |
| ---: | ---: | ---: | ---: |
| 1,975 tokens | 2.30 s (974 tok/s) | 9.2 tok/s | 43.9 tok/s |
| 7,704 tokens | 4.21 s (1,962 tok/s) | 2.4 tok/s, stalled about 4 s | 52.7 tok/s |
| 15,334 tokens | 7.69 s (2,059 tok/s) | 1.2 tok/s, stalled about 7.5 s | 46.1 tok/s |
| 30,103 tokens | 28.66 s (1,064 tok/s) | 30.8 tok/s, after a 3 s stall | 36.1 tok/s |

Alone, the stream decodes at 69-80 tok/s. A prompt of up to 16,384 tokens is read while the decoding slot waits, and its
first part of a longer one too (the 30K prompt's first 6,144 tokens). The rest of a longer one is read on its own thread
beside the windows, which share the GPU about evenly with it: windows took 76 ms against 36 ms alone, and the read ran at
half its speed alone. Lowering the read-alone limit to 2,048 tokens did not help: the joiner's prompt was still read with
the decoding request taken out of its slot (its conversation copied out and back), and a 15K joiner took 29.7 s. The
admission has to keep the slot decoding before a smaller limit can work.

C1-C4 aggregate output on the same build (717-token prompts, 160 tokens out): 49.1, 54.3, 61.5 and 62.1 tok/s; per-request
decode 76.9, 49.2, 30.1 and 22.8 tok/s; p95 first token 1.24, 2.69, 3.87 and 5.03 s.

## EXL3 quants

EXL3 (ExLlamaV3, MIT) stores each linear as trellis codes in 16×16 tiles at `16*K` 16-bit words a tile, with fp16 sign
vectors around 128-point Hadamard rotations, and decodes in registers. turboderp's self-calibrated Flash-Next quants list
mean KLD 0.0138 at 3.18 bpw (about 45 GiB of routed experts by bit count) against 0.0529 for a 2.90 bpw IQ3_XXS, on his own
measurement. Hero422's EXL3-V3 quants of Qwen3.8-27B use an integer "mad" codebook with int8-activation compute and a bit
width per tensor, and list 0.0215 KLD at 3.0 bpw (11.8 GiB) and 0.0064 at 4.0 bpw (14.9 GiB) on their own corpus; they are
Radiance containers whose kernels target RDNA4, and no Flash-Next EXL3-V3 exists from that uploader yet. Basalt reads
neither format today.

## Short prompts take the fused path

Below `stream_all_min()` a prompt chunk stages only its routed experts and runs MMQ per expert group, which costs about
45 to 80 us per (expert, layer) group on the N1X whatever the token count: 0.47 s for a 109-row chunk (10,358 groups),
1.2 s for 713 rows. The fused path streams every expert and costs about 0.29 s plus 0.44 ms per token. With no peer card
there is no link to move experts over, so an integrated GPU now streams from 64 tokens (`kStreamAllMinIntegrated`; other
cards keep 6,144). Uncached first-token times on the candidate build, two runs each: 89 tokens 0.35-0.41 s, 116 tokens
0.37-0.47 s (0.60 before), 170 tokens 0.40-0.55 s (0.66), 716 tokens 0.67-0.73 s (1.04), 5,846 tokens 3.07 s (4.22 at
cutoff 512 against the old 6,144). Under 64 new tokens the routed-only path stays cheaper (agent turns of a few dozen
tokens read in about 0.1 s).

The 100-chunk gate (wikitext-2 against the BF16 base, T 3, int8 KV, one GPU, UD-IQ3_XXS) reads the 256-token prompt
halves through either path:

| Prompt path | PPL (base 3.2733) | KLD mean | KLD p99 | same top-1 | prompt ms/chunk |
| --- | --- | --- | --- | --- | --- |
| routed-only below 6,144 (2ab91eb) | 3.2444 | 0.11814 | 1.3001 | 88.65% | 723 |
| fused from 64 | 3.2407 | 0.11634 | 1.2527 | 88.78% | 404 |

Only 3 of 25,500 rows are identical (the two paths round differently from the first scored position); the means do not
get worse.

This is also the first gate of the UD-IQ3_XXS pack on its own: KLD 0.118 and top-1 88.7%, against 0.0628 and 91.55% for
the NVFP4 pack on the same gate ([Flash-Next precision](precision/qwen38-flash-next.md)). The pack is Unsloth's mix:
gate and up are IQ2_S on 23 layers, IQ3_XXS on 21 and IQ4_XS on the last, and down is IQ4_NL almost everywhere. Its
speed comes with about twice NVFP4's divergence from the BF16 model.

## NVFP4 prompts on BF16 activations (W4A16)

Below `FP4_MIN_TOKENS` (1,024) an NVFP4 layer ran MMQ, which on the N1X reads a 716-token prompt at about 250-520
tokens/s. Lowering the FP4 tensor-core path's floor to 256 made that prompt 4.6x faster, but that path rounds the
activations to FP4 (W4A4) and the 100-chunk gate on Qwen3.8-FN-FP4-Local (wikitext-2 against the BF16 base, T 3, int8
KV) got worse in 73 of 100 chunks (paired t 4.7). It stays at 1,024.

The W4A16 kernels (`moe_fp4.cu`, `a16` / `a16x`) keep the FP4 path's blocks, read the NVFP4 weights in place and
widen them exactly to BF16 in registers (`cvt.rn.bf16x2.e2m1x2`, times the sub-block scale; an E2M1 value times a
UE4M3 scale has at most six significant bits), against activations rounded only to BF16, on the BF16 tensor cores
with FP32 accumulation; H stays BF16 for the down product. A block whose expert has at most 64 rows spreads its 128
weight rows over all eight warps, and 16-row slices without rows are skipped. An integrated GPU with no peer now runs
NVFP4 chunks below the floor this way instead of MMQ.

| Prompt path on FP4-Local | PPL (base 3.2733) | KLD mean | KLD p99 | same top-1 | prompt ms/chunk |
| --- | --- | --- | --- | --- | --- |
| MMQ (int8 activations) | 3.3042 | 0.08877 | 1.0304 | 90.20% | 1192 |
| FP4 from 256 (W4A4) | 3.3130 | 0.09291 | 1.0767 | 89.98% | 353 |
| W4A16 | 3.3048 | 0.08847 | 1.0582 | 90.32% | 776 |

Against MMQ, chunk by chunk, W4A16 is the same (mean dKLD -0.0003, se 0.0007, worse in 48 of 100). In
`prefill_fp4_test` the expert layer lands at RMS 2.9e-3 from double math on the unrounded activations, against MMQ's
1.6e-2 and the FP4 path's 0.16. The kernels' later speed-ups (128-value stages, the code conversion instruction, the
eight-warp split) left the output bitwise unchanged, in the test and in the engine (the same greedy text as the gated
build).

The expert layer in the test (512 experts, top 10, one run per build, the FP4 path in the same run as the yardstick):

| Tokens | MMQ | W4A16 | FP4 (W4A4) |
| --- | --- | --- | --- |
| 716 | 20.3 ms | 8.2 ms | 7.4 ms |
| 2,048 | 22.7 ms | 9.2 ms | 8.7 ms |
| 16,384 | 72.0 ms | 36.9 ms | 24.8 ms |

Served, a cold 716-token prompt reaches its first token in 0.66 s (MMQ 1.37 s, FP4 from 256 0.61 s). Chunks of 1,024
tokens and more keep the FP4 path, which has not been gated on prompts that long.

## Stuck GPU clocks

After a three-second Modern Standby entry from the power button, and a firmware thermal-profile test that ended within a
minute of it, the GPU stayed in P8 at about 810 MHz and 6 W while 96% busy, across engine restarts. Decode fell to 22-32
tok/s and a device-memory read probe to 53-58 GB/s (262-268 GB/s normally). A reboot cleared it.

It happened again after an AC unplug and replug: about 800 MHz and 7 W at 64 °C under load, 19 tok/s, and this time it
survived a warm reboot. Exiting SparkHelper from the tray cleared it at once (2,436 MHz, 50 W, 72 tok/s). The exit
uncaps the GPU, returns the HP firmware fan mode to Adaptive and restores Windows power settings; which of the three held
the clock is unverified. SparkHelper's tuning helper re-applies the profile, firmware fan mode included, at every logon
and power-source change, which would explain surviving the reboot. The NVML "Reliability" clock-event reason (bit
0x400) is set at full clocks too, so it says nothing here. Watch the GPU clock when decode drops for no reason.

## GPU power that does not hold under serving

Live agent traffic (one stream, 60K-110K-token cached contexts, 2026-11-22 12:10-12:45, on AC) did not keep the GPU at
one power level, and the steps line up with neither a fixed cap nor one temperature:

| When | P-state | Clock | GPU power | GPU / SoC temperature | Driver reason | Decode |
| --- | --- | --- | --- | --- | --- | --- |
| short test request, cool | P0 | 2,380-2,440 MHz | 48-50 W | 64 / 66 °C | none | 72-73 tok/s |
| 12:33, 96% busy | P4 | 1,350-1,650 MHz | 20-23 W | 61 / 66 °C | none | 64-68 tok/s |
| 12:35-12:38 | P0 | 2,000-2,250 MHz | 40-48 W | 80-86 / ~88 °C | none at the samples | 80-137 tok/s |
| 12:39-12:41 (per-minute means) | P0, then mostly P4 | 2,080 → 1,880 MHz | 47.6 → 36.0 → 32.8 W | 75-77 / 82-83 °C | none at the samples | |

The driver's cumulative counters moved between 12:33 and 12:38 by 39 s of "SW Power Capping" and 39 s of "SW Thermal
Slowdown", and not at all from 12:38 to 12:42, so both a software power cap and a thermal slowdown act at times, while
the 12:33 spell (cool, P4, half the power at 96% busy) and the 12:39-12:41 slide (power falling at a flat 83 °C SoC)
carry no reason at all. The flat temperature while power falls suggests a platform controller holding the SoC near a
temperature target by lowering the GPU's budget; the user also saw the fan start minutes after the chassis got hot. An
unthrottled run with a cold fan heats the SoC from 49 to 97 °C in about 90 s (the KLD gate's log), after which the
firmware's SwThermal slowdown sets in and the GPU settles at 40-45 W. Which controller picks P4 at 61 °C is
unverified. For benchmarks this means: a short run measures the boost, a long one the platform's sustained budget, and
the two differ by up to half; log P-state, clock, power and both temperatures with every number (`spark monitor`,
`nvidia-smi -lms 1000`).

## Still open

- The resident RAM copy holds 22 GiB of experts that are also on the GPU, for the slots elastic KV and prompt loans may
  take. Free RAM sat near 10 GB while serving. Its size against the KV ceiling needs a measured choice.
- Matched C1-C4 runs of the current build against the port's method and a GB10 host.
- The KLD gate for experts computed on the GPU instead of the CPU.
- NVFP4 and IQ3 on equivalent placements, and Qwen3.8-27B.
