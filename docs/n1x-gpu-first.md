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

## Drawn MTP drafts (speculative sampling)

A sampled Flash-Next request used to keep an MTP draft only where the target's own draw was the draft. Now a single
request's draft chain draws each draft from the draft head (the 2-bit proxy's rows rescored exactly, cut by the
request's top_k and top_p, softmax at its temperature, Philox's draft stream at the draft's cell) and hands the proposal
to the sampler, which keeps a draft with min(1, p/q) and otherwise draws from max(p - q, 0): the reply has the model's
distribution exactly. A step without the proxy writes a one-hot proposal at its argmax pick, also exact. Greedy and
seeded requests, lookup windows and batch windows keep the old rule. `q6k_head_parity` checks the proposal against
softmax(exact logit / T) (2e-7), its sum, membership and the draws' frequencies (within 2.1 sigma over 4,000 draws).

Same build, UD-IQ3_XXS, 12 sampled requests per temperature (six prompts twice, 300 tokens, one at a time):

| | tokens per window | drafts kept |
| --- | ---: | ---: |
| T 1.0, drawn | 2.570 | 68.3% |
| T 1.0, argmax drafts | 2.468 | 66.6% |
| T 0.7, drawn | 2.547 | 68.5% |
| T 0.7, argmax drafts | 2.560 | 67.2% |

Greedy text was identical in both arms. The gain at temperature 1.0 (+4%) is far below the +18% Radiance reports for
its coupled sampling on Flash-Next; these prompts give peaked distributions, where the argmax draft already matches most
draws.

## Stuck GPU clocks

After a three-second Modern Standby entry from the power button, and a firmware thermal-profile test that ended within a
minute of it, the GPU stayed in P8 at about 810 MHz and 6 W while 96% busy, across engine restarts. Decode fell to 22-32
tok/s and a device-memory read probe to 53-58 GB/s (262-268 GB/s normally). A reboot cleared it.

It happened again after an AC unplug and replug: about 800 MHz and 7 W at 64 °C under load, 19 tok/s, surviving a warm
reboot, and cleared at once by exiting SparkHelper from the tray (2,436 MHz, 50 W, 72 tok/s). The cause is the HP
firmware fan mode (BIOS WMI thermal profile, command 0x4C): with "Cool" or "Quiet" the GPU runs in P8 at 730-890 MHz and
6-11 W under a steady CUDA load, against 1.6-2.3 GHz and 45-71 W with Adaptive, Performance or Balanced. A change takes
effect within 2 s either way, and the embedded controller keeps the mode across a warm reboot. SparkHelper's
Performance profile had been set to Cool, so each of its re-applies (boot, profile applies) clamped the GPU and each
exit or restore (Adaptive) released it. SparkHelper 0.2.0-agent.2 labels Cool and Quiet as GPU power limits, asks before
using them and warns when the GPU sits in P8 under load. One onset (at an AC replug) came before the Cool setting and is
unexplained. Watch the GPU clock when decode drops for no reason, and check the fan mode first.

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

A later run of six back-to-back four-stream arms (about four minutes each) slid the same way: GPU power over the busy
samples 58.7, 51.7, 42.4, 31.8, 27.0 and 27.6 W, clocks 2.31 GHz down to 1.62 GHz, with SparkHelper's monitor reporting
"Reliability" as the reason in almost every busy sample from the third arm on. Back-to-back arms are biased against the
later ones; compare kernels in isolation or alternate arms with power logged.

## Where a four-stream window goes

UD-IQ3_XXS, four greedy streams, about 7.7 verify rows per window. Profiled windows (per-stage GPU stamps) take 53-62 ms
depending on the power state; at 53 ms:

| Part | ms per window |
| --- | ---: |
| experts (VRAM hits) | ~21 |
| hyper-connection reads + router | ~7.5 |
| q8 + qkv / index projections | ~6.6 |
| out projections | ~2.7 |
| GDN recurrence | 2.5-3.6 |
| shared expert + quant | ~2.1 |
| head | ~1.0 |
| launch and readback outside the stamps | 2.0-2.5 |
| draft section (MTP draft beside the GDN commit) | 7.6-8 |

The per-slot experts barely overlap (about 10 VRAM hits per row per layer), so each verify row costs roughly the same
expert traffic. The draft section is slow because the drafter runs beside the GDN state commit and both are bound by
memory: with the overlap turned off the catch-up drops from 4.2 to 0.27 ms and the draft to 3.6 ms, but the commit
then takes 4.6 ms on its own and the window time does not change. The commit moves the GDN state of every slot once in
and once out: 36 layers x 48 value heads x 128 x 128 FP32 = 113 MB per slot, about 906 MB at four slots. In
`dec_commit_parity --bench` it runs at 194-214 GB/s for every grid width from 32 to 340 blocks per segment, the same as
a chain of 4 MB device copies (about 207 GB/s), so the kernel is at the card's memory limit and block tuning cannot
help. So the commit now reads the state once: a continuing slot's batch commit writes only its conv history and
indexer, and its next window's recurrence replays the kept rows into the state first (the commit's sums in the
commit's order, so the state is the same bits; `dec_gdn_parity` checks window outputs and states against the full
commit) and writes it back, before its grid dependency wait. A slot that sits out a window, or any other launch on the
stage, gets the plain state pass first. Same build otherwise, ABAB at four streams:

| Arm | GPU W | ms per window | verify run | draft section | output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 55.1 | 52.26 | 44.32 | 7.81 | 109.3 |
| fused | 51.2 | 49.94 | 46.14 | 3.67 | 116.5 |
| before | 40.9 | 54.05 | 46.02 | 7.90 | 109.3 |
| fused | 30.3 | 54.67 | 50.56 | 3.98 | 104.7 |

The draft section halves in both pairs and the verify run takes about 1.8 ms more (the state's write). The second
pair ran 10 W apart and does not compare. One-stream greedy text was identical in every arm.

## Images

`--vision dynamic` on UD-IQ3_XXS, fresh images (two runs each, GPU about 52 W):

| Image tokens | Encode | Prompt | First tokens |
| ---: | ---: | ---: | ---: |
| 1,085 | 158 ms (301 ms cold) | 636-711 ms | 1.2-1.5 s |
| 3,197 | 850-874 ms | 1.56 s | 2.95 s |
| 4,157 | 1.21 s | 1.97-2.0 s | 3.8 s |

Lending the expert-cache slots, staging the encoder weights and refilling the slots take about 40 ms together. The encoder
is about 30% of a fresh image's first-token time; the image tokens' prompt is the rest. The NPU is no help here: the
encode and the prompt run one after the other, the DLA (FP16, batch 1, limited transformer operations) would have to beat
the GPU's 0.3 ms per image token to shorten a single image, overlap would only help several images in one request, and
the DLA shares the memory bandwidth (a continuous DLA load cut serving throughput from 63.6 to 48.0 output tokens/s, -24.5%, in the earlier test). It would also break the
bitwise-equal embeddings every vision mode gives today.

## Against vLLM for RTX Spark

BetterBench 0.6.1 with one config for both (temperature 0.7, cold prefix cache, concurrency 1-4) on the 96 GB GPU split:
Basalt 2dacc65 on UD-IQ3_XXS against vLLM for RTX Spark 0.2.0 (vLLM 0.32.0) on FP4-Local with its default recipe (MTP 7
with block verification, Marlin weight-only NVFP4 MoE, b12x MXFP8 dense; its kernels are sm_121a code, and its FP4 GEMMs
need sm_100, so the experts run weight-only).

| | Basalt | vLLM |
| --- | ---: | ---: |
| Single-stream decode, weighted median | 91.9 tok/s | 55.1 tok/s |
| First token, short prompts (p50) | 375 ms | 279 ms |
| Aggregate output at 1 / 2 / 3 / 4 streams | 77.6 / 80.9 / 91.4 / 93.9 | 53.0 / 81.2 / 97.8 / 104.4 |
| Prefill at 2K / 8K / 16K / 32K / 64K tokens | 1,218 / 1,590 / 1,768 / 1,840 / 1,845 | 1,370 / 1,773 / 1,815 / 2,035 / 2,182 |

The same Basalt build on the 64 GB split had measured 74.0 tok/s decode and 697 tok/s at 2K: there the prompt path read
most of its experts from RAM. Both runs burst for about eight minutes and then ran under the firmware's sustained clamp
(P4, 22-30 W, reason 0x400), so their prefill and concurrency phases are in the sustained regime. vLLM gains on
concurrency because it draws its MTP drafts at every concurrency; Basalt's batch windows took argmax drafts only.

## Still open

- The resident RAM copy holds 22 GiB of experts that are also on the GPU, for the slots elastic KV and prompt loans may
  take. Free RAM sat near 10 GB while serving. Its size against the KV ceiling needs a measured choice.
- Matched C1-C4 runs of the current build against the port's method and a GB10 host.
- The KLD gate for experts computed on the GPU instead of the CPU.
- NVFP4 and IQ3 on equivalent placements, and Qwen3.8-27B.
- The same deferred commit for one-stream windows (their lazy commit still reads the state a second time).
- The hyper-connection reads: about 14% of a four-stream window, the same without the profiler (105 us a layer cold at
  8 rows, 53 us with the weights in L2); the chain inside the read, not its bytes, is the cost.
- A BF16 GDN state would halve the state traffic (about 6% at four streams) but is not lossless. With the stores rounded
  to BF16 (an emulation, same build otherwise), the KLD gate's first 35 chunks against the FP32 state: ln(PPL/base)
  0.00106 -> 0.00225, KLD 0.12618 -> 0.12679, same top-1 90.59% -> 90.33%, and only 1.2% of rows bit-identical. The
  error compounds with context length, which 512-token chunks do not show; at most an opt-in.
- The hyper-connection read with its matrices prefetched into L2 before its dependency wait was slower cold (8 rows with
  the router 108 -> 119 us): a cold read moves about 90 GB/s, bound by its down, up, router and top-10 chain, not bytes.
