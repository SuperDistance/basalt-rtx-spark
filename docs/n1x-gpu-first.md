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

## Stuck GPU clocks

After a three-second Modern Standby entry from the power button, and a firmware thermal-profile test that ended within a
minute of it, the GPU stayed in P8 at about 810 MHz and 6 W while 96% busy, across engine restarts. Decode fell to 22-32
tok/s and a device-memory read probe to 53-58 GB/s (262-268 GB/s normally). A reboot cleared it. Which event caused it is
unverified. Watch the GPU clock when decode drops for no reason.

## Still open

- The resident RAM copy holds 22 GiB of experts that are also on the GPU, for the slots elastic KV and prompt loans may
  take. Free RAM sat near 10 GB while serving. Its size against the KV ceiling needs a measured choice.
- Matched C1-C4 runs of the current build against the port's method and a GB10 host.
- The KLD gate for experts computed on the GPU instead of the CPU.
- NVFP4 and IQ3 on equivalent placements, and Qwen3.8-27B.
