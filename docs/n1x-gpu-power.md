# GPU power on the N1X

Clocks and power that change Flash-Next speed on the RTX Spark N1X laptop, apart from the engine. Every speed number
on the [GPU-first page](n1x-gpu-first.md) and the [performance notes](n1x-performance.md) states its power state because of these.

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

The slide follows a fixed timeline after any write of the laptop's thermal profile (a profile switch or a restore of the
same profile), the decline described in NVIDIA's open-gpu-kernel-modules issue 1521: about two minutes at the full
budget (about 57 W, 1.75 GHz), then power falling about 5 W a minute to a floor of about 27.7 W and 1,040 MHz roughly
eight minutes after the write, with throttle reason 0x400 set the whole time while the temperature falls. Clock caps do
not move the floor. A run longer than about eight minutes measures the floor; a shorter one lands somewhere on the
decline, depending on when the profile was last written.

CPU load appears to draw on the same budget. In an alternating A/B of the n-gram table's buffered reads, which kept most CPU
cores busy during a prompt, against its unbuffered copy, the buffered arms' GPU drew 19-41 W and the unbuffered arms'
25-47 W over the same prompts ([prompt path](n1x-prompt-path.md#the-n-gram-tables-own-copy)). Builds and other
CPU-heavy work beside a benchmark likely lower its numbers.
