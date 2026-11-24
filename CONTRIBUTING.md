# Contributing to Basalt RTX Spark

This fork runs Qwen3.8-Flash-Next and Qwen3.8-27B natively on RTX Spark N1X laptops with Windows 11 on Arm. Start with the [Windows Arm64 setup](docs/rtx-spark-setup.md) and [N1X benchmark instructions](benchmarks/README.md).

## Scope

- **Target the N1X.** The Arm64 build compiles its main CUDA kernels for the integrated GPU's compute capability 12.1 (`sm_121`). CPU expert kernels use Arm NEON. Weight and KV placement must account for the GPU memory split, available RAM, and the Windows commit limit.
- **Preserve the inherited builds.** Linux Arm64 and the upstream x64 build for discrete Blackwell (`sm_120`) share engine code. Keep architecture selection in the build and platform layers. The available GPU backends and CPU fallback paths stay within Basalt's existing CUDA design.
- **Engine code is platform-neutral.** No OS or compiler names and no raw OS calls outside the platform layer; a missing seam is added there with a Linux and a Windows body. See [platforms](docs/platforms.md).
- **Engine log lines carry their level.** A line to stderr goes through `include/basalt/log.hpp` (`basalt::log::info`, `warn` or `error`), which writes `basalt <area>: [warning: |error: ]<text>`; the server and the terminal read the level from that marker and never from the words. A warning is a fallback or a lost function, an error is a failure or a refusal, everything else is info. `tools/check_stderr_logging.sh` fails when engine code names stderr anywhere else. Lines are terse facts (what was decided, its value, at most the flag to change it), and the text of lines the server parses does not change.
- **Keep runtime behavior in the engine and server.** The Windows distribution's setup and start scripts handle environment setup and launch. Implement serving, benchmarking and conversion through their existing entry points.
- **One fast path.** No per-user tuning flags or alternative kernels selected at run time. A change that wins replaces the old path; a change that doesn't win isn't merged. A/B and diagnostic flags are off by default and are not documented as settings. User-facing settings are limited to model, context, multi-GPU and serving options.
- **No shims and no dead code.** A removed or renamed flag, option, field or code path is simply gone: no alias, no fallback, no "was retired" message. The normal unknown-flag error covers it. Code nothing uses is deleted in its own commit.
- **Defaults are measured.** A default, or an automatic pick that replaces one, changes only after a measured run on hardware where it is picked.
- **No environment-variable knobs.** Everything the engine reads is a command-line flag or a config entry; diagnostics (profiling, timing, logs) are flags too, off by default.
- **Decode comes first.** A prefill gain never costs decode speed; every prefill change shows a decode number too.

## Code structure

- **Shared code is model-agnostic.** It sees a model only through the adapter, a model's own code stays in `src/models/<model>`, and a change for one model leaves the others' output and speed as they were. See [adding a model](docs/adding-a-model.md).
- **Logic exists once.** What two models need lives in shared code and takes each model's numbers as parameters. Follow the existing pattern before adding a new one.
- **One serving loop** runs every request of every model; a model plugs into it and never forks it. See [persistent serving](src/program/serve/README.md).
- **The engine does the work, the server serves.** Tokenizing and per-request work run in C++. `serve/` parses, routes and streams, names no model, and never repairs model output; constrained decoding (grammar) handles structure.
- **Build structure: no slow TU.** A clean Release build must finish in under 10 minutes on 32 threads, and nvcc compiles a translation unit on one core, so the build takes as long as its slowest `.cu`. Keep every `.cu` to about a minute or two alone: a kernel that is instantiated in several variants (like `dm_peer_kernel`) gets one `.cu` per variant, behind a getter that returns its address (no relocatable device code), and shared device code goes in a header with no kernel in it. A `.cu` does not include another `.cu` (`tools/check_cu_includes.sh` fails on a new one; `tools/cu_includes.allow` lists the few that must inline). Time a new or changed `.cu` alone (`nvcc --time`) and report it with the change.

## What every change must include

1. **A parity test** for every new or changed kernel: random inputs plus the real model geometry, compared against (a) the current Basalt kernel it replaces and (b) the plain CPU reference for that operation (dequantize, then accumulate in double precision). Report the max absolute and relative error against both. A kernel meant to be bitwise-neutral must match the frozen output hashes of the current kernels. The output for a token must not depend on the batch/window width or the token's position in it; the test must check that.
2. **A profile.** Give a microbench line for the kernel: µs per call and achieved GB/s (or TFLOP/s) against the bytes or FLOPs it must move. Give an end-to-end number from `basalt-bench`: ms per decode window and passes/s, not only tok/s, which drafting luck moves.
3. **A quality check:**
   - Bitwise-identical output: show the greedy-identity check against the previous build. Pin the expert tiers, since changes in GPU module size move the auto-sized tiers and change rounding.
   - Changed numerics (new reduction order, precision, quantization): run the perplexity/KLD gate against the BF16 reference ([docs/precision/quality.md](docs/precision/quality.md#the-bf16-reference)) through the decode path. Report PPL, mean KLD and top-1 agreement next to the previous build's. Large changes need the full 100-chunk run, not just the 20-chunk quick gate. The `docs/precision/` page that covers the change is updated in the same change, with fresh numbers where it quotes them.
4. **Benchmark hygiene:**
   - Report the N1X memory split, pack, placement, KV and draft settings, concurrency, Windows power mode and cooling profile. Record GPU power, clocks and temperature during the measured phase. Include the driver’s power limit when available. If it is unavailable, label the measured rails and averaging window. [Power methodology](benchmarks/results/2026-11-24-n1x-verified-default/power-method.md)
   - Exercise the applicable shipped defaults. Run BetterBench and the feature coverage workloads, including a cold fourth request arriving during three active decodes. Record join time to first token and the longest incumbent stall. [Feature coverage](benchmarks/feature-coverage.md)
   - A/B comparisons alternate the two builds in the same session (ABBA), with the expert tiers pinned to the same placement in both. Numbers taken in different sessions drift and aren't comparable.
   - Numbers are cold first runs of the real engine, at a placement real users get. A gain that only exists with experts spilled to the SSD is not a headline.
   - A claim that something can't get faster comes with a per-step breakdown against the measured hardware limit.
   - Keep warm-up and fixture order as `basalt-bench` defines them.

## Code style

- Kernels are `__device__` stage functions that take their block/warp index as an argument, with thin `__global__` wrappers, so they can be fused into persistent kernels. Never use `cp.async.bulk.shared::cluster`; use `cp.async.bulk.tensor.*.shared::cta` for TMA.
- Comments are brief and stateless: what the code does and why. No hardware names, measured numbers, dates or history in comments. Those belong in the commit message or the PR.
- Keep adapted third-party code's license headers intact (ggml/llama.cpp is MIT).

## Commits and pull requests

- Small, logical commits with the message style `area: what it does`. Each commit must build. A small change across areas is one commit.
- No personal names or serving configurations in commit messages, comments or docs.
- Ported or adapted work keeps its original authors as `Co-Authored-By` lines.
- Never amend or force-push shared branches; add new commits.
- No notes, scratch files or generated reports in the repo.
- Docs: no file over 300 lines; split a longer topic into a folder of files; keep docs short, stateless and free of duplication. Web UI text uses proper casing ("Basalt") and no em dashes.
- A PR description lists: what changed, the parity results, the microbench and `basalt-bench` before/after (with the power limit), and the quality-check result.

## Other ways to help

A BetterBench run on another RTX Spark laptop helps show how memory split, driver, and cooling affect this engine. Start with the [benchmark instructions](benchmarks/README.md#running-it). Submit the result, effective settings and sensor samples with the [hardware report template](benchmarks/TEMPLATE.md). For a competing engine, use the same laptop and requests and document the quantization and cache differences.

## License

Basalt is MIT licensed and is based on Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors). By contributing you agree your contribution is licensed under the same terms. See `LICENSE` and `THIRD_PARTY_LICENSES`.
