# AGENTS.md

Basalt RTX Spark is a native Windows Arm64 inference engine for the RTX Spark N1X laptop. Its integrated GPU has CUDA compute capability 12.1 (`sm_121`) and shares memory with the CPU. Work on Flash-Next and Qwen3.8-27B is measured on this target. The inherited x64 build uses `sm_120` for discrete Blackwell GPUs.

The C++/CUDA engine is in `src/` and `include/`. Model implementations live in `src/models/<model>`. The Python server in `serve/` provides OpenAI- and Anthropic-compatible APIs.

## Rules

[CONTRIBUTING.md](CONTRIBUTING.md) is the project's rulebook and applies to agents in full: scope, platform and model layering, one fast path, what every change must include, benchmark hygiene, commits and docs. The rules below are only for agents.

- **Design the end state first.** Before writing code, design the architecture properly and read similar code. If a model already implements the feature, reuse that logic or make it shared. Do not propose features or improvements before testing them and checking the architecture.
- **Flag odd numbers.** Sanity-check every result before reporting it (size against speed, run-to-run spread, earlier numbers) and report anything odd with its suspected cause.
- **Read before answering.** Check the code, docs and existing results before stating a fact about Basalt.
- **No amends, no force-pushes.** Always make new commits.
- For performance benchmarks, enable the applicable shipped feature flags, record effective settings and verify activation. Run the [feature coverage workloads](benchmarks/feature-coverage.md), including a cold fourth prompt arriving during three active decodes, alongside BetterBench. Report power, join TTFT and incumbent stalls.
- After each verified performance improvement, compare the current engine with measured competing engines in the README tables and charts, and state the workload and power conditions. Keep comparisons between this project's revisions in the detailed benchmark records. Commit and push the tested performance branch to the private `SuperDistance/basalt-rtx-spark-dev` repository (git remote `dev`); its history is private and is never pushed to the public `SuperDistance/basalt-rtx-spark`. Bring tested changes to the public repository's `main` as new commits, then update the README and charts there. Every commit, branch, pull request and issue text on the public repository is authored by SuperDistance alone: no co-author trailers, no AI attribution, no `claude/` or `codex/` branch names. Other remotes and release publication require separate approval.
- **Don't write tools nobody asked for.**
