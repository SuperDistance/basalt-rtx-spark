# AGENTS.md

Basalt is a local inference engine built for NVIDIA Blackwell (sm_120) on Linux and Windows x64. A model runs on one Blackwell GPU, optionally with more GPUs as peer expert tiers, and, for the mixture-of-experts models, system RAM holding the rest (a dense model runs fully on one GPU). The C++/CUDA engine is in `src/` and `include/`, each model's own code under `src/models/<model>`, and the Python server with an OpenAI- and Anthropic-compatible API is in `serve/`.

## Rules

[CONTRIBUTING.md](CONTRIBUTING.md) is the project's rulebook and applies to agents in full: scope, platform and model layering, one fast path, what every change must include, benchmark hygiene, commits and docs. The rules below are only for agents.

- **Design the end state first.** Before writing code, design the architecture properly and read similar code. If a model already implements the feature, reuse that logic or make it shared. Do not propose features or improvements before testing them and checking the architecture.
- **Flag odd numbers.** Sanity-check every result before reporting it (size against speed, run-to-run spread, earlier numbers) and report anything odd with its suspected cause.
- **Read before answering.** Check the code, docs and existing results before stating a fact about Basalt.
- **No amends, no force-pushes.** Always make new commits.
- For performance benchmarks, enable the applicable shipped feature flags, record effective settings and verify activation. Run the [feature coverage workloads](benchmarks/feature-coverage.md), including a cold fourth prompt arriving during three active decodes, alongside BetterBench. Report power, join TTFT and incumbent stalls.
- After each verified performance improvement, show measured competing engines alongside Basalt in the README graphs when comparable data exists, and update the README and benchmark charts with the measured workload, baseline, candidate and power conditions. Commit and push the tested performance branch to the private `SuperDistance/basalt-rtx-spark` remote, and update the README and charts on its default branch. Other remotes and release publication require separate approval.
- **Don't write tools nobody asked for.**
