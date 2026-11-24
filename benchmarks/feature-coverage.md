# Feature coverage for Flash-Next on N1X

The [BetterBench preset](betterbench-config.json) enables single-stream decode, cold prefill and mixed concurrency. The two native arrival checks below exercise timing that the standard request mix does not guarantee.

Run the production configuration for the pack and GPU split being measured. Record the binary hash, server config, effective native arguments, model pack, actual expert placement, context per slot, and power and clocks under load for each arm. Enable applicable shipped optimizations on both builds. Exercise a feature that requires a different workload or placement in a matching follow-up run. Record when the engine selected another path and why.

| Feature | Workload | Evidence to retain |
| --- | --- | --- |
| Single-stream speculation | BetterBench decode and fixed-output solo prompts | Speculative window setting, draft offers and accepts, output tokens and window time |
| Continuous batching and adaptive drafting | BetterBench mixed C1–C4 and four long structured replies | Four full-context slots, completion counts, aggregate throughput, TTFT, and committed tokens per window. Leave the verification row budget automatic. |
| Cold chunked prefill | BetterBench cold depth sweep | Actual token counts, cached tokens, prompt rate and TTFT at each depth |
| Cold prompt joining active decode | Submit a long fourth prompt after each of three incumbents has emitted at least 16 tokens | Zero reused tokens, positive native overlap windows and chunk jobs, progress on each incumbent, joiner TTFT and the longest incumbent gap during the join |
| Short arrival interrupting a long prefill | While a long prompt is being read, submit a shorter request with a free slot | Native yield at a chunk boundary, successful resume, and completed outputs for both requests |
| Prompt checkpoints and prefix reuse | Repeat a shared-prefix request with caching enabled | Reused and newly read tokens on cold and warm requests |
| Elastic KV, expert placement and segmented scratch | Exercise eligible placement and context sizes | Actual resident experts, KV growth, scratch or loan selected, and file staging. A fully resident run can leave adaptive swaps idle. |

Keep diagnostic traces, forced row budgets, alternative kernels and incompatible placement modes in separately labeled measurements. Run each new performance flag through the same inventory when it lands. Native serving logs show which path the engine actually selected.

On a single N1X, the plain Flash-Next server selects the four-slot full-context configuration automatically. With no engine flags, a live HTTP request completed and all four slots reported 262,144 tokens of context. Explicit slot configuration takes precedence.

## Run the cold join

Use the repository's serving Python environment and CUDA DLL path. The following PowerShell example uses the same pinned expert placement as the November 24 comparison. Use a new output directory for each arm. For an automatic-placement comparison, omit the two placement options in both arms.

```powershell
python benchmarks/cold_join.py --engine $engine --model $model --out $armDir -- --expert-cache 24576 --resident-budget-gib 1
```

This starts a fresh native server through the serving package's `ProcessEngine`. It uses four 262,144-token slots, INT8 KV, a speculation cap of eight, and the engine defaults for prompt checkpoints, chunk sizing and adaptive swaps. Supply any additional production native options after `--`. The check uses the repository's fixed token-ID fixtures. Fresh servers keep the long prompt cold while still allowing ordinary prefix reuse on the short requests.

Run this alongside the three official BetterBench phases. Alternate baseline, candidate, candidate, baseline for comparisons and record power alongside each run. The process returns a failure if the long prompt reused tokens, if native overlap did not execute, or if an incumbent stopped making progress through the join. `launch.json`, `engine.log` and `result.json` retain the settings and evidence. Join TTFT is the time from submission to the first generated token at the native client.

For the reverse arrival, start a fresh arm and pass `--scenario short-during-prefill` before the native options. Use a 4,096-token prefill chunk in both builds to make the boundary reproducible.

```powershell
python benchmarks/cold_join.py --engine $engine --model $model --out $yieldDir --scenario short-during-prefill -- --expert-cache 24576 --resident-budget-gib 1 --prefill 4096
```

The client waits for progress on a 49,966-token read, then submits a short request. It verifies that the short reply finishes before the long prompt's first token and that the engine resumes from the exact token position where it yielded.
