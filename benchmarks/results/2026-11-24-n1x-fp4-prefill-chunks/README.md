# Why cold FP4 prefill dipped at the 32K BetterBench target

On the RTX Spark laptop, the released engine sometimes borrowed GPU memory from expert weights to read a 23.6K-token prompt. That choice forced expert data to be read from file during the prompt and again when the engine restored the loan. The [full FP4-Local BetterBench run](../2026-11-24-n1x-fp4-local) saw this at the nominal 32K depth, while the 64K depth used smaller chunks and separate scratch space.

## Same prompt, different memory headroom

The same release binary processed the same **23,633 input IDs** in all eight runs. Each arm started fresh with 24,576 GPU expert slots and 3.11 GiB of host expert backing. A used bare automatic chunking. B used the existing `--prefill auto:16384` setting, which chose two balanced chunks. Each pair ran A/B/B/A. We repeated the experiment with a 3,072 MiB VRAM reserve to reproduce the file-backed path under a controlled memory constraint. [All eight runs, placement, output hashes and rail samples](chunk-control.json)

| VRAM reserve | First token with automatic chunks, A1 / A2 | First token with capped chunks, B1 / B2 | Expert blobs staged per read, A / B |
| --- | ---: | ---: | ---: |
| 700 MiB, release default | 10.36 / 10.36 s | 10.03 / 10.32 s | 0 / 0 |
| 3,072 MiB, controlled constraint | 15.77 / 17.66 s | 10.08 / 10.09 s | 1,586 / 0 |

With the larger reserve, replacing the file-backed loan with two chunks cut mean time to first token by **39.7%**. With the normal reserve, the original 23,808-token scratch layout already fit and the difference averaged **0.19 s**. All eight arms produced the same two reply IDs and the same 128 IDs on a subsequent structured decode control. The two-chunk arm used a different startup loan ceiling, so the result measures the complete configuration change.

The constrained A arms averaged 24.6 W GPU draw and 23.7–25.4 W CPU draw from submission to first token. B averaged 47.4–47.8 W GPU and 12.5–13.7 W CPU over that interval. These are whole-request averages at approximately two-second sampling intervals on AC. The GPU limit was unavailable. The default-reserve experiment used the same cooling and HP Performance profile. Windows was set to Balanced. The 50.7 GiB n-gram sidecar used the Windows file cache throughout.

## Reproduction

Use release `v0.4.1-rtx-spark.2` and the FP4-Local conversion. Generate the input with BetterBench 0.6.1 `make_prefill_messages(32000, "n1x-32k-chunk-abba")`. Wrap its user content as shown below, then send the resulting text to Basalt’s `/tokenize` endpoint with special tokens enabled. The JSON record includes a SHA-256 of the token IDs serialized with Python `json.dumps(ids)`.

```python
text = "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```

Use the native batched client on GPU 0 with four 262,144-token slots, INT8 KV, MTP 8, `--expert-cache 24576 --resident-budget-gib 3.11`, temperature 0, seed 17 and an output limit of 16. Follow with `data/bench/struct.ids` at a 128-token output limit. Use a fresh engine for each arm. Record native prefill rows, file staging, reuse, first-token latency, output IDs and telemetry. The capped runs used `auto:16384` and the pressure experiment added `--vram-reserve-mib 3072` to both arms. [Chunk selection and scratch allocation](../../../src/models/qwen4exp/serve/prompt.cpp)

## Keeping the original chunk width

The optimized engine keeps all FP4 experts on the GPU and tries a smaller staging ring if an idle prompt's ordinary scratch space exceeds the memory budget. It retains the original chunk size, KV geometry and fallback loan. In this case the ring went from **384 slots to 64**, reducing the requested allocation from **6.20 GiB to 5.38 GiB**. The saved 843.9 MiB can decide whether the prompt needs to evict and reload experts. [Source change](https://github.com/SuperDistance/basalt-rtx-spark/commit/1bc565f277a3fb02c6182ea5ba0e9086b3f637e0)

| VRAM reserve | Released engine A1 / A2 TTFT | Smaller-ring build B1 / B2 TTFT | Blobs staged per request, A / B |
| --- | ---: | ---: | ---: |
| 2,560 MiB | 14.20 / 14.16 s | **10.41 / 10.38 s** | 1,586 / **0** |
| 3,072 MiB | 16.06 / 14.17 s | 17.27 / 14.24 s | 1,586 / 1,586 |

Both binaries used bare automatic chunking. With a 2,560 MiB reserve, mean first-token latency fell **26.7%**. GPU draw during those candidate reads averaged **44.1–47.0 W**, compared with **29.2–33.5 W** on the release. The 3,072 MiB series ran first and recorded four ordinary loans. A follow-up at 2,560 MiB tested the narrower range of headroom in which the smaller ring fits. The trace records every repeat and the free Windows commit before each read. [A/B/B/A results, native path and power](resident-ring-abba.json)

### Output parity

A separate greedy run gave both binaries a **23,625-token** archive and requested the accession numbers for three records near **5.5%, 51% and 94.3%** of its length. Both returned all three correct values, with **25 identical output IDs**. All **32 recorded full-vocabulary logit hashes** also matched. The optimized arm logged the smaller ring and staged zero blobs. The released arm used the file-backed loan. This check exercised the changed allocation path. [Token IDs, expected answer and native row hashes](resident-ring-parity.json)
