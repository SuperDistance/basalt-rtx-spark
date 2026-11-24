# Former default replay

This is the November 24, 2026 BetterBench 0.6.1 replay of the formerly installed `0cb0b207` executable. It used the same 392-request preset and corpus hash as the current candidate and vLLM runs; all 392 requests succeeded and it reported zero cache hits. Only the result JSON host and endpoint fields were normalized to `RTX Spark N1X` and `http://localhost/v1`; all benchmark values and the corpus hash are preserved.

`c4-batch-summary.txt` retains the dominant mixed C4 engine cohort (2,403 batch windows over 161.423 s). `deep-prompt-residency.json` records the representative ~47K cold-prompt scratch evidence and its power caveat. Raw NVIDIA samples and phase extraction are in `gpu.csv` and `power.json`.
