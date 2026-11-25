# Retrieving records near the context limit

The released FP4-Local engine on Windows Arm64 completed two cold archive lookups using one of its four 262,144-token slots. Flash-Next returned all three requested accession numbers at **64,991 input tokens** and two of three at **259,985**. Each number appeared exactly once in the archive. The question named the three record IDs. The requested records were near 5.5%, 51% and 94.3% of the document. [Expected answers, actual replies and input hashes](results.json)

| Input tokens | Archive records | Exact lookups | Request wall time |
| ---: | ---: | ---: | ---: |
| 64,991 | 3,606 | 3 / 3 | 27.73 s |
| 259,985 | 14,439 | 2 / 3 | 125.78 s |

In the longer request the first lookup asked for **R00794** (810960). Flash-Next returned **947361**, the value at **R00793**. The middle and late lookups were correct. Both requests generated 25 tokens, stopped normally and reported zero cached prompt tokens. These are two retrieval examples on the shipping release, alongside the separate [controlled prefill and parity experiments](../2026-11-24-n1x-fp4-prefill-chunks).

To reproduce a document, look up `seed`, `record_count` and `requested_ids` in the results. Send the generated text as a single user message with the listed API settings. Set the sampling seed to the document seed.

```python
import random

numbers = random.Random(seed).sample(range(100000, 999999), record_count)
rows = "".join(f"Archive R{i:05d} | accession {n:06d}.\n"
               for i, n in enumerate(numbers))
labels = ", ".join(requested_ids)
text = (f"Archive {seed}. Each entry maps a record ID to an accession number. "
        f"Use this archive to answer the question.\n{rows}\n"
        f"Return the accession numbers for {labels}, in that order, "
        "as a JSON list of integers. Return only the list.")
```
