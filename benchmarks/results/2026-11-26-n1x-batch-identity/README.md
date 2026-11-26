# Batched requests match solo ones with the caches on · November 26, 2026

The [four-slot identity gate](../2026-11-26-n1x-join-and-sparkdash/identity-c4) found 5 of 32 batch runs differing from
their solo runs once the prompt and conversation caches were on. This run finds the cause, fixes it, fixes an engine
stop the fix exposed, and measures what both cost. It also tests a change to how a four-chat window shares its draft
rows.

Same RTX Spark N1X laptop (128 GB, 96 GB GPU split, physical cooling) as the
[November 25 runs](../2026-11-25-n1x-rerun). The build under test is the engine source of this change. Every run uses `UD-IQ3_XXS`, `--expert-cache 24576 --kv int8` for the
gate and the server's defaults otherwise.

## The cause: where a prompt read stops

A prompt is read in parts when the prompt cache is on. A solo read stops at the turn boundary (the last
`<|im_start|>`), saves a checkpoint there, and then reads the new turn's header. A short read into a slot beside
decoding slots skipped that stop and read the prompt in one piece, to save the extra part's setup.

Where a read stops changes its rounding. The GDN layers' chunks fall differently, and the second part reads the first
part's keys and values back from the INT8 cache. The engine logs show this exactly. Every run that differed had its
prompt read in one piece beside decode, where the solo run read it in two:

| Prompt | Solo read | Read beside other slots | Differed |
| --- | --- | --- | --- |
| prose (19 tokens) | 12 + 6 rows | 18 rows, one piece | 3 of 3 such rotations |
| code (46 tokens) | 39 + 6 rows | 45 rows, one piece | 2 of 2 such rotations |
| struct (98 tokens) | 91 + 6 rows | 97 rows, one piece | 0 of 2 (no near tie) |
| conv (38 tokens) | 31 + 6 rows | resumed at 31 from a checkpoint | never |

Keeping the solo read's stops for a read beside other slots makes the gate pass. This first used
`--prompt-cache-root 0` with no code change, then the engine change. It is now the default; `--exp slot_solo_split=0`
gives the old rule.

| Gate run (`tools/greedy_identity.py`, 32 batch runs each) | Result |
| --- | --- |
| `--slots-dirty`, before (drafting and both caches on) | 5 of 32 differ |
| `--slots-dirty`, `--prompt-cache-root 0`, no code change | **identical** |
| `--slots-dirty` with the solo split | **identical** |
| `--slots-dirty` with the solo split and value rows (below) | **identical** |
| `--slots --spec 8` with value rows | **identical** |
| Final build, defaults: `--slots`, `--slots --spec 8`, `--slots-dirty` | **identical**, all three |

## An engine stop the fix exposed

With the solo split, a 4,270-token prompt joining three chats now takes a turn checkpoint while the chats decode. In
the second of two server runs with the solo split, the first such join stopped the engine with
`dense_tc: operation failed due to a previous error during capture`.

The read's own thread saved the checkpoint, beside the decode windows. The save synchronizes the whole device. When the
main thread was capturing a window graph at that moment, the device-wide sync invalidated the capture. That window
needed a row count not captured yet, after the read had moved experts out of VRAM. Before the fix the same save ran
only for prompts longer than one 8,192-token chunk, which may be why it never showed in earlier runs.

The checkpoint is now taken on the main thread after the read, which is the thread that captures. The read's thread
still returns the lent experts beside the windows. After the fix, 30 joins of 300 to 6,000 tokens with the solo split
ran without an error.

A solo long prompt also goes through the read's thread, so its checkpoint moved too. Fresh 4K and 16K prompts on four
alternating servers (this build, `209a14bb`, this build, `209a14bb`), three of each per server: median first token
2.06 s against 2.07 s at 4K, and 7.13 s for both builds at 16K. [Per request](solo-first-token)

## What the solo split costs

`benchmarks/join_bench.py`: three chats stream, then a fresh prompt arrives. Four alternating server runs (split, old, split, old),
three joins per depth each, so six per cell; the table gives medians. Both arms include the checkpoint fix above.

| New prompt (actual tokens) | First token, old rule | First token, solo split | Longest pause in a chat, old | solo split |
| ---: | ---: | ---: | ---: | ---: |
| 365 | 0.59 s | 0.58 s | 0.50 s | 0.49 s |
| 1,618 | 1.11 s | 1.10 s | 1.01 s | 1.01 s |
| 2,690 | 1.88 s | 1.84 s | 0.39 s | 0.44 s |
| 4,271 | 2.71 s | 2.73 s | 0.47 s | 0.44 s |
| 6,436 | 3.67 s | 3.77 s | 0.61 s | 0.58 s |

At most 0.1 s more to the first token, and the longest pause in a chat did not grow. The new prompt's next turn also
finds a turn checkpoint,
which the old rule did not save for a read beside other slots under 8,192 tokens.

## Draft rows by expected value (`--exp batch_rows_value=1`, off by default)

At four chats a verify window has 8 rows (16 when the row-cost scheduler finds it pays). When the chats want more, the
default splits the rows about evenly. The option gives every chat its own row, then each further row to the chat whose
next draft is most likely to be kept. That likelihood comes from the draft-length rule's table, scaled by how each
chat's drafts have actually run. Only which drafts get verified changes, so output is unchanged: the gate above
passed with it on, and both arms of the four-chat test below produced identical text.

**Four greedy chats, 700 tokens each** (story, Python module, JSON list, explanation), thinking off. Four alternating
server runs, three rounds each.

| | Even split | By value |
| --- | ---: | ---: |
| All four done (first to last token) | 131.4 tok/s | 131.6 tok/s |
| Mean time for a chat to finish | 19.6 s | 17.8 s |
| Slowest chat (the story) | 33.1 tok/s | 32.9 tok/s |
| Expected drafts kept per over-budget window (engine trace) | 3.52 | 3.87 |

Three of the four chats finish 10–20% sooner. The batch still ends when the story does, since its drafts are rarely
kept either way.

**BetterBench 0.6.1 concurrency** (temperature 0.7, 48 requests per level, one server per arm in the order
even, value, value, even, each piece after a cool-down and kept only with the GPU in P0):

| Level | Even split, tok/s | By value, tok/s | Change | Per-chat decode p10 / p90, even | by value |
| --- | ---: | ---: | ---: | ---: | ---: |
| C2 | 103.1, 105.3 | 102.6, 105.2 | −0.3% | 46.0 / 71.5 | 45.1 / 72.5 |
| C3 | 115.1, 114.6 | 114.9, 116.3 | +0.6% | 36.3 / 49.2 | 34.3 / 54.8 |
| C4 | 119.0, 117.5 | 118.8, 120.3 | +1.1% | 28.5 / 35.1 | 26.8 / 41.3 |

The aggregate change is within the spread between same-setting arms. The split widens the gap between chats: at C4
the slowest tenth decode 6% slower and the fastest tenth 18% faster, and the 90th-percentile first token rises from
588 to 635 ms. It stays off by default. Busy GPU power was 51.6–53.2 W in every piece.

## Long prompts read beside three chats

A check outside the gate: a fresh 3K, 6K or 12K prompt, asked for a summary (96 greedy tokens). It is read once solo
and once beside three decoding chats, each on a fresh server, and the two replies are compared.

| Server setting | 3K | 6K | 12K |
| --- | --- | --- | --- |
| defaults, old rule | differs | differs | differs |
| defaults, solo split | differs | identical | differs |
| `--short-read 0`, solo split | **identical** | **identical** | differs |
| `--short-read 0`, old rule | identical | identical | identical |

- **The default `--short-read 64` is a known difference.** A solo request reads its last few prompt tokens through
  the decode windows. A request in a batch slot cannot, because the verifier holds the other slots' state, so it reads
  them through the prompt path. That is
  [a different computation](../../../docs/precision/qwen38-flash-next.md#prefill), which is why the gate runs with
  `--short-read 0`.
- **A long prompt's chunk size follows free memory.** Solo, the 13,005-token prompt was read in one 22,016-token
  chunk. Beside three decoding chats, with less memory free, it was read in two. Different chunk boundaries round differently. Fixing them for every read would
  cost solo long-prompt reads about 2% (sparkDash read 2,350 tok/s at 8K against 2,406 at 16K), so this run leaves
  it as it is.
- **These replies are a weak test.** Each depth is one prompt with a fresh lead nonce, so each arm saw different text.
  A reply changes only where the prompt meets a near tie, as the old-rule `--short-read 0` row shows. The gate above,
  with 32 runs over prompts that reach near ties, is the measurement.

## Files

- [`gate/`](gate): the identity gate runs. `final-*` are the final build with its defaults, `dirty-pcr0` is the
  earlier build `209a14bb` with `--prompt-cache-root 0`, and the rest are the flag runs. The run that found the 5
  differences is [`slots-dirty.txt`](../2026-11-26-n1x-join-and-sparkdash/identity-c4/slots-dirty.txt).
- [`join/`](join): `join_bench.py` output, one file per server run.
- [`four-chats/`](four-chats): per-round rates, GPU samples and the four replies of each run.
- [`betterbench/`](betterbench): the concurrency pieces and each arm's server command.
- [`long-prompts/`](long-prompts): the solo and joined replies at each depth.
- [`solo-first-token/`](solo-first-token): the solo first-token check, `N-new` this build and `N-old` `209a14bb`.
