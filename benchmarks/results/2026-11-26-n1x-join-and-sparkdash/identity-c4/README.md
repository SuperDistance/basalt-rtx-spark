# Four-slot greedy identity · November 26, 2026

`tools/greedy_identity.py` checks a build's batch slots against its own solo requests. Until this run its slot check had
three prompts for four slots, so no "together" run filled all four, and drafting was off. It now has a fourth prompt
(`data/bench/code.ids`). It fails a rotation in which any slot finished before the last one was admitted, and its
`--slots-dirty` mode runs with drafting and both caches on.

Each run is the source build `209a14bb` with `UD-IQ3_XXS`, `--expert-cache 24576 --kv int8`, and 300 greedy tokens per
prompt. Each run covers:
- 4 solo requests,
- each prompt alone in each of the 4 slots,
- the 4 prompts together under 4 slot rotations, with all 4 slots decoding at once in every rotation.

That is 32 batch runs compared token by token with the solo ones.

| Run | Drafting | Prompt and conversation caches | Result |
| --- | --- | --- | --- |
| [slots.txt](slots.txt) | off | off | **identical**, 32 of 32 |
| [slots-spec8.txt](slots-spec8.txt) | MTP 8 | off | **identical**, 32 of 32 |
| [slots-dirty-spec0.txt](slots-dirty-spec0.txt) | off | on | 5 of 32 differ |
| [slots-dirty.txt](slots-dirty.txt) | MTP 8 | on | 5 of 32 differ (the same 5) |

- **Drafting is not the cause.** Requests decoding together with drafting in the shared verify windows match their
  solo runs exactly.
- **The differences come with the caches.** They appear only in "together" runs after earlier runs have filled the
  prompt and conversation caches. Every "alone in a slot" run matched, with the caches warm.
- **Prose:** in 3 of 4 rotations the first token differs (solo `332`, batch `2`), and the next tokens match again.
- **Code:** in 2 rotations one token at position 225 differs.
- **Why it looks like rounding, not corruption:** each difference is one token at a near tie, after which the text
  continues identically. All four prompts share the chat template's opening tokens, so a prompt admitted beside the
  others can resume from another slot's prefix checkpoint and read only its remainder. That splits the prompt read
  differently from the cold solo read.
- **Still to do:** confirm that cause, and measure the logit difference (KLD) between the cold and restored reads.
