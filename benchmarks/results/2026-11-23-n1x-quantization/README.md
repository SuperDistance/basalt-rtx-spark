# Single-stream decode by weight pack

The November 23 N1X single-stream test (UD-IQ3_XXS, FP4-Local, FP4-Local, UD-IQ3_XXS) kept all 24,576 experts resident and 1 GiB of host copies in
each arm. IQ3 used 46.84 GiB for routed weights and FP4-Local used 63.29 GiB. INT8 KV, seed 17, MTP8,
context 16,720 and the fixtures were held fixed. The power limit was unavailable.

| Pack in run order | Conv / struct / prose tok/s | Conv / struct / prose ms per window | Cold 16K tok/s | Busy GPU W |
| --- | --- | --- | ---: | ---: |
| IQ3 | 79.0 / 141.3 / 75.4 | 33.54 / 51.89 / 29.06 | 1,789 | 36.8 |
| FP4-Local | 82.7 / 136.1 / 68.3 | 35.86 / 57.11 / 33.11 | 1,774 | 41.4 |
| FP4-Local | 81.6 / 133.5 / 69.2 | 36.35 / 58.23 / 32.67 | 1,844 | 38.4 |
| IQ3 | 84.1 / 147.0 / 73.3 | 31.52 / 49.87 / 29.90 | 1,627 | 42.4 |

FP4-Local's conversation window committed 2.99 tokens versus IQ3's 2.68, partly offsetting its longer
window. Both emitted 7,230 structured tokens. IQ3 used its existing PLE sidecar for direct reads. FP4-Local
used the OS file cache because creating its 50.7 GiB sidecar exceeded free disk space. This compares the
local packs in their attainable configurations.
