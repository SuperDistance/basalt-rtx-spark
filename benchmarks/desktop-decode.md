# Qwen3.8-Flash-Next on discrete GPUs

This project · Basalt RTX Spark on the RTX Spark N1X recorded 98.92 tok/s weighted single-stream decode on BetterBench with UD-IQ3_XXS (the best of five runs; the median was 97.29). The GPU averaged 54.6 W during busy decode. The external reference-engine results below used the same model family with different weights, prompts, GPUs and measurement intervals; they are context, not matched tests. [N1X runs and power method](results/2026-11-25-n1x-rerun)

| GPU | Engine | Weights | Reported single-stream decode |
| --- | --- | --- | ---: |
| RTX Spark N1X laptop | This project · Basalt RTX Spark | UD-IQ3_XXS | 98.92 tok/s |
| GeForce RTX 4080 SUPER desktop, Ryzen 9 5950X | Reference engine · Strata | IQ3_S | 90.2 tok/s |
| Same RTX 4080 SUPER and CPU | Reference engine · Strata | IQ3_XXS | 115.8 tok/s |
| Radeon AI PRO R9700 desktop, Ryzen 9 9950X | Reference engine · Radiance | Q4 | 90–100 tok/s |
| GeForce RTX 4090 Laptop GPU, Core i9-13900HX | Reference engine · Strata | IQ3_S | 51.2 tok/s |

The RTX 4080 SUPER report gives median decode across prompt lengths from 1K to 123K tokens. It lists average GPU board power of 257 W for both packs, sampled across the suite including gaps between requests. The N1X measurement covers CPU and GPU during busy decode. [Strata desktop benchmark](https://github.com/Niko1221/Strata/issues/2314#issuecomment-6226756317)

The R9700 figure comes from an owner running StillDeadcode’s Radiance on a single card. The RTX 4090 Laptop figure is the median following a 4,096-token cold prompt with a 131,072-token context limit. [R9700 owner’s report](https://www.reddit.com/r/ROCm/comments/1x6j8an/comment/phvaz73/) · [Strata laptop report and logs](https://github.com/Niko1221/Strata/blob/2af30ccaa0729177c4725ee2073f72bf550e2f86/docs/bench-results/2026-10-04-community-rtx-4090-laptop/README.md)
