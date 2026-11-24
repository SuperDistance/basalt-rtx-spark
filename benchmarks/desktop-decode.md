# Qwen3.8-Flash-Next on discrete GPUs

The RTX Spark N1X recorded 90.79 tok/s weighted single-stream decode on BetterBench with UD-IQ3_XXS. CPU and GPU draw averaged 45.28 W combined during busy decode samples. The community results below used the same model family with different weights, prompts and measurement intervals. [N1X run and power method](results/2026-11-24-n1x-verified-default/power-method.md)

| GPU | Engine | Weights | Reported single-stream decode |
| --- | --- | --- | ---: |
| GeForce RTX 4080 SUPER desktop, Ryzen 9 5950X | Strata | IQ3_S | 90.2 tok/s |
| Same RTX 4080 SUPER and CPU | Strata | IQ3_XXS | 115.8 tok/s |
| Radeon AI PRO R9700 desktop, Ryzen 9 9950X | Radiance | Q4 | 90–100 tok/s |
| GeForce RTX 4090 Laptop GPU, Core i9-13900HX | Strata | IQ3_S | 51.2 tok/s |

The RTX 4080 SUPER report gives median decode across prompt lengths from 1K to 123K tokens. It lists average GPU board power of 257 W for both packs, sampled across the suite including gaps between requests. The N1X measurement covers CPU and GPU during busy decode. [Strata desktop benchmark](https://github.com/Niko1221/Strata/issues/2314#issuecomment-6226756317)

The R9700 figure comes from an owner running StillDeadcode’s Radiance on a single card. The RTX 4090 Laptop figure is the median following a 4,096-token cold prompt with a 131,072-token context limit. [R9700 owner’s report](https://www.reddit.com/r/ROCm/comments/1x6j8an/comment/phvaz73/) · [Strata laptop report and logs](https://github.com/Niko1221/Strata/blob/2af30ccaa0729177c4725ee2073f72bf550e2f86/docs/bench-results/2026-10-04-community-rtx-4090-laptop/README.md)
