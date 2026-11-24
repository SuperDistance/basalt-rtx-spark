# Benchmarks

Run the [production feature coverage workloads](feature-coverage.md) alongside each BetterBench comparison. The cold-join check submits an uncached fourth prompt after three streams are producing output and verifies their progress during prefill.

Serving results on RTX Spark machines, measured with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.1 and
one shared config, so runs on different laptops (and other Spark machines) can be compared. Contributions are welcome:
add a folder under `results/` with your run (see [Contributing a result](#contributing-a-result)).

## Results

All three engines ran Qwen3.8-Flash-Next on the same RTX Spark N1X laptop with 128 GB of unified memory and a 96 GB GPU allocation. BetterBench 0.6.1 used the same 392-request workload in each run.

| Engine | Weight pack tested | Weighted single-stream decode (tok/s) | Short-prompt time to first token (ms) | Four-stream mixed throughput (tok/s) |
| --- | --- | ---: | ---: | ---: |
| Basalt RTX Spark | `UD-IQ3_XXS` | **90.79** | 289 | **110.50** |
| Basalt + Arm64 port | `UD-IQ3_XXS` | 87.52 | 622 | 85.32 |
| vLLM for RTX Spark 0.2.0 | `FP4-Local` (NVFP4 / MXFP8) | 54.19 | **271** | 100.75 |

Both Basalt engines used INT8 KV and MTP 8. vLLM used FP8 target KV, BF16 draft KV and MTP 7. On November 24, busy GPU draw averaged 39.3 W for Basalt RTX Spark and 34.5 W for vLLM during the four-stream test. The Arm64 port ran separately on November 23 and averaged 40.8 W across busy samples over its full run. The driver did not report a GPU power limit. [November 24 measurements](results/2026-11-24-n1x-verified-default) · [Arm64 port measurements](results/2026-11-23-hp-omnibook-x-n1x-128gb/basalt-arm-sm121-2ec7a84.json)

![Single-stream decode by scenario · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-decode.svg)

![Aggregate throughput at one to four streams · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-concurrency.svg)

![Cold prefill throughput by prompt depth · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-prefill.svg)

### Supported quantizations and single-stream decode rates

The documented Flash-Next weight packs for both Basalt engines include `NVFP4-DQ6_K`, `NVFP4-DQ8`, `UD-IQ1_S`, `UD-IQ1_M`, `UD-IQ2_XS`, `UD-IQ3_XXS`, `UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`, `Q8_0`. Basalt RTX Spark also ran NVIDIA `FP4-Local` converted to `.basalt`. The vLLM for RTX Spark 0.2.0 recipe for Flash-Next uses `FP4-Local`. The table above gives BetterBench throughput for the packs tested. [Basalt pack catalog](../docs/models/qwen38-flash-next.md#pack) · [vLLM release recipes](https://github.com/devjeonghwan/vllm-for-rtx-spark/blob/e9412963ff4619da5f97f8b5e3c2dd274967df87/README.md)

Earlier N1X single-request tests measured output tokens per second for three prompt types. The ranges span two runs per pack.

| Engine / model | Weight pack | Conversation | Structured output | Prose |
| --- | --- | ---: | ---: | ---: |
| Basalt RTX Spark / Flash-Next | `UD-IQ3_XXS` | 79.0–84.1 | 141.3–147.0 | 73.3–75.4 |
| Basalt RTX Spark / Flash-Next | `FP4-Local` | 81.6–82.7 | 133.5–136.1 | 68.3–69.2 |
| Basalt + Arm64 port / Flash-Next | `NVFP4-DQ6_K` | 80.8 | 140.3 | 67.8 |
| Basalt RTX Spark / Qwen3.8-27B | `QUASAR-NVFP4` | 48.6 | 143.7 | 33.1 |

The alternating Flash-Next IQ3 and FP4-Local runs used INT8 KV, MTP 8 and fully GPU-resident experts. Busy GPU draw was 36.8–42.4 W for IQ3 and 38.4–41.4 W for FP4-Local. The Arm64 port used FP16 KV for its NVFP4-DQ6_K test. The 27B run used INT8 KV and a 16-token block drafter. The other listed packs have no comparable N1X throughput measurement in these runs. [Per-run measurements](https://github.com/SuperDistance/basalt-rtx-spark/blob/234f26ef/docs/n1x-handoff.md#reproduced-on-november-23) · [Arm64 port fixtures](../docs/arm64.md#speed-on-the-rtx-spark-n1x)

The [benchmark archive](results) contains the individual runs and scheduler experiments.

From the repository root, regenerate the charts with PowerShell.

```powershell
$run = 'benchmarks/results/2026-11-24-n1x-verified-default'
$port = 'benchmarks/results/2026-11-23-hp-omnibook-x-n1x-128gb/basalt-arm-sm121-2ec7a84.json'
python benchmarks/plot.py --out benchmarks/img --prefix engines `
  "$run/basalt/results.json" 'Basalt RTX Spark (UD-IQ3_XXS)' `
  $port 'Basalt + Arm64 port (UD-IQ3_XXS)' `
  "$run/vllm/results.json" 'vLLM for RTX Spark (FP4-Local)'
```

## Running it

1. Install BetterBench 0.6.1 (`pip install -e .` in a clone of its repository; Python 3.10+).
2. Start the server with the config you are measuring (the one these runs used is in the [setup guide](../docs/rtx-spark-setup.md#install)).
3. Run, from the BetterBench checkout:

   ```
   python -m betterbench.cli run --no-update-check --endpoint http://127.0.0.1:8080/v1 --model flash-next \
     --config <this repo>/benchmarks/betterbench-config.json --max-model-len 262144 \
     --note engine=<build> --note quant=<pack> --note hardware=<machine and GPU split> \
     --name <name> --out <name>.json
   ```

   The config is BetterBench's community settings: temperature 0.7, top-p 0.95, top-k 20, a cold prefix cache, 3 warm-up
   and 20 measured passes per category, the 2K-64K prefill sweep, and 48 requests at each concurrency from 1 to 4. A run
   takes 35-45 minutes.

4. Log the GPU's power and clock beside it, for example
   `nvidia-smi --query-gpu=timestamp,pstate,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv -l 5 > <name>.gpu.csv`.

### The power state matters

On the N1X the GPU boosts to about 55-60 W, and on some laptops it then settles at a sustained floor of 22-28 W about
eight minutes into a load, which takes 20-40% off the later phases of a run. Holding the budget up (for example by
re-applying the laptop's performance thermal profile when it starts to decline) keeps it around 40-55 W. Say which
state your run was in and include the GPU log; compare runs only when they ran in the same state, ideally back to back.
See [GPU power on the N1X](../docs/n1x-gpu-power.md).

## Contributing a result

Add `results/<date>-<maker>-<model>-<chip>-<ram>/` with:

- the BetterBench JSON (`<engine>.json`), and the HTML report if you like;
- the GPU log (`<engine>.gpu.csv`);
- a `README.md` filled in from [TEMPLATE.md](TEMPLATE.md): the machine, the GPU memory split, driver, OS build, power
  source and mode, fan and thermal profile, anything holding the power budget up, the engine commit and server config,
  the model pack, and anything else that ran during the run.

Then add a row to the table above and open a pull request. Results from other engines on the same machine are
welcome too, run with the same config.
