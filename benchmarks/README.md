# Benchmarks

Run the [production feature coverage workloads](feature-coverage.md) alongside each BetterBench comparison. The cold-join check submits an uncached fourth prompt after three streams are producing output and verifies their progress during prefill.

Serving results on RTX Spark machines, measured with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.1 and
one shared config, so runs on different laptops (and other Spark machines) can be compared. Contributions are welcome:
add a folder under `results/` with your run (see [Contributing a result](#contributing-a-result)).

## Results

All three engines ran Qwen3.8-Flash-Next on the same RTX Spark N1X laptop with 128 GB of unified memory and a 96 GB GPU allocation. BetterBench 0.6.1 used the same 392-request workload in each run.

| Engine | Weighted single-stream decode (tok/s) | Short-prompt time to first token (ms) | Four-stream mixed throughput (tok/s) |
| --- | ---: | ---: | ---: |
| Basalt RTX Spark | **90.79** | 289 | **110.50** |
| Basalt + Arm64 port | 87.52 | 622 | 85.32 |
| vLLM for RTX Spark 0.2.0 | 54.19 | **271** | 100.75 |

Both Basalt engines used UD-IQ3_XXS weights, INT8 KV and MTP 8. vLLM used FP4-Local weights, FP8 target KV, BF16 draft KV and MTP 7. On November 24, busy GPU draw averaged 39.3 W for Basalt RTX Spark and 34.5 W for vLLM during the four-stream test. The Arm64 port ran separately on November 23 and averaged 40.8 W across busy samples over its full run. The driver did not report a GPU power limit. [November 24 measurements](results/2026-11-24-n1x-verified-default) · [Arm64 port measurements](results/2026-11-23-hp-omnibook-x-n1x-128gb/basalt-arm-sm121-2ec7a84.json)

![Single-stream decode by scenario · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-decode.svg)

![Aggregate throughput at one to four streams · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-concurrency.svg)

![Cold prefill throughput by prompt depth · Basalt RTX Spark, Basalt + Arm64 port, vLLM for RTX Spark](img/engines-prefill.svg)

The [benchmark archive](results) contains the individual runs and scheduler experiments.

From the repository root, regenerate the charts with PowerShell.

```powershell
$run = 'benchmarks/results/2026-11-24-n1x-verified-default'
$port = 'benchmarks/results/2026-11-23-hp-omnibook-x-n1x-128gb/basalt-arm-sm121-2ec7a84.json'
python benchmarks/plot.py --out benchmarks/img --prefix engines `
  "$run/basalt/results.json" 'Basalt RTX Spark' `
  $port 'Basalt + Arm64 port' `
  "$run/vllm/results.json" 'vLLM for RTX Spark 0.2.0'
```

## Running it

1. Install BetterBench 0.6.1 (`pip install -e .` in a clone of its repository; Python 3.10+).
2. Start the server with the config you are measuring (the one these runs used is in the [top README](../README.md#building-and-running)).
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
