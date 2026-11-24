# Benchmarks

Run the [production feature coverage workloads](feature-coverage.md) alongside each BetterBench comparison. The cold-join check submits an uncached fourth prompt after three streams are producing output and verifies their progress during prefill.

Serving results on RTX Spark machines, measured with [BetterBench](https://github.com/GGZ14/BetterBench) 0.6.1 and
one shared config, so runs on different laptops (and other Spark machines) can be compared. Contributions are welcome:
add a folder under `results/` with your run (see [Contributing a result](#contributing-a-result)).

## Results

The November 24 [verified-default run](results/2026-11-24-n1x-verified-default) tests the promoted four-slot scheduler against both the former default and vLLM for RTX Spark on the same N1X. All three completed the same 392-request preset and corpus hash. Cross-engine C4 power below uses complete aligned SparkHelper telemetry; vLLM's NVIDIA CSV is incomplete at the end of C4.

| Run | Engine | Busy C4 GPU / measured CPU+GPU+NPU power | Decode, weighted median (tok/s) | First token, short (ms) | Output at 1 / 2 / 3 / 4 streams (tok/s) | Prefill 2K / 8K / 64K (tok/s) |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [Nov 24 verified default](results/2026-11-24-n1x-verified-default) | Former Basalt default `0cb0b207`, UD-IQ3_XXS, INT8 KV, MTP 8 | 41.45 / 45.68 W | 87.55 | 304 | 80.79 / 93.33 / 103.14 / 106.34 | 1,377 / 2,110 / 1,280 |
| [Nov 24 verified default](results/2026-11-24-n1x-verified-default) | Current Basalt `3b27d149`, UD-IQ3_XXS, INT8 KV, MTP 8 | 39.33 / 44.19 W | 90.79 | 289 | 85.09 / 96.36 / 106.41 / 110.50 | 1,465 / 1,787 / 2,190 |
| [Nov 24 verified default](results/2026-11-24-n1x-verified-default) | vLLM RTX Spark 0.2.0, FP4-Local, FP8 target / BF16 draft KV, MTP 7 | 34.46 / 41.90 W | 54.19 | 271 | 53.38 / 78.70 / 93.21 / 100.75 | 1,511 / 1,966 / 2,035 |

All runs used the HP OmniBook X RTX Spark N1X, 96 GiB GPU split, Windows 11 on Arm64, AC and Balanced power. The driver reported no GPU power limit. The weight and KV formats differ, so these throughput measurements do not imply equal model quality. Read the [result package](results/2026-11-24-n1x-verified-default) for raw outputs, phase power and quality context.

![Basalt 3b27d149 current, Basalt 0cb0b207 previous, and vLLM RTX Spark 0.2.0 single-stream decode by scenario](img/verified-default-decode.svg)

![Basalt 3b27d149 current, Basalt 0cb0b207 previous, and vLLM RTX Spark 0.2.0 mixed concurrency](img/verified-default-concurrency.svg)

![Basalt 3b27d149 current, Basalt 0cb0b207 previous, and vLLM RTX Spark 0.2.0 cold prefill by prompt depth](img/verified-default-prefill.svg)

Compared with the former default, current Basalt was +3.7% in weighted decode and +3.9% in mixed C4, while busy GPU power and clock were lower (39.33 W / 2,034 MHz versus 41.45 W / 2,232.5 MHz). At 64K prefill, the current run was faster but also ran at substantially higher power and clock; 8K–32K prefill was slower. See the [full result package](results/2026-11-24-n1x-verified-default) for per-depth power and scratch-residency evidence.

The earlier November 24 [scheduler microbenchmark and joint research build](results/2026-11-24-n1x-four-stream-scheduler) remain available as separate experiments. The fixed four-stream scheduler microbenchmark gained **8.1%** on four 1,024-token replies; the joint research build measured **+0.2%** at mixed C4.

Earlier runs from November 22–23:

| Machine | Engine | Power state | Decode, weighted median (tok/s) | First token, short (ms) | Output at 1 / 2 / 3 / 4 streams (tok/s) | Prefill 2K / 8K / 64K (tok/s) |
| --- | --- | --- | ---: | ---: | --- | --- |
| [HP OmniBook X, N1X, 128 GB, 96 GB split](results/2026-11-23-hp-omnibook-x-n1x-128gb) | basalt-rtx-spark `25551ed` | sustained, GPU ~42 W while busy | 97.9 | 284 | 86.5 / 99.4 / 110.8 / 113.1 | 1,724 / 2,077 / 2,342 |
| [HP OmniBook X, N1X, 128 GB, 96 GB split](results/2026-11-23-hp-omnibook-x-n1x-128gb) | Basalt + Arm64 port `2ec7a84` | sustained, GPU ~41 W while busy | 87.5 | 622 | 74.6 / 81.1 / 85.7 / 85.3 | 983 / 1,506 / 2,095 |
| [HP OmniBook X, N1X, 128 GB, 96 GB split](results/2026-11-22-hp-omnibook-x-n1x-128gb-vllm) | basalt-rtx-spark `c01f9ed` | power floor (22-28 W) after ~8 min | 88.2 | 324 | 77.2 / 86.6 / 94.5 / 98.3 | 1,347 / 1,700 / 1,781 |
| [HP OmniBook X, N1X, 128 GB, 96 GB split](results/2026-11-22-hp-omnibook-x-n1x-128gb-vllm) | vLLM for RTX Spark 0.2.0, FP4-Local, MTP 7 | power floor (22-28 W) after ~8 min | 55.1 | 279 | 53.0 / 81.2 / 97.8 / 104.4 | 1,370 / 1,773 / 2,182 |

All with Qwen3.8-Flash-Next; the Basalt rows with UD-IQ3_XXS, int8 KV, four 262,144-token slots and MTP window 8.
Compare rows only within one power state.

### Single-stream decode by scenario

Against Basalt with the Arm64 port, back to back with the power budget held up:

![Decode by scenario, against Basalt with the Arm64 port](img/paired-decode.svg)

Against vLLM for RTX Spark, both on the power floor:

![Decode by scenario, against vLLM for RTX Spark](img/vllm-decode.svg)

The per-scenario tables, concurrency and prefill charts are in each result's folder. `plot.py` makes the same charts
from any BetterBench result files:

```
python benchmarks/plot.py --out benchmarks/img --prefix mine mine.json "My laptop" other.json "Other"
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
