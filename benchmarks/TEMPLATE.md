# <date>, <maker> <model>, <chip>, <RAM>

| | |
| --- | --- |
| Machine | RTX Spark laptop maker and model, chip (e.g. N1X), installed unified memory |
| GPU memory split | e.g. 96 GB (firmware setting), or the default |
| Driver, OS | NVIDIA driver version and Windows Arm64 build (give OS and architecture for other platforms) |
| Power | AC or battery; Windows power mode; the laptop's fan or thermal profile |
| Power budget | anything holding it up during the run, or "none" |
| Engine | name and commit; how it was built |
| Server config | the config file or flags (context, slots, KV type, speculative decoding) |
| Model pack | file name and source |
| Benchmark | BetterBench version and config (`benchmarks/betterbench-config.json` unless stated) |
| During the run | other programs using the GPU or much CPU, and when |

## Results

Paste the BetterBench summary: the weighted decode median, the short-prompt first token, the concurrency table and
the prefill sweep, and the GPU's mean power and clock while busy (give the utilization threshold). Include cold-join TTFT, incumbent stalls, and relevant thermal or clock changes. Use the workloads in `benchmarks/feature-coverage.md`.

## Notes

Anything unusual: thermal holds, a power-state change mid-run, errors or retries.
