# <date>, <maker> <model>, <chip>, <RAM>

| | |
| --- | --- |
| Machine | maker and model, chip (N1X / GB10 / ...), RAM |
| GPU memory split | e.g. 96 GB (firmware setting), or the default |
| Driver, OS | NVIDIA driver version; Windows build or Linux distribution and kernel |
| Power | AC or battery; Windows power mode; the laptop's fan or thermal profile |
| Power budget | anything holding it up during the run, or "none" |
| Engine | name and commit; how it was built |
| Server config | the config file or flags (context, slots, KV type, speculative decoding) |
| Model pack | file name and source |
| Benchmark | BetterBench version and config (`benchmarks/betterbench-config.json` unless stated) |
| During the run | other programs using the GPU or much CPU, and when |

## Results

Paste the BetterBench summary: the weighted decode median, the short-prompt first token, the concurrency table and
the prefill sweep, and the GPU's mean power and clock while busy (from the GPU log).

## Notes

Anything unusual: thermal holds, a power-state change mid-run, errors or retries.
