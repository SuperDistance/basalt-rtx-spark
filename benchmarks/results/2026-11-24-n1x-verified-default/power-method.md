# Power measurement method

The [Basalt BetterBench results](basalt/summary.json) are divided into decode, prefill and concurrency phases using the timestamps in the benchmark log. GPU samples come from `nvidia-smi` at roughly one sample per second. A sample is busy at GPU utilization of at least 80%. For each phase the package reports mean GPU draw and median SM clock over busy samples, plus mean GPU draw over the full phase. The underlying GPU samples are in `basalt/gpu.csv`.

For the headline decode result, [common-power.json](common-power.json) uses the SparkHelper samples between 10:44:06 and 10:57:52 Pacific on November 24. Of 411 distinct samples, 409 met the 80% GPU utilization threshold. Their mean CPU and GPU draws were 9.45 W and 35.83 W, or 45.28 W combined. The NPU averaged another 1.01 W. These component readings exclude the display and other system loads.

The paired four-request controls use the same 80% busy threshold. `controls/summary.json` reports the number of NVIDIA and SparkHelper busy samples, average GPU draw and median busy SM clock, and the measured SoC rail aggregate. It also retains the separately reported GPU, CPU and NPU rail draws. Each arm's raw GPU and rail samples are under its `A1`, `B1`, `B2` or `A2` directory.

The GPU power limit was unavailable from the driver. These figures are observed power, not a TDP rating. The A/B/B/A control sequence also showed a falling clock trend across the session, so throughput differences should be read with each arm's actual power and clock measurements rather than treated as an isolated engine effect.

For the Basalt/vLLM C4 comparison, [common-power.json](common-power.json) aligns full SparkHelper telemetry windows and applies the same 80% filter. It reports 39.33 W mean busy GPU draw and 44.19 W mean measured CPU+GPU+NPU rails for Basalt, versus 34.46 W and 41.90 W for vLLM. Mean SM clocks were 2,034 and 2,060 MHz. The common windows contain 83/84 and 94/95 busy/total samples, respectively.

vLLM's NVIDIA CSV missed roughly 42 seconds while the process was stopping. Its C4 interval has 148 samples across about 191 seconds, 146 at or above 80% GPU utilization. Those NVIDIA power values are retained in `vllm/nvidia-power-by-phase.json` as a partial sensor record and are excluded from the direct cross-engine power comparison.
