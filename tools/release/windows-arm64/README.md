# Basalt RTX Spark for Windows Arm64

This build runs Qwen3.8-Flash-Next and Qwen3.8-27B locally on the NVIDIA RTX Spark N1X integrated GPU (CUDA compute capability 12.1, `sm_121`). The archive includes the Arm64 engine and CUDA 13.5 cuBLAS libraries.

1. Install a current NVIDIA driver, the Microsoft Visual C++ Arm64 runtime and Python 3.10 or later for Windows Arm64. The release was tested on a 128 GB N1X laptop. [Hardware and software requirements](docs/rtx-spark-setup.md#hardware)
2. Extract the entire ZIP. Run `setup.cmd` from the extracted folder. It creates a Python environment and offers to download the Flash-Next `UD-IQ3_XXS` pack. Allow about 120 GB of free space for the pack and the n-gram table created on first use.
3. Run `start.cmd` and wait for `weights loaded`. Open the local dashboard at [127.0.0.1:8080](http://127.0.0.1:8080/). For an OpenAI-compatible client, use `http://127.0.0.1:8080/v1` and the model name `flash-next`.

The supplied configuration enables four full-context slots of 262,144 tokens each, INT8 KV and a speculative window of eight. [Client guide](docs/rtx-spark-setup.md#use-it-from-your-tools) · [Measured performance and GPU comparisons](https://github.com/SuperDistance/basalt-rtx-spark#results)
