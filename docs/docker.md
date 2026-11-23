# Docker

A container runs the same engine and server on a Linux host: a DGX Spark (GB10), an RTX Spark laptop under Linux, or an
RTX Spark laptop through Docker Desktop on Windows. The image's architecture picks the GPU: Arm64 builds for compute
capability 12.1 (GB10, N1X), x86-64 for 12.0 (RTX 50-series). On Windows on Arm the native build
([README](../README.md#install)) is the faster path; the container is for Linux hosts and for keeping the engine
apart from the rest of the machine.

## Host

- An NVIDIA driver of the R580 branch or newer.
- Docker with the NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker` if containers do not see
  the GPU). DGX OS ships both. Docker Desktop on Windows reaches the GPU through its WSL 2 backend: on an RTX Spark
  N1X laptop, `docker run --rm --gpus all nvidia/cuda:13.4.3-base-ubuntu24.04 nvidia-smi` lists the N1X.
- Disk for the pack (88 GB for UD-IQ3_XXS) and its n-gram table's copy (29 GB, written beside it on the first start).

## Run

From a checkout of this repository:

```sh
mkdir -p docker-data models
cp docker/basalt_config.json docker-data/
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-IQ3_XXS.basalt --local-dir models
docker compose up -d          # builds the image the first time (see Build)
docker compose logs -f        # "weights loaded" when it is ready; `docker ps` shows (healthy)
```

Then `http://127.0.0.1:8080/` (web UI), `http://127.0.0.1:8080/v1` (OpenAI API, model `flash-next`), or
`http://127.0.0.1:8080` as an Anthropic base URL.

Without compose, the same flags:

```sh
docker run -d --name basalt --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8080:8080 -v "$PWD/docker-data:/data" -v "$PWD/models:/models" basalt-rtx-spark:local
```

- `--ipc host` and an unlimited `memlock`: the engine page-locks its RAM expert tier and conversation checkpoints, as
  the DGX Spark recipes run their engines.
- `/data`: `basalt_config.json` and, since `HOME` points there, the server's folder per model under
  `~/.cache/basalt` (engine log, the conversation disk tier when it is on).
- `/models`: the packs, writable, at the paths the config names (`/models/<pack>.basalt`).
- Arguments after the image name go to the server, e.g. `--api-key KEY`. The port is `server.port` in the config
  (8080), and the healthcheck assumes it.

## Build

`docker compose up` builds `basalt-rtx-spark:local` from the Dockerfile; `docker build -t basalt-rtx-spark:local .`
does the same. Build on the target machine: building Arm64 on an x86-64 host (`docker buildx build --platform
linux/arm64`) runs the compiler under emulation and takes hours. A heavy CUDA file needs 2-4 GB to compile, so on a
host or VM with little memory cap the jobs: `--build-arg BUILD_JOBS=2` (in compose.yaml, `build.args.BUILD_JOBS`).
Docker Desktop's default VM (4 CPUs, 4 GB) is too small to build or to serve the model; give it more in its Resources
settings first.

## Unified memory

On a DGX Spark and an RTX Spark the GPU and the CPU share one memory.

- A container memory limit (`--memory`) does not bound the GPU's allocations: the model's weights and KV are GPU
  memory whatever the container's cgroup says. Leave the host enough free memory instead.
- The engine places the experts itself across the GPU's share and pinned RAM; the startup lines say where they went.
- On Linux the page cache can hold much of a pack after the load. If the host runs short while serving, the DGX Spark
  playbooks flush it: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
- With Docker Desktop on Windows, a model on the Windows disk reaches the container through file sharing, which is slow
  for a 52 GB load and for the n-gram table's reads. Keep the packs on the Linux side (a named volume or the WSL file
  system) for serving.

## Status

Checked on an RTX Spark N1X laptop (Windows 11 on Arm, Docker Desktop 29.9, driver 621.24): the GPU reaches an Arm64
container (CUDA 13.5), `docker build --check` passes with both CUDA base images resolved for Arm64, `docker compose
config` accepts compose.yaml, and the healthcheck's command reports a loaded server. Not yet run end to end: building
this image and serving from it, on a DGX Spark or a Linux RTX Spark host. Results from those hosts are welcome in
[benchmarks](../benchmarks).
