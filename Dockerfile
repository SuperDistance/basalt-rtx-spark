# syntax=docker/dockerfile:1
#
# Container image for Basalt: the engine, the Python server and its venv on an
# NVIDIA CUDA 13 runtime.
#
# Build from the repository root:
#   docker build -t basalt .                                   # for the build machine's own architecture
#   docker buildx build --platform linux/arm64 -t basalt .     # DGX Spark / RTX Spark, from another machine
#   docker compose up -d                                       # build and serve with compose.yaml's flags
#
# The image's architecture picks the GPU: x86-64 builds for compute capability 12.0 (RTX 50-series, sm_120),
# Arm64 for 12.1 (DGX Spark's GB10 and the RTX Spark laptops' N1X, sm_121); --build-arg CUDA_ARCHITECTURES=...
# overrides it. The host needs a driver of the R580 branch or newer and the NVIDIA Container Toolkit (Docker
# Desktop on Windows brings its own). Building needs no GPU: the engine is compiled with no device present, and
# nothing runs CUDA during the build. A heavy CUDA file takes 2-4 GB to compile: on a small VM, cap the jobs with
# --build-arg BUILD_JOBS=2. docs/docker.md has the run flags, the mounts and the unified-memory notes.
#
# The tree inside the image is a release archive's, with the venv already
# installed: bin/, serve/, tools/, data/ and .venv/ under one top directory. A
# binary finds its install root as the directory above itself, and reads the
# expert profiles from <install root>/data/profiles, execs the python of
# <install root>/.venv for `basalt serve` and looks for the basalt-vision
# sidecar beside itself. The top directory is the same fixed path in both stages
# because the venv's own scripts name it absolutely.
#
# Mounts:
#   /data    basalt_config.json (v2; tools/migrations/config_v1_to_v2.py converts
#            an older file) and, since HOME is pinned there, the server's folder
#            per model under ~/.cache/basalt: its engine log and, when the
#            conversation disk tier is on, that tier
#   /models  the .basalt model files

ARG CUDA_TAG=13.4.3

# ------------------------------------------------------------------ build stage
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS builder

# empty: the architecture's default (120 on x86-64, 121 on Arm64); BUILD_JOBS empty: one job per core
ARG CUDA_ARCHITECTURES=
ARG BUILD_JOBS=

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt
COPY . /opt/basalt/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        ${CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}} \
    && cmake --build build -j"${BUILD_JOBS:-$(nproc)}" \
    && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------- final stage
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04

# The base already carries the CUDA runtime and cuBLAS, the only CUDA libraries
# the engine links. python3 is what the venv's python points at.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# The runtime tree: the binaries, the server and the tools it imports, the data/
# the engine reads, the venv, and the licences.
COPY --from=builder /opt/basalt/.venv            /opt/basalt/.venv
COPY --from=builder /opt/basalt/bin              /opt/basalt/bin
COPY --from=builder /opt/basalt/serve            /opt/basalt/serve
COPY --from=builder /opt/basalt/tools            /opt/basalt/tools
COPY --from=builder /opt/basalt/data             /opt/basalt/data
COPY --from=builder /opt/basalt/requirements.txt \
                  /opt/basalt/requirements.txt
COPY --from=builder /opt/basalt/LICENSE \
                  /opt/basalt/THIRD_PARTY_LICENSES \
                  /opt/basalt/

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

# Serve the config from the /data mount, bound to 0.0.0.0 so a mapped port
# reaches the server (loopback is the built-in default). Arguments given after
# the image name follow the entrypoint, e.g. --api-key KEY. The served port is
# server.port (8080) and the host mapping has to match it.
EXPOSE 8080

# Healthy once the model is loaded (/v1/status says "loaded"); a first start reads the whole pack and may write the
# n-gram table's copy beside it, so it has a long start period. A port other than 8080 needs its own healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["/opt/basalt/.venv/bin/python", "-c", "import json, urllib.request, sys; s = json.load(urllib.request.urlopen('http://127.0.0.1:8080/v1/status', timeout=4)); sys.exit(0 if s.get('loaded') else 1)"]

ENTRYPOINT ["basalt", "serve", "--config", "/data/basalt_config.json", "--host", "0.0.0.0"]
