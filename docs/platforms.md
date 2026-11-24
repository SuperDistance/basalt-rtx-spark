# Platforms

Basalt RTX Spark targets Windows 11 on Arm64 and the RTX Spark N1X integrated GPU (`sm_121`, compute capability 12.1). The native Windows build uses Arm64 MSVC, NEON CPU kernels and CUDA 13.5. See the [Windows setup](rtx-spark-setup.md), [Arm64 build and test record](arm64.md) and [N1X benchmarks](../benchmarks).

## Inherited x64 platforms

Upstream Basalt targets discrete Blackwell GPUs (`sm_120`) on Linux and Windows x64. These are separate build targets. For multi-GPU desktops, an optional build can also compile the peer expert kernels for Ada (`sm_89`). [GPU selection](configuration/gpus.md#an-older-peer-card)

| | Linux | Windows |
| --- | --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) | Windows 10 1903+ x64 |
| Compiler, build | GCC 13+, CMake 3.24+ | MSVC 19.44 (VS 2022), CMake 3.24+, Ninja |
| CUDA, Python | 13, 3.10+ | 13, 3.10+ |

## Release archives

For the N1X, use the Windows Arm64 ZIP and follow the [RTX Spark setup guide](rtx-spark-setup.md#install).

Upstream x64 releases carry `basalt-vX.Y.Z-linux-x64.tar.gz` and `basalt-vX.Y.Z-windows-x64.zip`: `bin/` (`basalt`, `basalt-bench`,
`basalt-probe`, `basalt-vision`), `serve/`, `tools/` (what the server imports), `data/`, `docs/`, `requirements.txt` and the licences.
They hold no CUDA libraries and no Python environment. Needs:

- An NVIDIA driver of the R580 branch or newer. The archives are built with CUDA 13.0.2, whose Linux minimum is 580.95.05; NVIDIA
  publishes no Windows minimum since 13.0, so use a current driver there.
- The CUDA 13 runtime libraries (cudart, cuBLAS) on the library path (`PATH` on Windows).
- Python 3.10+ with the requirements in a `.venv` at the archive's top directory, which `bin/basalt serve` uses when it is there:

```sh
cd basalt-vX.Y.Z-linux-x64                   # tar xzf the archive first; on Windows unzip it
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

On Windows: `py -3 -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`. Then follow the [upstream README](../README.basalt.md)'s quick start from the
model download on: `bin/basalt serve --config basalt_config.json`. `bin/basalt-bench` starts the `basalt` beside it and
reads the archive's `data/bench`.

## Container image

For Linux Arm64 or Docker Desktop with WSL2, see the [container instructions and validation status](docker.md). The headline N1X measurements use the native Windows Arm64 executable. [Benchmark records](../benchmarks)

## Platform code

OS code lives only in `src/platform/` and `include/basalt/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.

## FFmpeg for video

Video input loads FFmpeg's shared libraries at run time (libavformat, libavcodec, libavutil and libswscale of release 6, 7, 8 or 9; nothing
is linked or shipped). Linux: the distribution's runtime packages (`libavformat62` and its siblings on Ubuntu 26.04, `libavformat60`
on 24.04), found by the system loader. Windows: a shared FFmpeg build (`avformat-62.dll` for 8, `avformat-63.dll` for 9, and the rest, from gyan.dev or BtbN) beside
`basalt.exe` or in a `PATH` directory; the current directory is never searched. Without them video is off and the server says why.
GPU decoding needs an FFmpeg build with NVDEC (the distribution packages and the gyan.dev and BtbN builds have it); without it
frames decode on the CPU.

## Windows limitations

- Portable host registrations stay within the visible adapters' DXGI non-local allowances, leaving driver headroom.
  Whole-expert slices retain their own device aliases; the unregistered tail is locked in the working set.
  Qwen and DeepSeek's resident RAM tiers share CUDA host VMM when supported, with ordinary registration as the fallback.
  Allocation-backed pins stay in RAM across SLEEP/WAKE; the expert caches can still return their VRAM.
- Large pages need "Lock pages in memory" (Administrators). Without it the engine uses 4 KiB pages and says so at start.
- No shared arena files and no cgroup limits.
- NTFS reads a mapped file's unbuffered reads one at a time, and its cached random reads cost a CPU core each, so `--ple-io direct`
  reads the n-gram table from its own copy beside the model (the model file's name plus `.ple`). The copy is written once at start when the volume
  keeps 16 GiB free after it (about 30 s for a 29 GB table), and compared against the model on every start. Without room the
  table is read through the file cache, with a warning.
- `basalt serve` runs the Python server as a child.
- A display-attached GPU can hit the driver's timeout (TDR). Raise `TdrDelay` under `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`.
- A card that drives the desktop has VRAM that moves between starts, so greedy output can differ run to run. Pin `expert-cache` and raise
  `vram-reserve-mib` for reproducible output.
- For a Windows issue, include the CPU architecture, GPU model, driver, Windows build and model pack. N1X performance measurements include the GPU memory split and power conditions. [Benchmark records](../benchmarks)

## Building on Windows

Use the [native Arm64 build instructions](arm64.md#windows-on-arm64) on the N1X. For the inherited x64 target, run these commands in an x64 Developer Command Prompt.

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L cpu -j 8 --output-on-failure
```

The executables land in `bin/` whatever the generator is (a Visual Studio generator would otherwise put them in `bin/Release`, away from `data/`).

`-DBASALT_MODELS=qwen4exp`, `deepseek41` or `qwen35` builds one model ([building](building.md#building-only-some-models)); the option works the same here.

If tests fail to create fixtures, point `TEMP` and `TMP` into the build tree. CUDA 13.2+ headers need `/Zc:preprocessor`; the CMakeLists
passes it to nvcc's host compiler.

## Historical x64 compiler comparison

The CPU expert kernel is compiled by MSVC on Windows and GCC on Linux. On one commit, `expert_parity` and `pool_test` give 0 failures
on both, and the pool output matches the serial run bit for bit.

Greedy tokens differ across the builds: with tiers pinned and adaptive swaps off, `struct` matches over 300 tokens, `conv` diverges at
token 23 and `prose` at token 44. One build matches itself run to run, so the difference is the host compiler's CPU code (78% of expert
rows). That historical comparison did not include a Windows x64 quality run ([quality](precision/quality.md)).
