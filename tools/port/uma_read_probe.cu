// GPU reads from three Windows-on-Arm allocation modes on the same RTX Spark.
#include <windows.h>
#include <psapi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1u << 20;
constexpr std::size_t kAlignment = 2 * kMiB;
constexpr std::size_t kTileBytes = 256u << 10;
constexpr unsigned kTileVectors = kTileBytes / sizeof(uint4);
constexpr unsigned kThreads = 256;
constexpr unsigned kMaxBlocks = 192;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

enum class Placement { Device, HostMapped, Registered64k, Registered2m };



struct Buffer {
    Placement placement;
    void* host = nullptr;
    void* reservation = nullptr;
    uint4* gpu = nullptr;
    bool registered = false;

    explicit Buffer(Placement placement) : placement(placement) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void allocate(std::size_t bytes) {
        switch (placement) {
        case Placement::Device:
            check(cudaMalloc(&gpu, bytes));
            break;
        case Placement::HostMapped:
            check(cudaHostAlloc(&host, bytes, cudaHostAllocMapped));
            map();
            break;
        case Placement::Registered64k: {
            constexpr std::size_t kGranularity = 64u << 10;
            reservation = VirtualAlloc(nullptr, bytes + kGranularity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!reservation) throw std::runtime_error("64 KiB aligned VirtualAlloc failed");
            host = reinterpret_cast<std::uintptr_t>(reservation) % kAlignment == 0
                ? static_cast<char*>(reservation) + kGranularity : reservation;
            register_host(bytes);
            break;
        }
        case Placement::Registered2m: {
            auto allocate = reinterpret_cast<decltype(&VirtualAlloc2)>(
                GetProcAddress(GetModuleHandleW(L"KernelBase.dll"), "VirtualAlloc2"));
            if (!allocate) throw std::runtime_error("VirtualAlloc2 is unavailable");
            MEM_ADDRESS_REQUIREMENTS range{};
            range.Alignment = kAlignment;
            MEM_EXTENDED_PARAMETER parameter{};
            parameter.Type = MemExtendedParameterAddressRequirements;
            parameter.Pointer = &range;
            reservation = allocate(GetCurrentProcess(), nullptr, bytes, MEM_RESERVE | MEM_COMMIT,
                                   PAGE_READWRITE, &parameter, 1);
            if (!reservation || reinterpret_cast<std::uintptr_t>(reservation) % kAlignment != 0) {
                throw std::runtime_error("2 MiB aligned VirtualAlloc2 failed");
            }
            host = reservation;
            register_host(bytes);
            break;
        }
        }
    }

    ~Buffer() {
        switch (placement) {
        case Placement::Device:
            if (gpu) cudaFree(gpu);
            break;
        case Placement::HostMapped:
            if (host) cudaFreeHost(host);
            break;
        case Placement::Registered64k:
        case Placement::Registered2m:
            if (registered) cudaHostUnregister(host);
            if (reservation) VirtualFree(reservation, 0, MEM_RELEASE);
            break;
        }
    }

private:
    void register_host(std::size_t bytes) {
        check(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
        registered = true;
        map();
    }

    void map() {
        void* device_pointer = nullptr;
        check(cudaHostGetDevicePointer(&device_pointer, host, 0));
        gpu = static_cast<uint4*>(device_pointer);
    }
};

struct FileView {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    void* host = nullptr;
    const uint4* gpu = nullptr;
    bool registered = false;

    FileView() = default;
    FileView(const FileView&) = delete;
    FileView& operator=(const FileView&) = delete;

    void open(const wchar_t* path, std::uint64_t offset, std::size_t bytes) {
        file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER size{};
        if (file == INVALID_HANDLE_VALUE || !GetFileSizeEx(file, &size) || size.QuadPart < 0 ||
            offset > static_cast<std::uint64_t>(size.QuadPart) ||
            bytes > static_cast<std::uint64_t>(size.QuadPart) - offset) {
            throw std::runtime_error("Cannot open a model slice of the requested size");
        }
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping) throw std::runtime_error("Cannot create a read-only file mapping");
        host = MapViewOfFile(mapping, FILE_MAP_READ, static_cast<DWORD>(offset >> 32),
                             static_cast<DWORD>(offset), bytes);
        if (!host) throw std::runtime_error("Cannot map the model slice");
        check(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        registered = true;
        void* alias = nullptr;
        check(cudaHostGetDevicePointer(&alias, host, 0));
        gpu = static_cast<const uint4*>(alias);
    }

    ~FileView() {
        if (registered) cudaHostUnregister(host);
        if (host) UnmapViewOfFile(host);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

struct Input {
    const char* name;
    const uint4* device;
};

struct Events {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;

    ~Events() {
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
    }
};

__global__ void read_tiles(const uint4* __restrict__ input, const unsigned* __restrict__ tile_order,
                           unsigned tile_count, unsigned long long* __restrict__ totals) {
    __shared__ unsigned long long warp_sums[kThreads / 32];
    unsigned long long sum = 0;
    for (unsigned tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
        const uint4* row = input + static_cast<std::size_t>(tile_order[tile]) * kTileVectors;
        for (unsigned col = threadIdx.x; col < kTileVectors; col += blockDim.x) {
            const uint4 value = row[col];
            sum += static_cast<unsigned long long>(value.x) + value.y + value.z + value.w;
        }
    }

    const unsigned lane = threadIdx.x % 32;
    for (unsigned distance = 16; distance; distance /= 2) sum += __shfl_down_sync(0xffffffffu, sum, distance);
    if (lane == 0) warp_sums[threadIdx.x / 32] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        sum = lane < kThreads / 32 ? warp_sums[lane] : 0;
        for (unsigned distance = 16; distance; distance /= 2) sum += __shfl_down_sync(0xffffffffu, sum, distance);
        if (lane == 0) totals[blockIdx.x] = sum;
    }
}

float timed_reads(const uint4* input, const unsigned* order, unsigned tiles, unsigned long long* totals,
                  const Events& events, unsigned repeats, unsigned long long expected) {
    const unsigned blocks = std::min(kMaxBlocks, tiles);
    check(cudaEventRecord(events.start));
    for (unsigned i = 0; i < repeats; ++i) read_tiles<<<blocks, kThreads>>>(input, order, tiles, totals);
    check(cudaEventRecord(events.end));
    check(cudaEventSynchronize(events.end));
    check(cudaGetLastError());
    float ms = 0;
    check(cudaEventElapsedTime(&ms, events.start, events.end));
    std::array<unsigned long long, kMaxBlocks> host_totals{};
    check(cudaMemcpy(host_totals.data(), totals, blocks * sizeof(host_totals[0]), cudaMemcpyDeviceToHost));
    const auto actual = std::accumulate(host_totals.begin(), host_totals.end(), 0ull);
    if (actual != expected) throw std::runtime_error("GPU read checksum mismatch");
    return ms;
}

unsigned char tile_value(unsigned tile) {
    return static_cast<unsigned char>(1 + tile % 251);
}

struct Test {
    const char* name;
    unsigned tiles;
    unsigned repeats;
    bool shuffle;
};

void run(const wchar_t* model, std::uint64_t model_offset) {
    constexpr unsigned kAllocationBytes = 512 * kMiB;
    constexpr unsigned kAllTiles = kAllocationBytes / kTileBytes;
    const std::array<Test, 3> cases{{
        {"512m_sequential", kAllTiles, 12, false},
        {"128m_shuffled", kAllTiles / 4, 24, true},
        {"2m_repeated", 2 * kMiB / kTileBytes, 120, false},
    }};

    static_assert(sizeof(unsigned) == 4);
    check(cudaSetDevice(0));
    cudaDeviceProp gpu{};
    check(cudaGetDeviceProperties(&gpu, 0));
    if (!gpu.integrated || gpu.major != 12 || gpu.minor != 1) {
        throw std::runtime_error("Expected an integrated SM121 device");
    }

    PERFORMANCE_INFORMATION windows{};
    windows.cb = sizeof(windows);
    if (!GetPerformanceInfo(&windows, sizeof(windows))) throw std::runtime_error("Cannot query Windows commit");
    std::size_t raw_free = 0, raw_total = 0;
    check(cudaMemGetInfo(&raw_free, &raw_total));
    std::printf("# CUDA reports %.2f GiB free of %.2f GiB, Windows commit %.2f GiB of %.2f GiB\n",
                raw_free / (1024.0 * kMiB), raw_total / (1024.0 * kMiB),
                static_cast<double>(windows.CommitTotal) * windows.PageSize / (1024.0 * kMiB),
                static_cast<double>(windows.CommitLimit) * windows.PageSize / (1024.0 * kMiB));

    Buffer device(Placement::Device), mapped(Placement::HostMapped);
    Buffer registered64k(Placement::Registered64k), registered2m(Placement::Registered2m);
    for (auto* allocation : {&device, &mapped, &registered64k, &registered2m}) {
        allocation->allocate(kAllocationBytes);
    }
    std::vector<Input> inputs{
        {"device", device.gpu}, {"host_mapped", mapped.gpu},
        {"host_registered_64k", registered64k.gpu}, {"host_registered_2m", registered2m.gpu},
    };

    FileView file;
    if (model) {
        file.open(model, model_offset, kAllocationBytes);
        std::memcpy(mapped.host, file.host, kAllocationBytes);
        inputs.push_back({"file_registered", file.gpu});
    } else {
        auto* bytes = static_cast<unsigned char*>(mapped.host);
        for (unsigned tile = 0; tile < kAllTiles; ++tile) {
            std::memset(bytes + static_cast<std::size_t>(tile) * kTileBytes, tile_value(tile), kTileBytes);
        }
    }
    std::memcpy(registered64k.host, mapped.host, kAllocationBytes);
    std::memcpy(registered2m.host, mapped.host, kAllocationBytes);
    check(cudaMemcpy(device.gpu, mapped.host, kAllocationBytes, cudaMemcpyHostToDevice));

    std::vector<unsigned long long> tile_sums(kAllTiles);
    const auto* words = static_cast<const unsigned*>(mapped.host);
    constexpr std::size_t kTileWords = kTileBytes / sizeof(unsigned);
    for (unsigned tile = 0; tile < kAllTiles; ++tile) {
        const auto* begin = words + tile * kTileWords;
        tile_sums[tile] = std::accumulate(begin, begin + kTileWords, 0ull);
    }

    Buffer work(Placement::Device);
    work.allocate(kAlignment);
    auto* gpu_order = reinterpret_cast<unsigned*>(work.gpu);
    auto* totals = reinterpret_cast<unsigned long long*>(gpu_order + kAllTiles);
    Events events;
    check(cudaEventCreate(&events.start));
    check(cudaEventCreate(&events.end));
    std::mt19937 rng(121);
    std::vector<unsigned> order(kAllTiles);
    std::puts("pattern,placement,median_gb_s,min_gb_s,max_gb_s");

    for (const Test& test : cases) {
        std::iota(order.begin(), order.end(), 0);
        if (test.shuffle) std::shuffle(order.begin(), order.end(), rng);
        check(cudaMemcpy(gpu_order, order.data(), test.tiles * sizeof(order[0]), cudaMemcpyHostToDevice));
        unsigned long long expected = 0;
        for (unsigned index = 0; index < test.tiles; ++index) {
            expected += tile_sums[order[index]];
        }
        for (const auto& input : inputs) timed_reads(input.device, gpu_order, test.tiles, totals, events, 3, expected);
        std::vector<std::array<double, 10>> samples(inputs.size());
        for (std::size_t round = 0; round < samples[0].size(); ++round) {
            for (std::size_t step = 0; step < inputs.size(); ++step) {
                const std::size_t mode = (round + step) % inputs.size();
                const float ms = timed_reads(inputs[mode].device, gpu_order, test.tiles, totals, events, test.repeats, expected);
                samples[mode][round] = static_cast<double>(test.tiles) * kTileBytes * test.repeats / (ms * 1e6);
            }
        }
        for (std::size_t mode = 0; mode < inputs.size(); ++mode) {
            auto& rates = samples[mode];
            std::sort(rates.begin(), rates.end());
            std::printf("%s,%s,%.2f,%.2f,%.2f\n", test.name, inputs[mode].name,
                        (rates[rates.size() / 2 - 1] + rates[rates.size() / 2]) / 2, rates.front(), rates.back());
        }
    }
}
}  // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc == 1) {
            run(nullptr, 0);
        } else if (argc == 3) {
            wchar_t* tail = nullptr;
            const auto offset = std::wcstoull(argv[2], &tail, 10);
            if (argv[2][0] == L'-' || tail == argv[2] || *tail || offset % kAlignment) {
                throw std::runtime_error("The model offset must be a nonnegative multiple of 2 MiB");
            }
            run(argv[1], offset);
        } else {
            throw std::runtime_error("Usage: uma-read-probe [MODEL_FILE OFFSET_BYTES]");
        }
    } catch (const std::exception& err) {
        std::fprintf(stderr, "UMA probe: %s\n", err.what());
        return 1;
    }
}
