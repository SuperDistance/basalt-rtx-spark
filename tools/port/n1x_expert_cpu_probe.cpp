// SPDX-License-Identifier: MIT
// Single-worker Basalt reference for n1x_expert_dla_probe.py.
#include "basalt/core/diag.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace basalt::kernels::cpu;

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: n1x_expert_cpu_probe.exe weights.bin input.f32 tokens output.f32 workers\n");
        return 1;
    }
    const int tokens = std::stoi(argv[3]);
    const int workers = std::stoi(argv[5]);
    if (tokens <= 0 || tokens > 64 || workers < 1) return 1;

    constexpr int embedding = 2560;
    constexpr int hidden = 640;
    NativeFmt fmt;
    std::string error;
    if (!native_fmt(GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, embedding, hidden, fmt, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }

    std::vector<uint8_t> blob(fmt.bytes);
    std::vector<float> input(tokens * embedding), output(tokens * embedding), intermediate(tokens * hidden);
    std::ifstream weights(argv[1], std::ios::binary), activations(argv[2], std::ios::binary);
    if (!weights.read(reinterpret_cast<char*>(blob.data()), blob.size()) ||
        !activations.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float))) {
        std::fprintf(stderr, "Could not read weights and activations\n");
        return 1;
    }

    std::vector<uint8_t> packed_input(tokens * fmt.act_bytes), packed_hidden(tokens * fmt.h_bytes);
    std::vector<const void*> act(tokens), hq(tokens);
    std::vector<float*> ff(tokens), out(tokens);
    for (int i = 0; i < tokens; ++i) {
        act[i] = packed_input.data() + i * fmt.act_bytes;
        hq[i] = packed_hidden.data() + i * fmt.h_bytes;
        ff[i] = intermediate.data() + i * hidden;
        out[i] = output.data() + i * embedding;
    }

    basalt::core::diagnostics().exp["n1x_i8mm"] = 1;
    auto quantize_input = [&] {
        for (int i = 0; i < tokens; ++i)
            native_quant_act(fmt, input.data() + i * embedding, packed_input.data() + i * fmt.act_bytes);
    };
    auto run_single = [&] {
        quantize_input();
        native_gu_rows(fmt, blob.data(), act.data(), tokens, ff.data(), 0, hidden);
        for (int i = 0; i < tokens; ++i)
            native_quant_h(fmt, ff[i], packed_hidden.data() + i * fmt.h_bytes);
        native_down_rows(fmt, blob.data(), hq.data(), tokens, out.data(), 0, embedding);
    };

    auto measure = [](auto run) {
        for (int i = 0; i < 8; ++i) run();
        std::array<double, 64> times;
        for (double& ms : times) {
            const auto start = std::chrono::steady_clock::now();
            run();
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        std::sort(times.begin(), times.end());
        std::printf("{\"median_ms\":%.6f,\"p95_ms\":%.6f,\"samples\":%zu}",
                    (times[31] + times[32]) / 2, times[60], times.size());
    };

    std::printf("{\"single_worker\":");
    measure(run_single);
    const std::vector<float> single_output = output;
    std::vector<ExpertJobMulti> jobs((tokens + MAXT - 1) / MAXT);
    for (size_t chunk = 0; chunk < jobs.size(); ++chunk) {
        const int first = static_cast<int>(chunk) * MAXT;
        ExpertJobMulti& job = jobs[chunk];
        job.blob = blob.data();
        job.nt = std::min(MAXT, tokens - first);
        for (int i = 0; i < job.nt; ++i) {
            job.nact[i] = act[first + i];
            job.out[i] = out[first + i];
        }
    }
    std::printf(",\"pool_workers\":%d,\"pool_jobs\":%zu,\"pool\":", workers, jobs.size());
    ExpertPool pool(workers, PoolAffinity::Auto);
    measure([&] {
        quantize_input();
        pool.run_split_multi_native(fmt, jobs.data(), static_cast<int>(jobs.size()));
    });
    const bool same = std::memcmp(output.data(), single_output.data(), output.size() * sizeof(float)) == 0;
    std::printf(",\"pool_bitwise_equal\":%s}\n", same ? "true" : "false");
    std::ofstream result(argv[4], std::ios::binary);
    if (!result.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float))) return 1;
}
