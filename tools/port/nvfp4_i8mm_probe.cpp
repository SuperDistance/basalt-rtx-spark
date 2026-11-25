// SPDX-License-Identifier: MIT
// Compare a two-token, two-row NVFP4 tile with Basalt's ARM64 SDOT kernel.
// Build for ARMv8.6-A and run on a CPU with FEAT_I8MM.
#include "../../src/kernels/cpu/nvfp4_i8mm.hpp"

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/ggml/ggml.h"
#include "basalt/platform/compiler.hpp"

#define UNUSED(...) (void)(__VA_ARGS__)
static inline float fp16_to_float(ggml_half value) { return basalt::platform::f16_bits_to_f32(value); }
#define GGML_CPU_FP16_TO_FP32 fp16_to_float
#include "../../src/kernels/cpu/ggml/ggml_nvfp4_neon.inc"
#undef GGML_CPU_FP16_TO_FP32
#undef UNUSED

namespace {

void i8mm(int blocks, const block_nvfp4* w0, const block_nvfp4* w1,
          const block_q8_0* a0, const block_q8_0* a1, float* output) {
    basalt::kernels::cpu::nvfp4_i8mm_dot_2x2(blocks * QK_NVFP4, w0, w1, a0, a1, output);
}

void sdot(int blocks, const block_nvfp4* w0, const block_nvfp4* w1,
          const block_q8_0* a0, const block_q8_0* a1, float* output) {
    ggml_vec_dot_nvfp4_q8_0(blocks * QK_NVFP4, output, 0, w0, 0, a0, 0, 1);
    ggml_vec_dot_nvfp4_q8_0(blocks * QK_NVFP4, output + 1, 0, w0, 0, a1, 0, 1);
    ggml_vec_dot_nvfp4_q8_0(blocks * QK_NVFP4, output + 2, 0, w1, 0, a0, 0, 1);
    ggml_vec_dot_nvfp4_q8_0(blocks * QK_NVFP4, output + 3, 0, w1, 0, a1, 0, 1);
}

bool compare(int width, int rows, bool check_only, std::mt19937& random) {
    const int blocks = width / QK_NVFP4;
    std::vector<block_nvfp4> weights(rows * blocks);
    std::vector<block_q8_0> activations(4 * blocks);
    for (auto& block : weights) {
        for (auto& scale : block.d) scale = static_cast<uint8_t>(random());
        for (auto& code : block.qs) code = static_cast<uint8_t>(random());
    }
    for (auto& block : activations) {
        block.d = static_cast<ggml_half>(0x2400 + random() % 0x1c00);
        for (auto& value : block.qs) value = static_cast<int8_t>(static_cast<int>(random() % 256) - 128);
    }
    const auto* a0 = activations.data();
    const auto* a1 = a0 + 2 * blocks;
    std::array<float, 4> expected{}, actual{};
    for (int r = 0; r < rows; r += 2) {
        const auto* w0 = weights.data() + r * blocks;
        const auto* w1 = w0 + blocks;
        sdot(blocks, w0, w1, a0, a1, expected.data());
        i8mm(blocks, w0, w1, a0, a1, actual.data());
        if (std::memcmp(expected.data(), actual.data(), sizeof expected) != 0) {
            std::fprintf(stderr, "width %d, row %d mismatch\n", width, r);
            for (size_t i = 0; i < actual.size(); ++i)
                std::fprintf(stderr, "%zu: SDOT %.9g, I8MM %.9g\n", i, expected[i], actual[i]);
            return false;
        }
    }
    std::printf("width=%d rows=%d byte-for-byte parity\n", width, rows);
    if (check_only) return true;

    static constexpr int repeats = 28;
    auto measure = [&](auto dot) {
        float check = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int rep = 0; rep < repeats; ++rep) {
            for (int r = 0; r < rows; r += 2) {
                const auto* w0 = weights.data() + r * blocks;
                dot(blocks, w0, w0 + blocks, a0, a1, actual.data());
                check += actual[(r / 2 + rep) % actual.size()];
            }
        }
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
        return std::pair(elapsed.count(), check);
    };
    std::array<double, 7> sdot_ms{}, i8mm_ms{};
    for (size_t sample = 0; sample < sdot_ms.size(); ++sample) {
        const auto first = sample % 2 == 0 ? measure(sdot) : measure(i8mm);
        const auto second = sample % 2 == 0 ? measure(i8mm) : measure(sdot);
        if (first.second != second.second) return false;
        sdot_ms[sample] = sample % 2 == 0 ? first.first : second.first;
        i8mm_ms[sample] = sample % 2 == 0 ? second.first : first.first;
    }
    std::sort(sdot_ms.begin(), sdot_ms.end());
    std::sort(i8mm_ms.begin(), i8mm_ms.end());
    const double baseline = sdot_ms[sdot_ms.size() / 2];
    const double tile = i8mm_ms[i8mm_ms.size() / 2];
    std::printf("SDOT %.3f ms, I8MM %.3f ms, speedup %.3fx\n", baseline, tile, baseline / tile);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const bool check_only = argc == 2 && std::strcmp(argv[1], "--check") == 0;
    if (argc != 1 && !check_only) {
        std::fprintf(stderr, "usage: %s [--check]\n", argv[0]);
        return 2;
    }
    std::mt19937 random(121);
    for (const int width : {512, 2048, 3072}) {
        if (!compare(width, 1024, check_only, random)) return 1;
    }
    return 0;
}
