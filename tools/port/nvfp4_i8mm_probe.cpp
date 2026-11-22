// SPDX-License-Identifier: MIT
// Compare a two-token, two-row NVFP4 tile with Basalt's ARM64 SDOT kernel.
// Build for ARMv8.6-A and run on a CPU with FEAT_I8MM.
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

#define UNUSED(...) (void)(__VA_ARGS__)
#if defined(_MSC_VER) && !defined(__clang__)
static inline float fp16_to_float(ggml_half value) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(value))), 0);
}
#else
static inline float fp16_to_float(ggml_half value) {
    __fp16 half;
    std::memcpy(&half, &value, sizeof(half));
    return half;
}
#endif
#define GGML_CPU_FP16_TO_FP32 fp16_to_float
#include "../../src/kernels/cpu/ggml/ggml_nvfp4_neon.inc"
#undef GGML_CPU_FP16_TO_FP32
#undef UNUSED

namespace {

int8x16_t unpack_subblock(const uint8_t* codes, int8x16_t values) {
    const uint8x8_t packed = vld1_u8(codes);
    const uint8x16_t indices = vcombine_u8(vand_u8(packed, vdup_n_u8(15)), vshr_n_u8(packed, 4));
    return vqtbl1q_s8(values, indices);
}

int32x4_t dot_tile(int8x16_t weight0, int8x16_t weight1, int8x16_t act0, int8x16_t act1) {
    const int32x4_t zero = vdupq_n_s32(0);
    const int8x16_t w_lo = vcombine_s8(vget_low_s8(weight0), vget_low_s8(weight1));
    const int8x16_t w_hi = vcombine_s8(vget_high_s8(weight0), vget_high_s8(weight1));
    const int8x16_t a_lo = vcombine_s8(vget_low_s8(act0), vget_low_s8(act1));
    const int8x16_t a_hi = vcombine_s8(vget_high_s8(act0), vget_high_s8(act1));
    return vmmlaq_s32(vmmlaq_s32(zero, w_lo, a_lo), w_hi, a_hi);
}

void i8mm(int blocks, const block_nvfp4* w0, const block_nvfp4* w1,
          const block_q8_0* a0, const block_q8_0* a1, float* output) {
    const int8x16_t values = vld1q_s8(kvalues_fp4);
    float32x4_t accum[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (int b = 0; b < blocks; ++b) {
        for (int half = 0; half < 2; ++half) {
            const block_q8_0& x = a0[2 * b + half];
            const block_q8_0& y = a1[2 * b + half];
            const float xd = fp16_to_float(x.d);
            const float yd = fp16_to_float(y.d);
            for (int sub = 0; sub < 2; ++sub) {
                const int s = 2 * half + sub;
                const int32x4_t products = dot_tile(
                    unpack_subblock(w0[b].qs + 8 * s, values),
                    unpack_subblock(w1[b].qs + 8 * s, values),
                    vld1q_s8(x.qs + 16 * sub), vld1q_s8(y.qs + 16 * sub));
                const float sd0 = nvfp4_scale(w0[b].d[s]);
                const float sd1 = nvfp4_scale(w1[b].d[s]);
                const float scales[] = {sd0 * xd, sd0 * yd, sd1 * xd, sd1 * yd};
                accum[s] = vfmaq_f32(accum[s], vcvtq_f32_s32(products), vld1q_f32(scales));
            }
        }
    }
    // Match the four sub-block accumulators and the pairwise horizontal sum in the SDOT kernel.
    vst1q_f32(output, vaddq_f32(vaddq_f32(accum[0], accum[1]), vaddq_f32(accum[2], accum[3])));
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
