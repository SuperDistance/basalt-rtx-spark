// SPDX-License-Identifier: MIT
#include "nvfp4_i8mm.hpp"

#include <arm_neon.h>
#include <cassert>
#include <cstring>

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
#include "ggml/ggml_nvfp4_neon.inc"
#undef GGML_CPU_FP16_TO_FP32
#undef UNUSED

namespace basalt::kernels::cpu {
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

}  // namespace

void nvfp4_i8mm_dot_2x2(int n, const void* weight0, const void* weight1,
                       const void* act0, const void* act1, float* output) {
    assert(n % QK_NVFP4 == 0);
    const auto* w0 = static_cast<const block_nvfp4*>(weight0);
    const auto* w1 = static_cast<const block_nvfp4*>(weight1);
    const auto* a0 = static_cast<const block_q8_0*>(act0);
    const auto* a1 = static_cast<const block_q8_0*>(act1);
    const int blocks = n / QK_NVFP4;
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

}  // namespace basalt::kernels::cpu
