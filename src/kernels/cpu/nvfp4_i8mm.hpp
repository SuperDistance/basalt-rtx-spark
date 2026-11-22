// SPDX-License-Identifier: MIT
#pragma once

namespace basalt::kernels::cpu {

// Two NVFP4 weight rows and two Q8_0 activation rows. Output order is [w0a0, w0a1, w1a0, w1a1].
// n is a multiple of 64 and the caller checks FEAT_I8MM before dispatch.
void nvfp4_i8mm_dot_2x2(int n, const void* weight0, const void* weight1,
                       const void* act0, const void* act1, float* output);

}  // namespace basalt::kernels::cpu
