// include/basalt/models/qwen4exp/kernels/draft_draw.hpp - a drawn MTP draft's parameters (speculative sampling).
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// The request's draw, in mapped host memory so a captured graph reads the current window's values: `on` 0 keeps the
/// argmax pick.  The chain's cut (top_k, top_p) and temperature shape the proposal; seed keys Philox's draft stream.
struct DraftDraw {
    int on = 0;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    unsigned long long seed = 0;
};

/// Where one draft step's draw reads and writes: the draw, the step's cell (the Philox counter), and its proposal
/// (kDraftDrawQ entries of token ids and probabilities).  All null: the argmax pick.
constexpr int kDraftDrawQ = 32;
/// The most activation rows one pass of a two-stage draft head takes (q5k_proxy_top1's n_x).
constexpr int kProxyMaxRows = 8;
struct DraftDrawArgs {
    const DraftDraw* draw = nullptr;
    const int32_t* cell = nullptr;
    int32_t* q_ids = nullptr;
    float* q_p = nullptr;
};

}  // namespace basalt::kernels
