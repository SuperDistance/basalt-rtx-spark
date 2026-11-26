#pragma once

#include <algorithm>
#include <cmath>

namespace basalt::models::qwen4exp::serve {

// The C4 zero-doorbell graph can spend either one or three extra rows per slot. Compare the
// tokens the two budgets actually commit, including the draft work paid before each window.
class BatchRowCost {
public:
    static constexpr int kSlots = 4;
    static constexpr int kNarrow = 8;
    static constexpr int kWide = 16;

    struct Phase {
        int windows{};
        double tokens{}, verify_ms{}, draft_ms{}, other_ms{};
        double ms() const { return verify_ms + draft_ms + other_ms; }
    };

    void reset() { *this = BatchRowCost{}; }

    void observe(int budget, int rows, int kept, double verify_ms, double draft_ms, double other_ms) {
        if ((budget != kNarrow && budget != kWide) || rows < kSlots || rows > budget || kept < kSlots || kept > rows ||
            !(verify_ms > 0 && draft_ms >= 0 && other_ms >= 0) ||
            !std::isfinite(verify_ms + draft_ms + other_ms))
            return;
        Phase& p = budget == kWide ? wide_ : narrow_;
        const double weight = 1.0 / std::min(++p.windows, 16);
        auto add = [weight](double& mean, double value) { mean += weight * (value - mean); };
        add(p.tokens, kept);
        add(p.verify_ms, verify_ms);
        add(p.draft_ms, draft_ms);
        add(p.other_ms, other_ms);
        ++windows_;
        narrow_since_wide_ = budget == kWide ? 0 : narrow_since_wide_ + 1;
    }

    int choose(double weakest_acceptance, double extra_tokens) {
        if (narrow_.windows < 8 || weakest_acceptance < 0.82) return kNarrow;
        if (narrow_since_wide_ >= 128) wide_ = {};
        if (wide_.windows < 4) {
            // Use a conservative prior only to justify a short probe. Its actual cost and yield decide.
            return weakest_acceptance >= 0.90 && extra_tokens > 0.55 * narrow_.tokens ? kWide : kNarrow;
        }
        // Refresh the narrow reference at intervals so a clock or context change moves both estimates.
        if (windows_ % 64 < 4) return kNarrow;
        const double delta_tokens = wide_.tokens - narrow_.tokens;
        const double delta_ms = wide_.ms() - narrow_.ms();
        const double narrow_rate = narrow_.tokens / narrow_.ms();
        return delta_tokens > 0 && delta_tokens > narrow_rate * (delta_ms + 0.03 * wide_.ms()) ? kWide : kNarrow;
    }

private:
    Phase narrow_, wide_;
    unsigned windows_{};
    unsigned narrow_since_wide_{};
};

// A window over its row budget: S slots want want[s] >= 1 rows each, `budget` >= S rows to give.  Every slot keeps its
// own row; each further row goes to the slot whose next draft adds the most expected accepted tokens: scale[s] times
// the product of acc(s, j) over the drafts j it would then verify.  A product of chances only falls as a slot takes
// more drafts, so taking the best next row each time maximizes the expected accepted drafts of the window, and each
// slot gets a prefix of its drafts.  Ties go to the lower slot.  give[s] <= want[s], and sum(give) is
// min(budget, sum(want)).
template <class Acc>
inline void split_rows_by_value(const int* want, const float* scale, int S, int budget, Acc acc, int* give) {
    constexpr int kMaxSlots = 64;
    float chain[kMaxSlots];
    for (int s = 0; s < S; ++s) {
        give[s] = 1;
        chain[s] = scale[s];
    }
    for (int left = budget - S; left > 0; --left) {
        int best = -1;
        float best_v = -1.0f;
        for (int s = 0; s < S; ++s) {
            if (give[s] >= want[s]) continue;
            const float v = chain[s] * acc(s, give[s] - 1);   // the draft this row would verify
            if (v > best_v) { best_v = v; best = s; }
        }
        if (best < 0) break;
        chain[best] = best_v;
        ++give[best];
    }
}

}  // namespace basalt::models::qwen4exp::serve
