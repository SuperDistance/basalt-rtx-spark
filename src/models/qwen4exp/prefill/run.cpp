#include "run.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/device_memory.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/models/qwen4exp/decode/layout.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    Run run(*this, tokens, n, pos0, err);
    return run.execute();
}

bool Prefill::Run::execute() {
    count_expert_groups = core::exp_flag("n1x_prefill_group_stats") != 0;
    // an integrated GPU reads the experts pinned in RAM where they lie instead of staging copies of them (the same
    // bytes through the same kernels; the N1X reads every prompt size faster this way)
    direct_ram_experts = core::device_memory_model().integrated && owner.stage_lb_ == 0 && owner.next_ == nullptr &&
                         !m.ring_peer && !m.remote_res && !m.pp;
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    owner.rows_in_place_ = false;
    if (m.hist_on && !rows_in_place) m.hist_tokens += n;   // a kept chunk's tokens counted in the run that kept it
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    for (int64_t c0 = 0; c0 < n; c0 += chunk_len(c0)) {
        if (owner.should_stop && owner.should_stop()) { err = "cancelled"; return false; }
        Chunk chunk(*this, c0);
        if (!chunk.execute()) return false;
        const int flow = chunk.finish();
        if (flow < 0) return false;
        if (flow > 0) break;
    }
    return finish();
}

bool Prefill::Run::finish() {
    // Do not drain the successor here. This is the overlap: an intermediate
    // stage can return while later GPUs are still processing the previous chunk.
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // an expert copy that failed on the copy stream surfaces here, not in the next request
    if (const cudaError_t cst = cudaStreamSynchronize(m.copy); cst != cudaSuccess) {
        err = std::string("prefill: expert copy stream: ") + cudaGetErrorString(cst);
        return false;
    }
    owner.stats_.ms_total += ms_since(t_start);
    if (direct_ram_experts) {
        const PrefillStats& s1 = owner.stats_;
        basalt::log::info("basalt prefill", "direct RAM expert groups for %lld prompt rows: %lld; staged %lld blobs "
                          "(%lld by DMA) in %.1f ms of host time, %lld resident groups, %lld CPU experts, "
                          "%lld chunks in %.0f ms",
                          (long long) n, (long long) (direct_ram_groups + s1.experts_direct - stats0.experts_direct),
                          (long long) (s1.experts_streamed - stats0.experts_streamed),
                          (long long) (s1.experts_dma - stats0.experts_dma), s1.ms_experts_host - stats0.ms_experts_host,
                          (long long) (s1.experts_resident - stats0.experts_resident),
                          (long long) (s1.experts_cpu - stats0.experts_cpu), (long long) (s1.chunks - stats0.chunks),
                          s1.ms_total - stats0.ms_total);
    }
    if (count_expert_groups) {
        const auto& h = nonresident_expert_groups;
        basalt::log::info("basalt prefill", "nonresident expert groups for %lld prompt rows: 1-7=%llu 8-31=%llu 32-63=%llu 64+=%llu",
                          (long long) n, (unsigned long long) h[0], (unsigned long long) h[1],
                          (unsigned long long) h[2], (unsigned long long) h[3]);
    }
    return true;
}

bool Prefill::Run::Chunk::execute() {
    core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);
    ++owner.stats_.chunks;
    if (!inputs()) return false;
    if (!start_stream()) return false;
    m.ahead_layer[0] = m.ahead_layer[1] = m.guess_layer = -1;
    for (int64_t l = LB; l < LE; ++l) {
        if (!layer(l)) return false;
    }
    return true;
}

}  // namespace basalt::prefill
