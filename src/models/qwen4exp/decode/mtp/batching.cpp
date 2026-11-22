#include "common.hpp"
#include "basalt/core/spin_wait.hpp"

namespace basalt::core {

using namespace mtp_detail;

const float* MtpDrafter::keep_rows(const float* R_rows, int rows, std::string& err) {
    if (rkeep_ == nullptr || rows > kKeepRows) { err = "mtp: no room to keep the window rows"; return nullptr; }
    const OnDevice on_device(device_);
    if (cudaMemcpyAsync(rkeep_, R_rows, (size_t) rows * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return nullptr;
    }
    return rkeep_;
}

bool MtpDrafter::draft_seq(int id, const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                           std::string& err, float* probs, int* n_drafts) {
    if (!flush(id, err) || !(seg_on_ ? target(id, err) : use_store(id, err))) return false;
    const OnDevice on_device(device_);
    const size_t bytes = (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float);
    if (R_rows != window_R_) {   // through a copy: the rows may overlap the ones the graphs read
        if (rstage_ == nullptr) { err = "mtp: no row staging"; return false; }
        if (cudaMemcpyAsync(rstage_, R_rows, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
            cudaMemcpyAsync((void*) window_R_, rstage_, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: staging the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::draft_batch(int S, const int* ids, const float* R_rows, const int* r0s, const int* Ts,
                             const int32_t* const* toks, const int64_t* p0s, const int* as, const float* cum_ths,
                             int32_t* const* drafts, float* const* probs, int* n_drafts, std::string& err) {
    using namespace basalt::kernels;
    if (!seg_on_ || S < 1 || S > kDraftRowsMax) { err = "mtp: a batched draft needs stores in place and 1-4 sequences"; return false; }
    const OnDevice on_device(device_);
    int Ttot = 0;
    for (int s = 0; s < S; ++s) {
        if (ids[s] < 0 || ids[s] >= n_stores_ || Ts[s] < 1 || as[s] < 0 || as[s] >= Ts[s]) {
            err = "mtp: batched draft arguments out of range";
            return false;
        }
        Ttot += Ts[s];
        if (!flush(ids[s], err)) return false;   // its deferred rows, in its own store
    }
    if (Ttot > std::min(max_t_, kVerifyMaxT)) { err = "mtp: a batched draft's window rows exceed the window"; return false; }
    // per sequence, what draft_rows does before its launch: the committed picks into the subset's context, the tier
    for (int s = 0; s < S; ++s) {
        if (n_xcap_ > 0)
            for (int t = 0; t <= as[s]; ++t) ctx_note(toks[s][t]);
        if (!tier2_.empty())
            for (int t = 0; t < Ts[s]; ++t)
                if (toks[s][t] >= 0 && toks[s][t] < (int32_t) tier2_.size() && tier2_[(size_t) toks[s][t]]) tier2_recent_ = 32;
    }
    if (n_xcap_ > 0) ctx_flush(cs_);
    if (!tier2_.empty()) {
        tier_ = tier2_recent_ > 0 ? 1 : 0;
        if (tier2_recent_ > 0) --tier2_recent_;
    }
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    const int cap = step_cap_ > 0 ? std::min(max_steps, step_cap_) : max_steps;
    const int64_t NH = g_->n_head, HCN = g_->hc * g_->n_embd;
    // one upload of rows' records: token, four step words and NH positions each, then the segment rows
    auto upload = [&](int rows, const std::vector<int32_t>& tk, const std::vector<int32_t>& st, const std::vector<int32_t>& ps,
                      const std::vector<QsaSeg>& sg, bool with_tok) {
        brec_.assign(tk.begin(), tk.end());
        brec_.insert(brec_.end(), st.begin(), st.end());
        brec_.insert(brec_.end(), ps.begin(), ps.end());
        const int64_t need = (int64_t) brec_.size();
        if (brec_cap_ < need) {
            if (brec_dev_) cudaFree(brec_dev_);
            brec_dev_ = nullptr;
            brec_cap_ = 0;
            if (cudaMalloc((void**) &brec_dev_, (size_t) need * sizeof(int32_t)) != cudaSuccess) return false;
            brec_cap_ = need;
        }
        // from pageable memory: the copy has the bytes before it returns, so the vectors may change after
        return cudaMemcpyAsync(brec_dev_, brec_.data(), (size_t) need * 4, cudaMemcpyHostToDevice, cs_) == cudaSuccess &&
               cudaMemcpyAsync(seg_rows_, sg.data(), sizeof(QsaSeg) * sg.size(), cudaMemcpyHostToDevice, cs_) == cudaSuccess &&
               (!with_tok || cudaMemcpyAsync(tok_, brec_dev_, (size_t) rows * 4, cudaMemcpyDeviceToDevice, cs_) == cudaSuccess) &&
               cudaMemcpyAsync(step_, brec_dev_ + tk.size(), (size_t) rows * 16, cudaMemcpyDeviceToDevice, cs_) == cudaSuccess &&
               cudaMemcpyAsync(pos_, brec_dev_ + tk.size() + st.size(), (size_t) (rows * NH) * 4, cudaMemcpyDeviceToDevice,
                               cs_) == cudaSuccess;
    };
    auto put_cell = [&](std::vector<int32_t>& st, std::vector<int32_t>& ps, int row, int64_t cell) {
        st[(size_t) row * 4 + 0] = (int32_t) cell;
        st[(size_t) row * 4 + 1] = (int32_t) (cell + 1);
        st[(size_t) row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        st[(size_t) row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) ps[(size_t) (row * NH + h)] = (int32_t) cell;
    };
    // ---- the catch-up: every sequence's window rows' K/V in one forward, each row in its sequence's slot
    {
        std::vector<int32_t> tk((size_t) Ttot), st((size_t) Ttot * 4), ps((size_t) (Ttot * NH));
        std::vector<QsaSeg> sg((size_t) kSegRows, QsaSeg{n_stores_, 0, 1, 0});
        int r = 0;
        for (int s = 0; s < S; ++s) {
            note_cells_of(ids[s], p0s[s], p0s[s] + Ts[s] + max_t_ + 1);
            if (cudaMemcpyAsync(Rin_ + (size_t) r * HCN, R_rows + (size_t) r0s[s] * HCN, (size_t) Ts[s] * HCN * sizeof(float),
                                cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) { err = "mtp: staging the window rows failed"; return false; }
            for (int t = 0; t < Ts[s]; ++t, ++r) {
                tk[(size_t) r] = toks[s][t];
                put_cell(st, ps, r, p0s[s] + t);
                sg[(size_t) r] = QsaSeg{ids[s], 0, 1, 0};
            }
        }
        if (!capture_catch(Ttot, err)) return false;
        if (!upload(Ttot, tk, st, ps, sg, true) || cudaGraphLaunch(catch_exec_[Ttot], cs_) != cudaSuccess) {
            err = std::string("mtp batched catch-up: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    // ---- the draft steps: one full-layer row per sequence, at its draft cell, until every sequence's rule stops it
    if (!capture_rows(S, err)) return false;
    bool alive[kDraftRowsMax] = {};
    float pr[kDraftRowsMax][kMtpChainMax] = {};
    for (int s = 0; s < S; ++s) { alive[s] = cap > 0; n_drafts[s] = 0; }
    std::vector<QsaSeg> sg((size_t) kSegRows, QsaSeg{n_stores_, 0, 1, 0});
    for (int s = 0; s < S; ++s) sg[(size_t) s] = QsaSeg{ids[s], 0, 1, 0};
    for (int j = 0; j < cap; ++j) {
        std::vector<int32_t> tk((size_t) S), st((size_t) S * 4), ps((size_t) (S * NH));
        for (int s = 0; s < S; ++s) {
            tk[(size_t) s] = toks[s][as[s]];
            put_cell(st, ps, s, p0s[s] + as[s] + j);
        }
        // row s's input: the window row it drafts from, then its previous step's final residual
        for (int s = 0; s < S; ++s) {
            const float* src = j == 0 ? R_rows + (size_t) (r0s[s] + as[s]) * HCN : R_ + (size_t) s * HCN;
            if (cudaMemcpyAsync(Rin_ + (size_t) s * HCN, src, (size_t) HCN * sizeof(float), cudaMemcpyDeviceToDevice,
                                cs_) != cudaSuccess) { err = "mtp: staging the draft rows failed"; return false; }
        }
        for (int s = 0; s < S; ++s) ((volatile int32_t*) h_bout_)[s] = -1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!upload(S, tk, st, ps, sg, j == 0) || cudaGraphLaunch(rows_exec_[tier_][S], cs_) != cudaSuccess) {
            err = std::string("mtp batched draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        (void) cudaStreamQuery(cs_);
        for (int s = 0; s < S; ++s) {   // the rows' drafts arrive together (one launch)
            uint32_t spins = 0;
            SpinWait spin;
            while (((volatile int32_t*) h_bout_)[s] < 0) {
                spin();
                if ((++spins & (spin.yielding() ? 63u : 1023u)) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady &&
                    ((volatile int32_t*) h_bout_)[s] < 0) {
                    err = std::string("mtp batched draft ended without a draft: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            }
        }
        bool any = false;
        for (int s = 0; s < S; ++s) {
            if (!alive[s]) continue;
            drafts[s][j] = ((volatile int32_t*) h_bout_)[s];
            pr[s][j] = ((volatile float*) h_bprob_)[s];
            if (probs) probs[s][j] = pr[s][j];
            ++n_drafts[s];
            alive[s] = j + 1 < cap && draft_keep(pr[s], j, cum_ths[s]);   // the rule draft_rows applies
            any = any || alive[s];
        }
        if (!any) break;
    }
    for (int s = 0; s < S; ++s)
        for (int j = n_drafts[s]; j < max_t_ - 1; ++j) { drafts[s][j] = 0; if (probs) probs[s][j] = 0.0f; }
    return true;
}

bool MtpDrafter::flush(int id, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size() || pend_tok_[(size_t) id].empty()) return true;
    std::vector<int32_t> tk;
    tk.swap(pend_tok_[(size_t) id]);
    if (!(seg_on_ ? target(id, err) : use_store(id, err))) return false;
    return prefill_cur(pend_R_[(size_t) id], tk.data(), (int64_t) tk.size(), pend_cell0_[(size_t) id], err);
}

bool MtpDrafter::defer_rows(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size()) { err = "mtp: no such K/V store"; return false; }
    std::vector<int32_t>& tk = pend_tok_[(size_t) id];
    if (!tk.empty() && (pend_cell0_[(size_t) id] + (int64_t) tk.size() != cell0 || (int64_t) tk.size() + n > kPendRows)) {
        if (!flush(id, err)) return false;
    }
    if (n > kPendRows) return catch_up(id, R_rows, next_tokens, n, cell0, err);
    const OnDevice on_device(device_);
    const size_t HCN = (size_t) (g_->hc * g_->n_embd);
    if (tk.empty()) pend_cell0_[(size_t) id] = cell0;
    if (cudaMemcpyAsync(pend_R_[(size_t) id] + tk.size() * HCN, R_rows, (size_t) n * HCN * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return false;
    }
    tk.insert(tk.end(), next_tokens, next_tokens + n);
    return true;
}

bool MtpDrafter::catch_up(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!flush(id, err) || !(seg_on_ ? target(id, err) : use_store(id, err))) return false;
    return prefill_cur(R_rows, next_tokens, n, cell0, err);
}


}
