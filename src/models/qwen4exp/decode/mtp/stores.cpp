#include "common.hpp"
#include "basalt/core/device_memory.hpp"

namespace basalt::core {

using namespace mtp_detail;

int MtpDrafter::kv_layout(KvArr* out) const {
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const uint64_t rows = (uint64_t) s.n_head_kv * (uint64_t) s.page_size;
    const uint64_t hd = (uint64_t) s.head_dim;
    if (st_.kv_hybrid) return 0;
    if (st_.kv_q4) {
        const uint64_t q4 = rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim);
        out[0] = {st_.k_q4, q4};
        out[1] = {st_.v_q4, q4};
        return 2;
    }
    if (st_.kv_int8) {
        const uint64_t sc = rows * (hd / basalt::kernels::KV_Q8_GROUP) * 2;
        out[0] = {st_.k_q, rows * hd};
        out[1] = {st_.v_q, rows * hd};
        out[2] = {st_.k_scale, sc};
        out[3] = {st_.v_scale, sc};
        return 4;
    }
    out[0] = {st_.k_pool, rows * hd * 2};
    out[1] = {st_.v_pool, rows * hd * 2};
    return 2;
}

void MtpDrafter::note_cells(int64_t lo, int64_t hi) { note_cells_of(writer(), lo, hi); }

void MtpDrafter::note_cells_of(int id, int64_t lo, int64_t hi) {
    hi = std::min(hi, st_.max_cells);
    if (hi <= lo) return;
    if (id == cur_) {   // the working state's range not yet saved to its store (a store written in place has none)
        if (dirty_hi_ <= dirty_lo_) { dirty_lo_ = lo; dirty_hi_ = hi; }
        else { dirty_lo_ = std::min(dirty_lo_, lo); dirty_hi_ = std::max(dirty_hi_, hi); }
    }
    if (store_hi_.empty()) return;
    // the sequence ends where its latest write ends; a write that starts well below the end (another prompt, a
    // rolled-back turn) starts it again there, past the cells a window writes ahead of its position
    int64_t& end = store_hi_[(size_t) id];
    end = lo + 4 * (int64_t) max_t_ + 64 < end ? hi : std::max(end, hi);
}

bool MtpDrafter::init_seg(std::string& err) {
    using namespace basalt::kernels;
    seg_on_ = false;
    tgt_ = cur_;
    const QsaShapes s = shapes_of(*g_);
    if (n_stores_ <= 0 || !st_.kv_int8 || st_.kv_q4 || st_.kv_rot || st_.kv_hybrid || !qsa_dec_fused_supported(s) ||
        !qsa_dec_attn_supported(qsa_attn_pools(st_), s) || (st_.kv_mode != 0 && st_.kv_mode != 2))
        return true;   // the stores are swapped through st_ (use_store)
    const int n = n_stores_ + 1;
    const int64_t pages = st_.n_pages;
    std::vector<QsaSegSlot> slots((size_t) n);
    std::vector<int32_t> pt((size_t) (n_stores_ * pages));
    for (int id = 0; id < n_stores_; ++id) {   // a store: its ring buffers, page p at p % ring, no host copy
        const int64_t ring = std::max<int64_t>(1, store_ring_[(size_t) id]);
        for (int64_t p = 0; p < pages; ++p) pt[(size_t) (id * pages + p)] = (int32_t) (p % ring);
        QsaSegSlot& sl = slots[(size_t) id];
        sl.k_q = (int8_t*) stores_[(size_t) id][0]; sl.v_q = (int8_t*) stores_[(size_t) id][1];
        sl.k_scale = (uint16_t*) stores_[(size_t) id][2]; sl.v_scale = (uint16_t*) stores_[(size_t) id][3];
        sl.max_cells = (int32_t) st_.max_cells;
    }
    {   // the main sequence: the working state as qsa_dec_kv_post and qsa_dec_attn read it
        QsaSegSlot& sl = slots[(size_t) n_stores_];
        sl.k_q = st_.k_q; sl.v_q = st_.v_q; sl.k_scale = st_.k_scale; sl.v_scale = st_.v_scale;
        sl.page_table = st_.page_table; sl.host = st_.host; sl.max_cells = (int32_t) st_.max_cells;
    }
    std::vector<QsaSeg> rows((size_t) n * kSegRows);
    for (int id = 0; id < n; ++id)
        for (int r = 0; r < kSegRows; ++r) rows[(size_t) (id * kSegRows + r)] = QsaSeg{id, 0, 1, 0};
    if (cudaMalloc(&seg_slots_, sizeof(QsaSegSlot) * (size_t) n) != cudaSuccess ||
        cudaMalloc(&seg_rows_, sizeof(QsaSeg) * kSegRows) != cudaSuccess ||
        cudaMalloc(&seg_rows_of_, sizeof(QsaSeg) * rows.size()) != cudaSuccess ||
        cudaMalloc(&store_pt_, sizeof(int32_t) * pt.size()) != cudaSuccess) {
        err = "mtp: the store tables do not fit";
        return false;
    }
    for (int id = 0; id < n_stores_; ++id) slots[(size_t) id].page_table = store_pt_ + id * pages;
    using basalt::core::device_copy;   // pageable tables the draft kernels read on another stream: landed on return
    if (device_copy(store_pt_, pt.data(), sizeof(int32_t) * pt.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        device_copy(seg_slots_, slots.data(), sizeof(QsaSegSlot) * slots.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        device_copy(seg_rows_of_, rows.data(), sizeof(QsaSeg) * rows.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        device_copy(seg_rows_, rows.data() + (size_t) n_stores_ * kSegRows, sizeof(QsaSeg) * kSegRows,
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "mtp: uploading the store tables failed";
        return false;
    }
    vram_ += sizeof(QsaSegSlot) * slots.size() + sizeof(QsaSeg) * (rows.size() + kSegRows) + sizeof(int32_t) * pt.size();
    seg_on_ = true;
    tgt_ = n_stores_;
    return true;
}

basalt::kernels::QsaSegTable MtpDrafter::seg_table() const {
    basalt::kernels::QsaSegTable t;
    t.slots = seg_slots_; t.rows = seg_rows_; t.segs = seg_rows_; t.n_rows = kSegRows; t.n_seg = 1;
    return t;
}

bool MtpDrafter::stage_target(std::string& err) {
    if (!seg_on_) return true;
    if (tgt_ < 0 || tgt_ > n_stores_) { err = "mtp: no such K/V store"; return false; }
    if (cudaMemcpyAsync(seg_rows_, seg_rows_of_ + (size_t) tgt_ * kSegRows, sizeof(basalt::kernels::QsaSeg) * kSegRows,
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: staging the target sequence failed";
        return false;
    }
    return true;
}

void MtpDrafter::copy_pages(int to, bool to_work, int from, bool from_work, int64_t a, int64_t b, const KvArr* lay,
                            int na) {
    // the working state holds every page in place, or with KV streaming its ring (`block % n_slots`), which reaches as
    // far back as any store (load)
    const int64_t work_ring = st_.kv_mode == 2 ? st_.n_slots : 0;
    auto at = [&](int id, bool work, int i, int64_t p) {
        const int64_t ring = work ? work_ring : store_ring_[(size_t) id];
        return (uint8_t*) (work ? lay[i].p : stores_[(size_t) id][(size_t) i]) +
               (uint64_t) (ring > 0 ? p % ring : p) * lay[i].page_bytes;
    };
    auto run = [&](int id, bool work, int64_t p) {   // pages from p before that side's ring wraps
        const int64_t ring = work ? work_ring : store_ring_[(size_t) id];
        return ring > 0 ? ring - p % ring : b - p;
    };
    for (int64_t p = a; p < b;) {
        const int64_t n = std::min({b - p, run(to, to_work, p), run(from, from_work, p)});
        for (int i = 0; i < na; ++i)
            cudaMemcpyAsync(at(to, to_work, i, p), at(from, from_work, i, p), (size_t) ((uint64_t) n * lay[i].page_bytes),
                            cudaMemcpyDeviceToDevice, cs_);
        p += n;
    }
}

bool MtpDrafter::use_store(int id, std::string& err) {
    if (seg_on_) {   // st_ always holds the main sequence; the stores are addressed in place (target)
        if (id != n_stores_) { err = "mtp: a store is not swapped in while stores are read in place"; return false; }
        tgt_ = id;
        return true;
    }
    if (id == cur_) return true;
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    if (dirty_hi_ > dirty_lo_) {   // the written pages go back to the sequence that held them, as far as its ring reaches
        const size_t c = (size_t) cur_;
        const int64_t pg1 = std::min<int64_t>(store_pages_[c], (std::min(dirty_hi_, store_hi_[c]) + ps - 1) / ps);
        copy_pages(cur_, false, cur_, true, std::max<int64_t>(dirty_lo_ / ps, pg1 - store_ring_[c]), pg1, lay, na);
    }
    // the new sequence's last pages: all its window reads
    const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (store_hi_[(size_t) id] + ps - 1) / ps);
    copy_pages(id, true, id, false, std::max<int64_t>(0, n1 - store_ring_[(size_t) id]), n1, lay, na);
    cur_ = id;
    dirty_lo_ = dirty_hi_ = 0;
    if (cudaGetLastError() != cudaSuccess) { err = "mtp: swapping the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::copy_store(int from, int to, int64_t cells, std::string& err) {
    if (n_stores_ == 0 || from == to) return true;
    if (from < 0 || from > n_stores_ || to < 0 || to > n_stores_) { err = "mtp: no such K/V store"; return false; }
    if (!flush(from, err)) return false;
    pend_tok_[(size_t) to].clear();
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    const int64_t n1 = std::min<int64_t>({store_pages_[(size_t) from], store_pages_[(size_t) to], (cells + ps - 1) / ps});
    // the last pages both sides keep (a store its ring, the working state all)
    const int64_t keep = std::min(from == cur_ ? n1 : store_ring_[(size_t) from], to == cur_ ? n1 : store_ring_[(size_t) to]);
    copy_pages(to, to == cur_, from, from == cur_, std::max<int64_t>(0, n1 - keep), n1, lay, na);
    if (to == cur_) note_cells_of(to, 0, cells);
    store_hi_[(size_t) to] = std::min<int64_t>({cells, st_.max_cells, store_pages_[(size_t) to] * ps});
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: copying the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::save_store(int id, int64_t cells, ConversationKv& out, std::string& err) {
    if (n_stores_ == 0) return conversation_kv_save(out, st_, *g_, cells, false, err);
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const int prev = cur_;
    if (!pend_tok_[(size_t) id].empty()) {
        // its deferred rows are computed in the working state (in its own store when stores are in place); the
        // sequence there comes back whole within its ring
        if (!seg_on_) note_cells(0, store_hi_[(size_t) prev]);
        if (!flush(id, err)) return false;
    }
    // the image comes from the host copy, which a ring shares with every sequence: the one it holds puts its pages
    // back first (another sequence's come from its store, below)
    if (cur_ == id && !ring_to_host(err)) return false;
    if (!conversation_kv_save(out, st_, *g_, cells, false, err)) return false;
    if (cur_ != id) {   // the pages the store keeps, over the working state's
        const OnDevice on_device(device_);
        KvArr lay[4];
        const int na = kv_layout(lay);
        ConversationBuffer* dst[4] = {&out.k, &out.v, &out.k_scale, &out.v_scale};
        const int64_t ps = shapes_of(*g_).page_size, ring = store_ring_[(size_t) id];
        const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (cells + ps - 1) / ps);
        for (int64_t p = std::max<int64_t>(0, n1 - ring); p < n1;) {
            const int64_t n = std::min(n1 - p, ring - p % ring);
            for (int i = 0; i < na; ++i) {
                const uint8_t* src = (const uint8_t*) stores_[(size_t) id][(size_t) i] +
                                     (uint64_t) (p % ring) * lay[i].page_bytes;
                const size_t off = (size_t) ((uint64_t) p * lay[i].page_bytes);
                if (!dst[i]->visit(off, (size_t) ((uint64_t) n * lay[i].page_bytes), [&](uint8_t* q, size_t m, size_t at) {
                        return basalt::core::device_copy(q, src + (at - off), m, cudaMemcpyDeviceToHost) == cudaSuccess;
                    })) {
                    err = "mtp: saving the K/V failed";
                    return false;
                }
            }
            p += n;
        }
    }
    return cur_ == prev || use_store(prev, err);
}

bool MtpDrafter::ring_to_host(std::string& err) {
    if (st_.kv_mode != 2 || n_stores_ == 0) return true;
    const OnDevice on_device(device_);
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const size_t c = (size_t) cur_;
    const int64_t n1 = std::min<int64_t>(store_pages_[c], (store_hi_[c] + s.page_size - 1) / s.page_size);
    basalt::kernels::kv_ring_save(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_),
                                  std::max<int64_t>(0, n1 - store_ring_[c]), n1, st_.n_slots, s, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: saving the draft ring failed"; return false; }
    return true;
}

QsaState& MtpDrafter::kv_state_rw() {
    std::string e;
    if (!use_main(e)) basalt::log::error("basalt mtp", "%s", e.c_str());
    return st_;
}


}
