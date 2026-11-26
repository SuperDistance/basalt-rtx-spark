// src/models/qwen4exp/serve/batch.cpp - batch operations for the persistent engine.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/core/device_memory.hpp"
#include "basalt/protocol.hpp"
#include "qfn_state.hpp"

#include <algorithm>
#include <cstring>
#include "../../../program/serve/logprobs.hpp"

namespace basalt::models::qwen4exp::serve {

namespace {
// stable insertion sort for the few slots a window holds
template <class Less>
void sort_small(int* v, int n, Less less) {
    for (int i = 1; i < n; ++i) {
        const int x = v[i];
        int j = i;
        for (; j > 0 && less(x, v[j - 1]); --j) v[j] = v[j - 1];
        v[j] = x;
    }
}
}  // namespace

uint64_t QfnRuntime::row_hash(int t) {
    lh_buf.resize((size_t) n_vocab);
    if (!ver.copy_logits(t, lh_buf.data())) return 0;
    uint64_t h = 1469598103934665603ull;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(lh_buf.data());
    for (size_t i = 0; i < lh_buf.size(); ++i) { h ^= w[i]; h *= 1099511628211ull; }
    return h;
}

// one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
// so it keeps the conversation it holds
bool QfnRuntime::batch_step() {
    constexpr int MR = basalt::kernels::kVerifyMaxT;
    // Keep one verifier row per active slot. Larger batches also need room to verify a draft for each slot.
    const int row_cap = std::min(MR, ver.max_t());
    const double requested_rows = basalt::core::exp_flag("batch_rows", 0);
    const int fixed_rows = requested_rows >= 1.0 ? (int) std::min(requested_rows, (double) row_cap) : 0;
    auto row_budget = [&](int n) {
        return std::clamp(fixed_rows > 0 ? fixed_rows : std::max(8, 2 * n), n, row_cap);
    };
    int S = 0, R = 0;
    int rows[MR] = {}, lens[MR] = {}, n_keep[MR] = {}, r0s[MR] = {};
    int32_t tok[MR] = {}, outb[MR] = {};
    int64_t pos[MR] = {}, p0s[MR] = {};
    std::vector<int32_t> drafts[MR];
    // the elastic KV: each slot's next cells backed ahead of its windows; beside a prompt read only from free
    // VRAM, and a slot that needs more than that waits for the read to end
    std::vector<uint8_t> kv_wait(bs.size(), 0);
    if (kv_floor > 0)
        for (int b = 0; b < (int) bs.size(); ++b) {
            if (!bs[(size_t) b].active) continue;
            basalt::core::SessionState& u = *bslot_ss[0][(size_t) b];
            const int64_t backed = basalt::core::session_kv_cells(u), at = bs[(size_t) b].p;
            if (at + 2 * MR + 64 <= backed) continue;
            std::string ke;
            if (!ov_active && ba_job) { ba_thr.join(); ba_job = false; }   // its copies land before the drain
            if (kv_fit(u, at + kKvGrowCells, ke, !ov_active, at + 2 * MR + 64)) continue;
            if (!ke.empty() && kv_short) {   // the expert cache's floor: this slot's reply ends here
                BSlot& sl = bs[(size_t) b];
                basalt::log::warn("basalt batch", "slot %d stopped at %lld cells: %s", b, (long long) at, ke.c_str());
                std::string fe;   // its sessions keep its state: what its last commit left to a next window lands now
                if (!ver.flush_deferred(fe)) { std::printf("ERR %s\n", fe.c_str()); return false; }
                std::printf("BDONE %d %lld length %.1f %lld %lld %lld\n", b, (long long) sl.produced,
                            std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count(),
                            (long long) sl.drafts_accepted, (long long) sl.drafts_offered, (long long) sl.windows);
                sl.active = false;
                sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids
                sl.idle_busy = busy_now();
                sl.guard.reset();
                kv_wait[(size_t) b] = 1;
                continue;
            }
            if (!ke.empty()) { std::printf("ERR %s\n", ke.c_str()); return false; }
            if (at + 2 * MR > backed) kv_wait[(size_t) b] = 1;
        }
    for (int b = 0; b < (int) bs.size() && S < row_cap; ++b)
        if (bs[(size_t) b].active && !kv_wait[(size_t) b]) {
            rows[S] = b;
            pos[S] = bs[(size_t) b].p;
            lens[S] = 1;
            ++S;
        }
    if (S == 0) {
        // A KV wait can suspend the entire cohort before any row is verified.
        if (batch_row_prepared != 0) {
            if (batch_row_cost_trace)
                basalt::log::info("basalt batch rows", "cohort suspended, prepared=%d next_budget=%d",
                                  batch_row_prepared, BatchRowCost::kNarrow);
            batch_row_cost.reset();
            batch_row_prepared = 0;
            batch_row_commit_tail = 0.0;
        }
        return true;
    }
    bool scripted = false;
    for (int sg = 0; sg < S; ++sg) scripted = scripted || !bs[(size_t) rows[sg]].script.empty();
    const bool real = batch_spec && !scripted;
    auto quiet = [&] {
        std::lock_guard<std::mutex> lock(in_mu);
        return in_lines.empty() && !in_eof;
    };
    auto steady_slots = [&] {
        int64_t newest = INT64_MAX, oldest = 0;
        for (int sg = 0; sg < S; ++sg) {
            const int b = rows[sg];
            const BSlot& slot = bs[(size_t) b];
            if (slot.stop || slot.windows < 16 || slot.max_new - slot.produced < 64 ||
                slot_cap[(size_t) b] - slot.p < 64)
                return false;
            newest = std::min(newest, slot.windows);
            oldest = std::max(oldest, slot.windows);
        }
        return oldest - newest <= 12;
    };
    const bool row_cost_window = batch_row_cost_enabled && real && S == BatchRowCost::kSlots &&
                                 !ov_active && admit_slot < 0 && steady_slots() && quiet() && ver.all_routed_resident();
    const int previous_budget = batch_row_prepared;
    const int MB = row_cost_window && previous_budget == BatchRowCost::kWide ? BatchRowCost::kWide : row_budget(S);
    batch_row_prepared = 0;
    if (!row_cost_window && previous_budget != 0) batch_row_cost.reset();
    if (real) {
        // Size each slot's window with its draft threshold and prompt-lookup policy.
        int Tw[MR] = {};
        for (int sg = 0; sg < S; ++sg) {
            BSlot& sl = bs[(size_t) rows[sg]];
            int T = 1;
            while (T < S_mtp && T - 1 < sl.n_dr && basalt::core::draft_keep(sl.dprob, T - 1, sl.cum.th())) ++T;
            sl.from_sfx = false;
            sl.sk = 0;
            if (sl.sfx && sl.n_dr > 0) {
                sl.sk = sl.sfx->propose(S_all - 1, sl.sbuf);
                sl.smatch = sl.sfx->last_match();
                if (sl.sk > 0 && sl.sbuf[0] == sl.dr[0]) {
                    const basalt::spec::DraftPolicy::Pick pk = bpolicy->choose(T, sl.sk, sl.smatch);
                    if (pk.lookup) { T = pk.t; sl.from_sfx = true; }
                }
            }
            const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 1;
            T = (int) std::max<int64_t>(1, std::min<int64_t>(T, room));
            Tw[sg] = T;
        }
        int sum = 0;
        for (int sg = 0; sg < S; ++sg) sum += Tw[sg];
        if (sum > MB && batch_rows_value) {
            // Rows by expected accepted tokens: every slot keeps its own row, and each further row goes to the slot
            // whose next draft adds the most - the chance that it and all the slot's drafts before it are accepted
            // (the draft-length rule's bin table, scaled by how the slot's drafts have run against it).  Within a
            // slot that chance only falls, so taking the best next row each time is the best split of the budget.
            // Only which drafts are verified changes: every slot's tokens stay its solo tokens.
            const auto acc = [&](int sg, int j) -> float {
                const BSlot& sl = bs[(size_t) rows[sg]];
                return sl.from_sfx ? (float) bpolicy->lookup_rate(sl.smatch)
                       : j < sl.n_dr ? basalt::core::draft_cum_acc(sl.dprob[j]) : 0.0f;
            };
            int give[MR];
            float scale[MR];
            for (int sg = 0; sg < S; ++sg) {
                const BSlot& sl = bs[(size_t) rows[sg]];
                scale[sg] = sl.cum.exp >= 0.5 ? (float) std::clamp(sl.cum.obs / sl.cum.exp, 0.5, 1.5) : 1.0f;
            }
            split_rows_by_value(Tw, scale, S, MB, acc, give);
            if (batch_rows_value_trace) {
                // what the even split would have expected, for the same windows
                int even[MR], ord[MR];
                for (int sg = 0; sg < S; ++sg) { ord[sg] = sg; even[sg] = Tw[sg]; }
                sort_small(ord, S, [&](int x1, int x2) { return Tw[x1] < Tw[x2]; });
                for (int i = 0, left = MB; i < S; ++i) {
                    even[ord[i]] = std::min(even[ord[i]], (left + (S - i) - 1) / (S - i));
                    left -= even[ord[i]];
                }
                auto expected = [&](const int* t) {
                    double e = 0.0;
                    for (int sg = 0; sg < S; ++sg) {
                        double c = 1.0;
                        for (int j = 0; j + 1 < t[sg]; ++j) {
                            c *= acc(sg, j);
                            e += c;
                        }
                    }
                    return e;
                };
                brv_even += expected(even);
                brv_value += expected(give);
                if (++brv_windows % 256 == 0)
                    basalt::log::info("basalt batch rows", "value split: %llu windows over budget, expected drafts "
                                      "accepted per window %.3f (even split %.3f)", (unsigned long long) brv_windows,
                                      brv_value / (double) brv_windows, brv_even / (double) brv_windows);
            }
            for (int sg = 0; sg < S; ++sg) Tw[sg] = give[sg];
        } else if (sum > MB) {
            // Serve the slots wanting the fewest rows first. Each share leaves a row for every remaining slot.
            int ord[MR];
            for (int sg = 0; sg < S; ++sg) ord[sg] = sg;
            sort_small(ord, S, [&](int x1, int x2) { return Tw[x1] < Tw[x2]; });
            int left_rows = MB;
            for (int i = 0; i < S; ++i) {
                const int share = (left_rows + (S - i) - 1) / (S - i);
                Tw[ord[i]] = std::min(Tw[ord[i]], share);
                left_rows -= Tw[ord[i]];
            }
        }
        for (int sg = 0; sg < S; ++sg) {
            BSlot& sl = bs[(size_t) rows[sg]];
            lens[sg] = Tw[sg];
            const int32_t* src = sl.from_sfx ? sl.sbuf : sl.dr;
            drafts[sg].assign(src, src + (Tw[sg] - 1));
            // drawn drafts are kept by min(1, p/q) against the proposals they were drawn from
            if (sl.drawn && !sl.from_sfx && Tw[sg] > 1 && !sl.qids.empty())
                ver.set_seg_drawn(sg, sl.qids.data(), sl.qp.data(), basalt::kernels::kDraftDrawQ);
        }
    } else {   // the rows beyond one per slot go to the scripted slots, as many drafts as the script and the context allow
        int extra = MB - S;
        const int first = (int) (bs[(size_t) rows[0]].rng() % (uint64_t) S);
        for (int i = 0; i < S && extra > 0; ++i) {
            const int sg = (first + i) % S;
            BSlot& sl = bs[(size_t) rows[sg]];
            if (sl.script.empty()) continue;
            const int64_t avail = (int64_t) sl.script.size() - sl.produced;
            const int64_t room = slot_cap[(size_t) rows[sg]] - sl.p - 2;
            const int want = (int) std::min<int64_t>({(int64_t) (sl.rng() % (uint64_t) (extra + 1)), avail, room});
            if (want <= 0) continue;
            drafts[sg].assign(sl.script.begin() + sl.produced, sl.script.begin() + sl.produced + want);
            if (sl.rng() % 2 == 0) {   // a wrong draft at a random place
                const size_t c = (size_t) (sl.rng() % (uint64_t) want);
                drafts[sg][c] = (int32_t) ((drafts[sg][c] + 1 + sl.rng() % 100) % n_vocab);
            }
            lens[sg] += want;
            extra -= want;
        }
    }
    for (int sg = 0; sg < S; ++sg) {
        r0s[sg] = R;
        tok[R] = bs[(size_t) rows[sg]].x;
        for (size_t j = 0; j < drafts[sg].size(); ++j) tok[R + 1 + (int) j] = drafts[sg][j];
        R += lens[sg];
    }
    // the penalty histories, one per row: a slot's row j follows the tokens its state holds, the fed-back token and
    // the drafts before it, as a solo window's row does (penalty_rows); a slot without penalties never reads its rows
    int pen_h = 0;
    for (int sg = 0; sg < S; ++sg) pen_h = std::max(pen_h, bs[(size_t) rows[sg]].pen_n);
    if (pen_h > 0) {
        for (int sg = 0; sg < S; ++sg) {
            const BSlot& sl = bs[(size_t) rows[sg]];
            basalt::kernels::penalty_rows(sl.ids.data(), (int64_t) sl.ids.size(), tok + r0s[sg], lens[sg], pen_h,
                                          hist_stage.data() + (size_t) r0s[sg] * (size_t) pen_h);
        }
        const basalt::core::OnDevice on_h(hist_dev);
        if (basalt::core::device_copy(d_hist, hist_stage.data(), (size_t) R * (size_t) pen_h * sizeof(int32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::printf("ERR batch: the penalty histories could not be staged\n");
            return false;
        }
    }
    ver.set_batch_history(pen_h > 0 ? d_hist : nullptr, pen_h);
    const bool was_busy = basalt::core::progress().busy.load();
    basalt::core::progress().busy.store(true);
    drive.d.layers = 0;
    drive.d.experts = 0;
    drive.d.failed = false;
    if (!ov_active) apply_pending(false);
    if (batch_adapt && !ov_active) {
        // the tier follows what all the slots' rows route: a round every --adapt-every windows.  Its steps
        // are taken once their copies are done; the copies out of the RAM tier (which may read the experts
        // file) are queued from a worker thread beside the windows, so no window waits for one.
        const Clock::time_point ta = Clock::now();
        bool ok = true, acted = false;
        ++ba_win;
        if (ba_job && ba_thr.finished()) { ba_thr.join(); ba_job = false; }
        if (!ba_job) {
            if (xq_stage != 0) {
                if (cudaEventQuery(adapt_ev) == cudaSuccess) {
                    const int was = xq_stage;
                    ok = xq_advance(true);
                    if (ok && was == 1 && xq_stage == 2) {
                        ba_thr.run([&] { ver.flush_post_launch(); });
                        ba_job = true;
                    }
                    acted = true;
                }
            } else if (ba_win % o.adapt_every == 0) {
                const int64_t np0 = xq_noplace;
                ok = xq_round();
                ba_swaps += (int64_t) xq.size();
                ba_noplace += xq_noplace - np0;
                ++ba_rounds;
                acted = true;
            }
        }
        std::string perr;
        if (ok && acted && !ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
        if (!ok) { std::printf("ERR an adaptive refill failed\n"); return false; }
        ba_ms += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
        if (ba_win % 2000 == 0 && ba_win > 0) {   // the swap rate and the hit rate over the last 2000 windows
            const int64_t h = drive.d.cache_hits, m = drive.d.multi_misses;
            basalt::log::info("basalt batch adapt", "window %lld: %lld swaps in %lld rounds, VRAM hits %.2f and CPU "
                                 "experts %.2f per layer-window", (long long) ba_win, (long long) (ba_swaps - ba_swaps_l),
                         (long long) (ba_rounds - ba_rounds_l), (double) (h - ba_h_l) / (2000.0 * g.n_layers),
                         (double) (m - ba_m_l) / (2000.0 * g.n_layers));
            ba_swaps_l = ba_swaps; ba_rounds_l = ba_rounds; ba_h_l = h; ba_m_l = m;
        }
    }
    if (bt_windows == 0) {
        bt_start = Clock::now();
        bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
        bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
        bt_offload0 = drive.d.offload_entries;
    }
    const Clock::time_point w0 = Clock::now();
    if (!ver.run_slot_segs(rows, lens, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
        std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
        return false;
    }
    const Clock::time_point w1 = Clock::now();
    // per segment: the drafts the picks confirm, then how far the request goes (stop token, limits) within them
    const char* fin_of[MR] = {};
    auto fin_at = [&](const BSlot& sl, int b, int32_t y, int e) -> const char* {   // how the request ends at pick e
        const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
        return eos ? "stop" : sl.stop ? "cancel" : sl.produced + e + 1 >= sl.max_new ? "length"
             : sl.p + e + 2 > slot_cap[(size_t) b] ? "length" : nullptr;
    };
    for (int sg = 0; sg < S; ++sg) {
        const BSlot& sl = bs[(size_t) rows[sg]];
        int k = 1;
        while (k < lens[sg] && outb[r0s[sg] + k - 1] == drafts[sg][(size_t) k - 1]) ++k;
        for (int e = 0; e < k; ++e) {
            fin_of[sg] = fin_at(sl, rows[sg], outb[r0s[sg] + e], e);
            if (fin_of[sg] != nullptr) { k = e + 1; break; }
        }
        n_keep[sg] = k;
    }
    // grammar: a slot with a guard scans its kept picks; the first one the grammar refuses is drawn again
    // under the mask (the solo rule) and the segment ends at that row, which stays committed; with no token allowed
    // at all the request ends there as an error
    for (int sg = 0; sg < S; ++sg) {
        BSlot& sl = bs[(size_t) rows[sg]];
        if (sl.guard == nullptr || !sl.guard->on()) continue;
        const int miss = sl.guard->scan(outb + r0s[sg], n_keep[sg], 0);
        if (miss < 0) continue;
        int32_t gtok = 0;
        const basalt::program::grammar::Redraw verdict = sl.guard->redraw(
            {},
            [&](const std::vector<float>& bias, int32_t& t) { return ver.resample_row(r0s[sg] + miss, p0s[sg], bias, t, err); },
            [&](std::vector<float>& lg) {
                lg.resize((size_t) n_vocab);
                return ver.copy_logits(r0s[sg] + miss, lg.data());
            },
            gtok);
        if (verdict == basalt::program::grammar::Redraw::failed) {
            std::printf("ERR %s\n", err.empty() ? "the masked draw failed" : err.c_str());
            return false;
        }
        n_keep[sg] = miss + 1;
        if (verdict == basalt::program::grammar::Redraw::dead) {
            fin_of[sg] = "error";
            continue;
        }
        outb[r0s[sg] + miss] = gtok;
        fin_of[sg] = fin_at(sl, rows[sg], gtok, miss);
    }
    // logprobs: a slot that asked gets a BP line per kept pick, read before the commit and the drafts touch the head
    std::string lp_lines[MR];
    for (int sg = 0; sg < S; ++sg) {
        const int top = bs[(size_t) rows[sg]].lp_top;
        if (top < 0) continue;
        constexpr int LS = basalt::kernels::kLogprobStride;
        lh_buf.resize((size_t) n_keep[sg] * LS);
        if (!ver.row_logprobs(r0s[sg], n_keep[sg], outb + r0s[sg], top, lh_buf.data())) {
            std::printf("ERR logprobs: the head logits read failed\n");
            return false;
        }
        for (int e = 0; e < n_keep[sg]; ++e)
            lp_lines[sg] += "BP " + std::to_string(rows[sg]) +
                            basalt::program::serve::logprob_fields(lh_buf.data() + (size_t) e * LS, outb[r0s[sg] + e], top) + "\n";
    }
    if (logit_hash)
        for (int sg = 0; sg < S; ++sg)
            for (int e = 0; e < n_keep[sg]; ++e)
                basalt::log::raw("LH %d %lld %016llx\n", rows[sg], (long long) (bs[(size_t) rows[sg]].p + e),
                             (unsigned long long) row_hash(r0s[sg] + e));
    // the commit runs beside the drafts when nothing reads the committed state meanwhile
    const bool overlap = real && !bdigest;
    // a continuing slot's GDN state rows are replayed by its next window's recurrence instead of a pass of their own
    bool defer[MR] = {};
    for (int sg = 0; sg < S; ++sg) defer[sg] = fin_of[sg] == nullptr;
    if (!(overlap ? ver.commit_slots_begin(n_keep, defer, err) : ver.commit_slots(n_keep, err))) {
        std::printf("ERR %s\n", err.c_str());
        return false;
    }
    const Clock::time_point w2 = Clock::now();
    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
    bt_run += msd(w0, w1);
    bt_commit += msd(w1, w2);
    ++bt_windows;
    bt_rows += R;
    for (int sg = 0; sg < S; ++sg) {
        const int b = rows[sg];
        BSlot& sl = bs[(size_t) b];
        const int k = n_keep[sg];
        std::string bt = "BT " + std::to_string(b);   // one line per slot and window: every token it kept
        for (int e = 0; e < k; ++e) {
            sl.ids.push_back(tok[r0s[sg] + e]);    // the window fed it: the slot's sessions hold it now
            bt += " " + std::to_string((int) outb[r0s[sg] + e]);
        }
        basalt::protocol::emit_text(lp_lines[sg] + bt + "\n");
        sl.produced += k;
        if (real) {   // the window's verdict on its drafts: the draft-length rule, the lookup history, the policy
            const int a = k - 1, Tw = lens[sg];
            if (Tw > 1 && !sl.from_sfx) sl.cum.observe(sl.dprob, Tw - 1, a);
            if (Tw > 1) sl.acc = 0.8 * sl.acc + 0.2 * ((double) a / (double) (Tw - 1));
            sl.drafts_offered += Tw - 1;
            sl.drafts_accepted += a;
            if (sl.sfx) for (int e = 0; e < k; ++e) sl.sfx->append(outb[r0s[sg] + e]);
            if (sl.windows > 0 && fin_of[sg] == nullptr)
                bpolicy->observe(sl.from_sfx, Tw, a, sl.smatch, (msd(w0, w2) + bt_draft_last) * (double) Tw / (double) R);
            bt_slotwin += 1;
            bt_tokens += k;
            bt_dr_off += Tw - 1;
            bt_dr_acc += a;
            p0s[sg] = sl.p;
        }
        ++sl.windows;
        if (bdigest) {
            uint64_t part[10] = {};
            slot_digest(b, sl.p + k, part);
            std::printf("BDIG %d %lld", b, (long long) (sl.p + k));
            for (int i = 0; i < 10; ++i) std::printf("%c%llx", i == 0 ? ' ' : ',', (unsigned long long) part[i]);
            std::printf("\n");
        }
        if (fin_of[sg] != nullptr) {
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
            std::printf("BDONE %d %lld %s %.1f %lld %lld %lld\n", b, (long long) sl.produced, fin_of[sg], ms,
                        (long long) sl.drafts_accepted, (long long) sl.drafts_offered, (long long) sl.windows);
            sl.active = false;
            sl.cached = o.prompt_cache > 0;   // its sessions hold sl.ids for the next turn
            sl.idle_busy = busy_now();
            sl.guard.reset();
        } else {
            sl.x = outb[r0s[sg] + k - 1];
            sl.p += k;
        }
    }
    std::fflush(stdout);
    const Clock::time_point w3 = Clock::now();
    bt_emit += msd(w2, w3);
    if (real) {
        // the next windows' drafts, one slot at a time on the slot's own draft K/V.  The rows beyond one per
        // slot go to the slots most likely to use them; the others only advance their draft K/V.
        int cont[MR], nc = 0;
        for (int sg = 0; sg < S; ++sg) if (fin_of[sg] == nullptr) cont[nc++] = sg;
        int next_rows = row_budget(nc);
        const bool keep_learning = row_cost_window && nc == BatchRowCost::kSlots && steady_slots() && quiet();
        if (keep_learning) {
            // Drafting and the commit tail after the previous emission are part of this window's cost.
            int kept = 0;
            double weakest = 1.0, expected_extra = 0.0;
            for (int sg = 0; sg < S; ++sg) {
                const double q = bs[(size_t) rows[sg]].acc;
                weakest = std::min(weakest, q);
                expected_extra += q * q * (1.0 + q);
                kept += n_keep[sg];
            }
            const double verify_ms = msd(w0, w1);
            const double other_ms = msd(w1, w3) + batch_row_commit_tail;
            if (previous_budget == MB)
                batch_row_cost.observe(MB, R, kept, verify_ms, bt_draft_last, other_ms);
            next_rows = batch_row_cost.choose(weakest, expected_extra);
            batch_row_prepared = next_rows;
            if (batch_row_cost_trace)
                basalt::log::info("basalt batch rows", "budget=%d rows=%d offered=%d accepted=%d kept=%d verify_ms=%.3f "
                                  "draft_ms=%.3f other_ms=%.3f acceptance_min=%.3f next_budget=%d sampled=%d",
                                  MB, R, R - S, kept - S, kept, verify_ms, bt_draft_last, other_ms, weakest, next_rows,
                                  previous_budget == MB ? 1 : 0);
        } else if (row_cost_window) {
            batch_row_cost.reset();
        }
        if (batch_row_cost_trace && previous_budget != 0 && !keep_learning)
            basalt::log::info("basalt batch rows", "cohort interrupted, slots=%d continuing=%d prepared=%d next_budget=%d",
                              S, nc, previous_budget, next_rows);
        const int nfull = std::min(nc, next_rows - nc);
        sort_small(cont, nc, [&](int x1, int x2) {
            const BSlot& s1 = bs[(size_t) rows[x1]];
            const BSlot& s2 = bs[(size_t) rows[x2]];
            return s1.acc + 0.05 * (double) s1.since_draft > s2.acc + 0.05 * (double) s2.since_draft;
        });
        const float* Rw = mtp.keep_rows(ver.final_R_all(), R, err);
        if (Rw == nullptr) { std::printf("ERR %s\n", err.c_str()); return false; }
        const int64_t HCN = (int64_t) g.hc * g.n_embd;
        // a slot gets about its share of the next window's rows: drafts past it cost the drafter's GPU time and
        // are cut (at four slots of an 8-row budget, two drafts a slot); the verdicts stay exact
        mtp.set_step_cap(nc > 1 ? (next_rows + nc - 1) / nc : 0);
        mtp.set_draw(basalt::kernels::DraftDraw{});   // the one-sequence drafts below keep the argmax pick
        // the drafting slots in one pass (stores in place): every slot's catch-up in one forward and one full-layer row
        // per slot for each draft step, each row in its slot's own K/V and with its slot's own rule and cap
        // in groups of up to kDraftRowsMax slots whose window rows fit one catch-up forward (a window of the drafter's
        // width or less is one group); a slot left alone drafts on its own below
        int done = 0;
        while (mtp.seg_on() && done < nfull) {
            int nb = 0, rows_g = 0;
            while (done + nb < nfull && nb < (int) basalt::core::MtpDrafter::kDraftRowsMax &&
                   rows_g + lens[cont[done + nb]] <= mtp.batch_rows_max())
                rows_g += lens[cont[done + nb++]];
            if (nb < 2) break;
            const int* const gc = cont + done;   // this group's slots
            int ids[MR], r0[MR], Tn[MR], an[MR];
            int64_t pn[MR];
            float th[MR];
            const int32_t* tk[MR];
            int32_t* dr[MR];
            float* pb[MR];
            int nd[MR];
            // speculative sampling: a sampled slot (narrow list, no fixed seed, no penalties or grammar) draws its drafts
            basalt::kernels::DraftDraw dd[MR] = {};
            int32_t* qi[MR] = {};
            float* qp[MR] = {};
            constexpr int Q = basalt::kernels::kDraftDrawQ;
            for (int ci = 0; ci < nb; ++ci) {
                BSlot& sl = bs[(size_t) rows[gc[ci]]];
                const basalt::kernels::SamplerParams* ssp = ver.slot_sampling(rows[gc[ci]]);
                const bool draw = ssp != nullptr && !ssp->greedy && ssp->temperature > 0.0f && ssp->draw_drafts &&
                                  sl.pen_n == 0 && (sl.guard == nullptr || !sl.guard->on()) && mtp.can_draw() &&
                                  basalt::kernels::sampler_list_len(ssp->top_k, (int) n_vocab) <= basalt::kernels::kSamplerNarrowMax;
                if (draw) {
                    dd[ci].on = 1; dd[ci].temperature = ssp->temperature; dd[ci].top_k = ssp->top_k;
                    dd[ci].top_p = ssp->top_p; dd[ci].seed = ssp->seed;
                    sl.qids.resize((size_t) basalt::spec::DraftPolicy::kMaxT * Q);
                    sl.qp.resize((size_t) basalt::spec::DraftPolicy::kMaxT * Q);
                    qi[ci] = sl.qids.data();
                    qp[ci] = sl.qp.data();
                }
                sl.drawn = draw;
            }
            mtp.set_draw_rows(dd, nb);
            for (int ci = 0; ci < nb; ++ci) {
                const int sg = gc[ci];
                BSlot& sl = bs[(size_t) rows[sg]];
                ids[ci] = rows[sg]; r0[ci] = r0s[sg]; Tn[ci] = lens[sg]; an[ci] = n_keep[sg] - 1; pn[ci] = p0s[sg];
                th[ci] = sl.cum.th(); tk[ci] = outb + r0s[sg]; dr[ci] = sl.dr; pb[ci] = sl.dprob;
            }
            if (!mtp.draft_batch(nb, ids, Rw, r0, Tn, tk, pn, an, th, dr, pb, nd, err, qi, qp)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            for (int ci = 0; ci < nb; ++ci) {
                BSlot& sl = bs[(size_t) ids[ci]];
                sl.n_dr = nd[ci];
                sl.since_draft = 0;
            }
            done += nb;
        }
        for (int ci = done; ci < nc; ++ci) {
            const int sg = cont[ci];
            const int b = rows[sg];
            BSlot& sl = bs[(size_t) b];
            const int Tw = lens[sg];
            const int a = n_keep[sg] - 1;
            mtp.set_ple_session(bslot_ss[0][(size_t) b].get());
            sl.n_dr = 0;
            sl.drawn = false;   // a one-sequence draft here keeps the argmax pick
            if (ci < nfull) {
                mtp.set_cum_th(sl.cum.th());
                int nd = 0;
                if (!mtp.draft_seq(b, Rw + (size_t) r0s[sg] * HCN, Tw, outb + r0s[sg], p0s[sg], a, sl.dr, err, sl.dprob,
                                   &nd)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                sl.n_dr = nd;
                sl.since_draft = 0;
            } else {
                ++sl.since_draft;
                if (!mtp.defer_rows(b, Rw + (size_t) r0s[sg] * HCN, outb + r0s[sg], n_keep[sg], p0s[sg], err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
            }
        }
        mtp.set_step_cap(0);
        mtp.set_ple_session(rsp);
        if (!mtp.idle(err)) { std::printf("ERR %s\n", err.c_str()); return false; }   // (the kept rows are copied)
        bt_draft += msd(w3, Clock::now());
        bt_draft_last = msd(w3, Clock::now());
    }
    batch_row_commit_tail = 0.0;
    if (overlap) {
        const Clock::time_point c0 = Clock::now();
        if (!ver.commit_slots_end(err)) {
            std::printf("ERR %s\n", err.c_str());
            return false;
        }
        batch_row_commit_tail = msd(c0, Clock::now());
        bt_commit += batch_row_commit_tail;
    }
    basalt::core::progress().busy.store(was_busy);
    if (!batch_on() && bt_windows > 0) {
        const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
        const double L = (double) g.n_layers;
        basalt::log::info("basalt batch", "%lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach "
                             "wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f; per layer-window: CPU experts "
                             "%.2f, VRAM hits %.2f, PCIe %.2f; %.1f rows/s over %.0f ms of wall time (admissions "
                             "included)",
                     (long long) bt_windows, bt_rows / w, (bt_run + bt_commit + bt_emit + bt_draft) / w, bt_run / w,
                     (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                     (drive.d.multi_misses - bt_miss0) / (w * L), (drive.d.cache_hits - bt_hits0) / (w * L),
                     (drive.d.pcie_experts - bt_pcie0) / (w * L), 1000.0 * bt_rows / std::max(wall, 1e-9), wall);
        if (drive.d.offload_entries > bt_offload0)
            basalt::log::info("basalt batch", "GPU expert entries outside the device cache: %lld",
                              (long long) (drive.d.offload_entries - bt_offload0));
        if (drive.d.count_cpu_group_sizes) {
            std::string counts;
            for (size_t i = 1; i < drive.d.cpu_group_sizes.size(); ++i)
                if (drive.d.cpu_group_sizes[i] != 0)
                    counts += " " + std::to_string(i) + "=" + std::to_string(drive.d.cpu_group_sizes[i]);
            basalt::log::info("basalt batch", "CPU expert groups by token count:%s", counts.empty() ? " none" : counts.c_str());
            drive.d.cpu_group_sizes.fill(0);
        }
        if (batch_adapt) {
            basalt::log::info("basalt batch adapt", "%lld rounds, %lld swaps queued (%.1f per round), %lld deferred "
                                 "(no free RAM place), %.3f ms/window of host time, peer swaps total %lld",
                         (long long) ba_rounds, (long long) ba_swaps,
                         ba_rounds > 0 ? (double) ba_swaps / (double) ba_rounds : 0.0,
                         (long long) ba_noplace, ba_ms / w,
                         (long long) (peer.valid() ? peer.swaps() : 0));
            ba_swaps = ba_rounds = ba_swaps_l = ba_rounds_l = ba_noplace = 0;
            ba_ms = 0;
        }
        if (bt_slotwin > 0)
            basalt::log::info("basalt batch spec", "%.2f tokens/window/slot (%.2f slots/window, %.2f rows/window), "
                                 "drafts %lld accepted of %lld, draft %.2f ms/window (%.2f ms/slot-window), %.1f "
                                 "tokens/s", (double) bt_tokens / (double) bt_slotwin, (double) bt_slotwin / w,
                         bt_rows / w, (long long) bt_dr_acc, (long long) bt_dr_off, bt_draft / w,
                         bt_draft / (double) bt_slotwin, 1000.0 * (double) bt_tokens / std::max(wall, 1e-9));
        for (size_t k = 0; k <= stages.size(); ++k) {
            const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
            if (!pr.empty()) basalt::log::raw("basalt batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
        }
        bt_run = bt_commit = bt_emit = bt_draft = 0;
        bt_windows = bt_rows = bt_tokens = bt_slotwin = bt_dr_off = bt_dr_acc = 0;
    }
    return true;
}

int QfnRuntime::init_batch() {
    bs = std::vector<BSlot>((size_t) std::max(o.batch, 0));
    drive.d.count_cpu_group_sizes = basalt::core::exp_flag("n1x_cpu_group_stats") != 0;
    // the real MTP drafts in the batch windows (a draft K/V per slot); the scripted test drive keeps its own
    // Benchmark uncached batches with the same model allocations and solo drafter in both arms.
    batch_spec = mtp_on && mtp.main_store() > 0 && o.batch > 0 &&
                 (o.prompt_cache > 0 || basalt::core::exp_flag("n1x_batch_draft", 1) != 0);
    batch_row_cost_enabled = batch_spec && o.batch == BatchRowCost::kSlots && ver.max_t() >= BatchRowCost::kWide &&
                             S_mtp >= BatchRowCost::kWide / BatchRowCost::kSlots && !multi_gpu && !piped && !peer.valid() &&
                             xpeers.empty() && basalt::core::exp_flag("batch_rows", 0) < 1.0 &&
                             basalt::core::device_memory_model().integrated;
    batch_row_cost_trace = batch_row_cost_enabled && basalt::core::exp_flag("batch_row_cost_trace") != 0;
    batch_rows_value = batch_spec && basalt::core::exp_flag("batch_rows_value", 0) != 0;
    batch_rows_value_trace = batch_rows_value && basalt::core::exp_flag("batch_rows_value_trace", 0) != 0;
    brv_windows = 0;
    brv_even = brv_value = 0.0;
    S_all = S;
    bpolicy.emplace(S, 0.03, false);   // the lookup-or-MTP choice, costs per row
    bt_draft = 0;
    bt_draft_last = 0;
    bt_slotwin = 0;
    bt_dr_off = 0;
    bt_dr_acc = 0;
    bscript = std::vector<std::pair<std::vector<int32_t>, uint64_t>>((size_t) std::max(o.batch, 0));
    bdigest = basalt::core::exp_flag("concur5_digest", 0) != 0;
    // concur9_logit_hash: a hash of the head logits of every accepted row, on stderr (solo and batch alike)
    logit_hash = basalt::core::exp_flag("concur9_logit_hash", 0) != 0;

    // adaptive tier rounds in the batch windows (routing counts from every row of every slot)
    batch_adapt = adapt_fast && !drive.d.usage.empty();
    ba_win = 0;
    ba_swaps = 0;
    ba_rounds = 0;
    ba_swaps_l = 0;
    ba_rounds_l = 0;
    ba_h_l = 0;
    ba_m_l = 0;
    ba_noplace = 0;
    ba_ms = 0;
    ba_job = false;
    // timing of the batch windows since the slots were last all idle (one stderr line then)
    bt_run = 0;
    bt_commit = 0;
    bt_emit = 0;
    bt_wait0 = 0;
    bt_pool0 = 0;
    bt_miss0 = 0;
    bt_hits0 = 0;
    bt_pcie0 = 0;
    bt_windows = 0;
    bt_rows = 0;
    bt_tokens = 0;
    bt_start = Clock::now();
    admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
    solo_slot = -1;                ///< SGEN: the slot a solo request runs in, on that slot's own state
    admit_max_new = 0;

    // a prompt read on its own thread beside the slots' windows: the tier stays as it is meanwhile
    ov_active = false;
    ov_post_done = false;

    // ---- --batch-groups: the slot groups pipelined through the stages (--batch-groups G > 1 with a layer split)
    n_pipe = (int) stages.size() + 1;
    piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1;
    GS = piped ? o.batch / o.batch_groups : o.batch;

    pg = std::vector<PGroup>((size_t) (piped ? o.batch_groups : 0));
    stage_group = std::vector<int>((size_t) n_pipe, -1);
    pipe_tick = 0;
    rr = 0;
    return 0;
}

}  // namespace basalt::models::qwen4exp::serve
