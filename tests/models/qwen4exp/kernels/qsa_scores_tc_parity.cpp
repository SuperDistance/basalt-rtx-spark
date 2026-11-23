// tests/models/qwen4exp/kernels/qsa_scores_tc_parity.cpp - the prompt path's block scores on tensor cores
// (qsa_block_scores_tc) at the real indexer geometry: scores against a double-precision CPU reference (every query of
// the short cases, a sample of the long one), an FNV-1a hash of every score the kernel writes (the same across builds
// when a change is meant to be bitwise-neutral), and a microbench line at a long context.
//   qsa_scores_tc_parity [iters]
#include "basalt/models/qwen4exp/kernels/qsa.hpp"
#include "basalt/models/qwen4exp/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Case {
    const char* name;
    int64_t nq, pos0, max_blocks;
    int sample;   // queries checked against the CPU reference (0: all)
};

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("qsa_scores_tc_parity: no CUDA device\n");
        return 77;
    }
    const k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.idx_n_head, D = (int) s.idx_dim, R = (int) s.idx_block;
    const Case cases[] = {{"short", 37, 1000, 400, 0}, {"mid", 300, 5003, 1400, 0}, {"long", 8192, 60000, 17100, 48}};
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    int bad = 0;
    for (const Case& c : cases) {
        std::mt19937 rng(17 + (unsigned) c.nq);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> pooled((size_t) c.max_blocks * D), q((size_t) c.nq * H * D), dead((size_t) D);
        for (float& x : pooled) x = nd(rng);
        for (float& x : q) x = nd(rng) * 0.1f;
        for (float& x : dead) x = nd(rng);
        std::vector<int32_t> steps((size_t) c.nq * k::kStepCount);
        int64_t active = 0;
        for (int64_t i = 0; i < c.nq; ++i) {
            const int64_t pos = c.pos0 + i, n_kv = pos + 1, n_bid = n_kv / R;
            steps[(size_t) i * k::kStepCount + k::kStepPos] = (int32_t) pos;
            steps[(size_t) i * k::kStepCount + k::kStepNKv] = (int32_t) n_kv;
            steps[(size_t) i * k::kStepCount + k::kStepNBid] = (int32_t) n_bid;
            steps[(size_t) i * k::kStepCount + k::kStepWidth] = 0;
            active = std::max(active, n_bid + 1);
        }
        if (active > c.max_blocks) { std::printf("%s: the case outgrows max_blocks\n", c.name); return 2; }
        float *d_pooled, *d_q, *d_dead, *d_out;
        int32_t* d_steps;
        ck(cudaMalloc(&d_pooled, pooled.size() * 4), "pooled");
        ck(cudaMalloc(&d_q, q.size() * 4), "q");
        ck(cudaMalloc(&d_dead, dead.size() * 4), "dead");
        ck(cudaMalloc(&d_steps, steps.size() * 4), "steps");
        ck(cudaMalloc(&d_out, (size_t) c.nq * c.max_blocks * 4), "out");
        ck(cudaMemcpy(d_pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "pooled up");
        ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q up");
        ck(cudaMemcpy(d_dead, dead.data(), dead.size() * 4, cudaMemcpyHostToDevice), "dead up");
        ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps up");
        ck(cudaMemset(d_out, 0xFF, (size_t) c.nq * c.max_blocks * 4), "out clear");
        if (!k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, c.nq, c.max_blocks, s, d_out, st, active)) {
            std::printf("%s: qsa_block_scores_tc refused the real geometry\n", c.name);
            return 1;
        }
        ck(cudaStreamSynchronize(st), "scores");
        std::vector<float> out((size_t) c.nq * c.max_blocks);
        ck(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost), "out down");
        // the hash of every written score (blocks 0..n_bid of each query), and the reference on the checked queries
        uint64_t h = 1469598103934665603ull;
        double max_abs = 0.0, max_rel = 0.0;
        int64_t unwritten = 0, checked = 0, bonus_bad = 0;
        const int64_t step = c.sample > 0 ? std::max<int64_t>(1, c.nq / c.sample) : 1;
        for (int64_t i = 0; i < c.nq; ++i) {
            const int64_t n_bid = steps[(size_t) i * k::kStepCount + k::kStepNBid];
            const float* row = out.data() + (size_t) i * c.max_blocks;
            for (int64_t b = 0; b <= n_bid && b < c.max_blocks; ++b) {
                uint32_t bits;
                std::memcpy(&bits, &row[b], 4);
                if (bits == 0xFFFFFFFFu) ++unwritten;
                for (int j = 0; j < 4; ++j) { h ^= (bits >> (8 * j)) & 0xFF; h *= 1099511628211ull; }
            }
            if (i % step != 0) continue;
            ++checked;
            const float* qi = q.data() + (size_t) i * H * D;
            for (int64_t b = 0; b <= n_bid && b < c.max_blocks; ++b) {
                const float* kb = b < n_bid ? pooled.data() + (size_t) b * D : dead.data();
                double ref = 0.0, mag = 0.0;
                for (int hh = 0; hh < H; ++hh) {
                    double d = 0.0, m = 0.0;
                    for (int x = 0; x < D; ++x) {
                        d += (double) qi[hh * D + x] * (double) kb[x];
                        m += std::fabs((double) qi[hh * D + x] * (double) kb[x]);
                    }
                    ref += d > 0.0 ? d : 0.0;
                    mag += m;
                }
                // the tail block of a query whose last block is incomplete carries the kernel's +1e9 (always selected)
                const int64_t n_kv = steps[(size_t) i * k::kStepCount + k::kStepNKv];
                if (b == n_bid && n_kv % R != 0) {
                    if (std::fabs((double) row[b] - (double) ((float) ref + 1e9f)) > 128.0) ++bonus_bad;
                    continue;
                }
                const double err = std::fabs((double) row[b] - ref);
                max_abs = std::max(max_abs, err);
                max_rel = std::max(max_rel, err / std::max(mag, 1e-30));
            }
        }
        // FP32-level: an error of a few ulps of the summed magnitudes
        const bool ok = unwritten == 0 && bonus_bad == 0 && max_rel < 1e-6;
        std::printf("%-5s %5lld queries at %6lld.., %6lld blocks: max abs %.3e, max rel (of sum |q k|) %.3e over %lld "
                    "queries, unwritten %lld, hash %016llx%s\n",
                    c.name, (long long) c.nq, (long long) c.pos0, (long long) active, max_abs, max_rel,
                    (long long) checked, (long long) unwritten, (unsigned long long) h, ok ? "" : "  FAIL");
        bad += ok ? 0 : 1;
        if (c.sample > 0 && iters > 0) {   // the microbench: the long case, keys warm in L2 after the first call
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "event");
            ck(cudaEventCreate(&e1), "event");
            float best = 1e30f, tot = 0.0f;
            for (int it = 0; it < iters; ++it) {
                ck(cudaEventRecord(e0, st), "record");
                k::qsa_block_scores_tc(d_pooled, d_dead, d_q, d_steps, c.nq, c.max_blocks, s, d_out, st, active);
                ck(cudaEventRecord(e1, st), "record");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms);
                tot += ms;
            }
            double pairs = 0.0;
            for (int64_t i = 0; i < c.nq; ++i) pairs += (double) steps[(size_t) i * k::kStepCount + k::kStepNBid];
            const double flop = pairs * H * D * 2.0;
            std::printf("bench long: %.3f ms/call (best %.3f), %.1f TFLOP/s of scores (%.1f with the three TF32 "
                        "products)\n", tot / iters, best, flop / (best * 1e9), 3.0 * flop / (best * 1e9));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        cudaFree(d_pooled); cudaFree(d_q); cudaFree(d_dead); cudaFree(d_steps); cudaFree(d_out);
    }
    cudaStreamDestroy(st);
    std::printf("qsa_scores_tc_parity %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
