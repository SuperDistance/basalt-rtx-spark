// tests/models/qwen4exp/kernels/dense_tc_parity.cpp - dense_tc (DEC_DTC) against native_mmvq, and its column contract.
//
//     build/dense_tc_parity [--bench] [--hashes]
//
// Random GGUF blocks (fp16 scales rewritten as normal values) and normal activations, one q8_1 quantization shared
// by both paths, at the model's projection shapes.  Checks:
//   - every column j of a T-column dense_tc call (T = 1..16) is bitwise equal to column j of the 8-column call on
//     columns 0..7 (j < 8) or column j - 8 of the 8-column call on columns 8..15 (two n-tiles: the second n-tile's
//     columns are the first's arithmetic);
//   - the outputs of the calls of T = 1..8 hash to the frozen values of the one-n-tile kernels (FROZEN);
//   - the error against native_mmvq's single-column outputs (the current numerics): max |diff| / max |ref| and the
//     RMS of the diff over the RMS of ref, per case; it fails above 1e-3 RMS.
// --hashes: only T = 1..8, the hashes printed (a FROZEN table for a build that changes the numerics on purpose).
// --bench: native_mmvq (T <= 8) and dense_tc timed per T as CUDA graphs over enough weight copies that the L2 holds
// none (launched as programmatic dependents under PDL=1).
#include "basalt/models/qwen4exp/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char* name;
    int type, n_in, n_out;
    int block_bytes;
    int scale_at[2];
};

const Case CASES[] = {
    {"qkv Q6_K", 14, 2560, 10240, 210, {208, -1}},
    {"qkv Q4_K", 12, 2560, 10240, 144, {0, 2}},
    {"qkv Q5_K", 13, 2560, 10240, 176, {0, 2}},
    {"qkv IQ4_XS", 23, 2560, 10240, 136, {0, -1}},
    {"z Q6_K", 14, 2560, 6144, 210, {208, -1}},
    {"z IQ4_XS", 23, 2560, 6144, 136, {0, -1}},
    {"z IQ3_S", 21, 2560, 6144, 110, {0, -1}},
    {"out Q6_K", 14, 6144, 2560, 210, {208, -1}},
    {"out IQ4_XS", 23, 6144, 2560, 136, {0, -1}},
    {"out Q4_K", 12, 6144, 2560, 144, {0, 2}},
    {"out Q5_K", 13, 6144, 2560, 176, {0, 2}},
    {"out IQ3_S", 21, 6144, 2560, 110, {0, -1}},
    {"q Q6_K", 14, 2560, 12288, 210, {208, -1}},
    {"kv Q6_K", 14, 2560, 512, 210, {208, -1}},
    {"tail Q4_K", 12, 1024, 1000, 144, {0, 2}},
    {"tail Q6_K", 14, 1536, 999, 210, {208, -1}},
    {"qkv Q3_K", 11, 2560, 10240, 110, {108, -1}},
    {"z Q3_K", 11, 2560, 6144, 110, {108, -1}},
    {"out Q3_K", 11, 6144, 2560, 110, {108, -1}},
    {"q Q3_K", 11, 2560, 12288, 110, {108, -1}},
    {"qkv Q2_0", 42, 2560, 10240, 18, {0, -1}},
    {"z Q2_0", 42, 2560, 6144, 18, {0, -1}},
    {"q Q2_0", 42, 2560, 12288, 18, {0, -1}},
    {"tail Q3_K", 11, 1536, 999, 110, {108, -1}},
    {"tail Q2_0", 42, 1024, 1000, 18, {0, -1}},
    {"qkv Q8_0", 8, 2560, 10240, 34, {0, -1}},
    {"z Q8_0", 8, 2560, 6144, 34, {0, -1}},
    {"out Q8_0", 8, 6144, 2560, 34, {0, -1}},
    {"q Q8_0", 8, 2560, 12288, 34, {0, -1}},
    {"kv Q8_0", 8, 2560, 512, 34, {0, -1}},
    {"tail Q8_0", 8, 1024, 1000, 34, {0, -1}},
    {"tail2 Q8_0", 8, 1536, 999, 34, {0, -1}},
    {"kv IQ3_S", 21, 6144, 512, 110, {0, -1}},
};

constexpr int TM = 16;   // columns of the widest call
constexpr int T8 = 8;    // columns of one n-tile
bool g_hashes = false;   // --hashes

// FNV-1a over the calls of T = 1..8 (each call's T columns), per case; pairs: both outputs, by pair index: the
// one-n-tile kernels' outputs (--hashes).
struct Frozen {
    const char* name;
    unsigned long long h;
};
const Frozen FROZEN[] = {
    {"qkv Q6_K 2560x10240", 0x080622278ab7b3e2ull},
    {"qkv Q4_K 2560x10240", 0xb516b3ad54dd78adull},
    {"qkv Q5_K 2560x10240", 0x74fcd82ba45550fcull},
    {"qkv IQ4_XS 2560x10240", 0x3cbc2dd5b289edebull},
    {"z Q6_K 2560x6144", 0x914b7fdc46e912a2ull},
    {"z IQ4_XS 2560x6144", 0xbbe2884cd58c550eull},
    {"out Q6_K 6144x2560", 0x00721259607dd60eull},
    {"out IQ4_XS 6144x2560", 0xf5510b3f7fdc8ff4ull},
    {"out Q4_K 6144x2560", 0x4bc9c7c8317208ebull},
    {"out Q5_K 6144x2560", 0xfb9c81390549316cull},
    {"q Q6_K 2560x12288", 0x5ec6e83a4d182889ull},
    {"kv Q6_K 2560x512", 0x66e1752753b95663ull},
    {"tail Q4_K 1024x1000", 0x021dc2d272920f48ull},
    {"tail Q6_K 1536x999", 0x49c0e89e1df7bfe7ull},
    {"qkv Q3_K 2560x10240", 0xd99f5b7e8ca9c3deull},
    {"z Q3_K 2560x6144", 0xcb93261305173315ull},
    {"out Q3_K 6144x2560", 0x9e7c8d946ef5860full},
    {"q Q3_K 2560x12288", 0xd92ab00e3544a943ull},
    {"qkv Q2_0 2560x10240", 0x58d1687ffb82526full},
    {"z Q2_0 2560x6144", 0xa38551cfba548631ull},
    {"q Q2_0 2560x12288", 0xe3036e0bbbec5873ull},
    {"tail Q3_K 1536x999", 0x95508f019f4650abull},
    {"tail Q2_0 1024x1000", 0xab429d9e0e14f485ull},
    {"qkv Q8_0 2560x10240", 0xd5f267eb52ab25f6ull},
    {"z Q8_0 2560x6144", 0x4a10d2b61e2bd1a9ull},
    {"out Q8_0 6144x2560", 0x55127d3742cd8df4ull},
    {"q Q8_0 2560x12288", 0xa334ecca3f25c7a5ull},
    {"kv Q8_0 2560x512", 0x1f14c0c68b43505full},
    {"tail Q8_0 1024x1000", 0x9509f8dae4a8f0c1ull},
    {"tail2 Q8_0 1536x999", 0x46c482ee11599726ull},
    {"kv IQ3_S 6144x512", 0x5663803af929fa5cull},
    {"pair 0", 0x3fb30d9f4984218aull},
    {"pair 1", 0x835fe2ffdaa1741full},
    {"pair 2", 0x2c9ee302f527432aull},
    {"pair 3", 0x9879661c8bd4cc00ull},
    {"pair 4", 0x165baec6b91af9ceull},
    {"pair 5", 0x2ed310f6839d0813ull},
    {"pair 6", 0xbc063b3ac5c510a1ull},
    {"pair 7", 0xc1a03130d0baa7c5ull},
    {"pair 8", 0x85b4009e5d321ad9ull},
    {"pair 9", 0xe1459053a7d07db2ull},
    {"pair 10", 0x4337ae3083697b24ull},
    {"pair 11", 0x17f9692090673873ull},
    {"pair 12", 0x006116a248b16838ull},
    {"pair 13", 0x30af01053efe1613ull},
    {"", 0},
};
unsigned long long frozen_of(const std::string& name) {
    for (const Frozen& f : FROZEN)
        if (name == f.name) return f.h;
    return 0;
}
void fnv(unsigned long long& h, const float* p, std::size_t n) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    for (std::size_t i = 0; i < n * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
}
// "same" / "DIFFERS" / "none frozen" against the frozen hash
const char* frozen_check(const std::string& name, unsigned long long h, bool& ok) {
    const unsigned long long f = frozen_of(name);
    if (f == 0) return "none frozen";
    if (f != h) ok = false;
    return f == h ? "same" : "DIFFERS";
}

// q8_1 columns of x, 8 at a time (native_quantize_q8_1 takes at most 8; every block is its column's alone)
void quantize_cols(const float* x, void* xq, int n_in, int ncols, cudaStream_t s) {
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(n_in, 1);
    for (int t0 = 0; t0 < ncols; t0 += T8)
        basalt::kernels::native_quantize_q8_1(x + (std::size_t) t0 * n_in, (uint8_t*) xq + t0 * qcol, n_in,
                                              std::min(T8, ncols - t0), s);
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

std::vector<uint8_t> make_weights(const Case& c, std::mt19937& rng) {
    const std::size_t wbytes = basalt::kernels::native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t nb = wbytes / (std::size_t) c.block_bytes;
    for (std::size_t k = 0; k < nb; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    return w;
}

// float64 reference for Q8_0: sum over blocks of d_w * d_x * (exact integer dot), from the same q8_1 columns
std::vector<double> cpu_q8_0(const Case& c, const std::vector<uint8_t>& w, const std::vector<uint8_t>& xq, int cols) {
    const int nb = c.n_in / 32;
    const std::size_t qcol = (std::size_t) nb * 36;
    auto h2f = [](uint16_t h) {
        const int e = (h >> 10) & 31, m = h & 1023;
        const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (1024 + m), e - 25);
        return (h >> 15) ? -v : v;
    };
    std::vector<double> y((std::size_t) cols * c.n_out);
    for (int j = 0; j < cols; ++j)
        for (int r = 0; r < c.n_out; ++r) {
            double sum = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* wb = &w[((std::size_t) r * nb + b) * 34];
                const uint8_t* xb = &xq[j * qcol + (std::size_t) b * 36];
                uint16_t dwh, dxh;
                std::memcpy(&dwh, wb, 2);
                std::memcpy(&dxh, xb, 2);
                long dot = 0;
                for (int k = 0; k < 32; ++k) dot += (long) (int8_t) wb[2 + k] * (int8_t) xb[4 + k];
                sum += h2f(dwh) * h2f(dxh) * (double) dot;
            }
            y[(std::size_t) j * c.n_out + r] = sum;
        }
    return y;
}

bool parity(const Case& c, cudaStream_t s) {
    std::mt19937 rng(777u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    const std::vector<uint8_t> w = make_weights(c, rng);
    std::vector<float> x((std::size_t) TM * c.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);   // columns 0..7 are the values an 8-column run of this test draws first
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(c.n_in, 1);
    void *dw = nullptr, *xq = nullptr;
    float *dx = nullptr, *yref = nullptr, *yt = nullptr;
    const std::size_t ybytes = (std::size_t) TM * c.n_out * 4;
    if (!ck(cudaMalloc(&dw, w.size()), "malloc") || !ck(cudaMalloc(&xq, qcol * TM), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * 4), "malloc") || !ck(cudaMalloc(&yref, ybytes), "malloc") ||
        !ck(cudaMalloc(&yt, ybytes * (TM + 2)), "malloc"))
        return false;
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    // IQ3_S has only the one-tile path; the engine takes it where dense_tc_supported says so, so does the test
    if (c.type == 21 && !basalt::kernels::dense_tc_supported(c.type, c.n_in, c.n_out, dw)) {
        std::printf("%-12s %5d x %5d  skipped: shape outside the one-tile path on this device\n", c.name, c.n_in, c.n_out);
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yt);
        return true;
    }
    cudaDeviceSynchronize();   // pageable copies may still be in flight for a non-blocking stream
    quantize_cols(dx, xq, c.n_in, TM, s);
    for (int j = 0; j < TM; ++j)
        basalt::kernels::native_mmvq(c.type, dw, (const uint8_t*) xq + j * qcol, yref + (std::size_t) j * c.n_out,
                                     c.n_in, c.n_out, 1, s);
    // slot T (1..16): the T-column call; slot 0: the 8-column call on columns 0..7, then on columns 8..15
    const int tmax = g_hashes ? T8 : TM;
    float* const y8 = yt + (std::size_t) (TM + 1) * TM * c.n_out;   // the two 8-column calls' outputs
    cudaMemsetAsync(yt, 0xff, ybytes * (TM + 2), s);
    for (int T = 1; T <= tmax; ++T)
        basalt::kernels::dense_tc(c.type, dw, xq, yt + (std::size_t) T * TM * c.n_out, c.n_in, c.n_out, T, s);
    basalt::kernels::dense_tc(c.type, dw, xq, y8, c.n_in, c.n_out, T8, s);
    basalt::kernels::dense_tc(c.type, dw, (const uint8_t*) xq + T8 * qcol, y8 + (std::size_t) T8 * c.n_out, c.n_in,
                              c.n_out, T8, s);
    if (!ck(cudaStreamSynchronize(s), "run")) return false;
    std::vector<float> ref((std::size_t) TM * c.n_out), got((std::size_t) (TM + 2) * TM * c.n_out);
    cudaMemcpy(ref.data(), yref, ybytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(got.data(), yt, got.size() * 4, cudaMemcpyDeviceToHost);
    long long bits = 0, nonfinite = 0;
    const float* g8 = got.data() + (std::size_t) (TM + 1) * TM * c.n_out;   // column j of the two 8-column calls
    unsigned long long h = 1469598103934665603ull;
    for (int T = 1; T <= tmax; ++T) {
        if (T <= T8) fnv(h, &got[(std::size_t) T * TM * c.n_out], (std::size_t) T * c.n_out);
        for (int j = 0; j < T; ++j)
            for (int r = 0; r < c.n_out; ++r) {
                const float a = got[((std::size_t) T * TM + j) * c.n_out + r], b = g8[(std::size_t) j * c.n_out + r];
                if (std::memcmp(&a, &b, 4) != 0) ++bits;
                if (!std::isfinite(a)) ++nonfinite;
            }
    }
    const int ncmp = tmax;   // the columns compared with native_mmvq / float64: the widest call's
    const float* gw = got.data() + (std::size_t) tmax * TM * c.n_out;
    double maxd = 0, maxr = 0, sd = 0, sr = 0;
    for (std::size_t i = 0; i < (std::size_t) ncmp * c.n_out; ++i) {
        const double d = std::fabs((double) gw[i] - (double) ref[i]);
        maxd = std::max(maxd, d);
        maxr = std::max(maxr, std::fabs((double) ref[i]));
        sd += d * d;
        sr += (double) ref[i] * ref[i];
    }
    const double rms = std::sqrt(sd / std::max(sr, 1e-300));
    bool ok = bits == 0 && nonfinite == 0 && rms < 1e-3;
    if (c.type == 8) {   // against float64, for every T's columns; native_mmvq's own error alongside
        std::vector<uint8_t> xh(qcol * TM);
        cudaMemcpy(xh.data(), xq, xh.size(), cudaMemcpyDeviceToHost);
        const std::vector<double> ex = cpu_q8_0(c, w, xh, ncmp);
        double dd = 0, dn = 0, rr = 0, mx = 0;
        for (std::size_t i = 0; i < ex.size(); ++i) {
            dd += (gw[i] - ex[i]) * (gw[i] - ex[i]);
            dn += (ref[i] - ex[i]) * (ref[i] - ex[i]);
            rr += ex[i] * ex[i];
            mx = std::max(mx, std::fabs(ex[i]));
        }
        const double rd = std::sqrt(dd / std::max(rr, 1e-300)), rn = std::sqrt(dn / std::max(rr, 1e-300));
        double worst = 0;   // every T's column j against float64 too
        for (int T = 1; T <= tmax; ++T)
            for (int j = 0; j < T; ++j)
                for (int r = 0; r < c.n_out; ++r)
                    worst = std::max(worst, std::fabs(got[((std::size_t) T * TM + j) * c.n_out + r] -
                                                      ex[(std::size_t) j * c.n_out + r]));
        const bool k = rd < 1e-5 && worst < 1e-4 * mx;
        std::printf("   %s float64: dense_tc rms %.3e  native_mmvq rms %.3e  worst |d| over all T %.3e  %s\n", c.name, rd,
                    rn, worst, k ? "PASS" : "FAIL");
        ok = ok && k;
    }
    const std::string key = std::string(c.name) + " " + std::to_string(c.n_in) + "x" + std::to_string(c.n_out);
    const char* fz = g_hashes ? "-" : frozen_check(key, h, ok);
    if (g_hashes) std::printf("    {\"%s\", 0x%016llxull},\n", key.c_str(), h);
    std::printf("%-12s %5d x %5d  T 1..%d columns bitwise: %s (%lld diff, %lld nonfinite)  T 1..8 hash %016llx (%s)  "
                "vs native: max|d|/max|ref| %.3e  rms %.3e  %s\n",
                c.name, c.n_in, c.n_out, tmax, bits == 0 ? "yes" : "NO", bits, nonfinite, h, fz,
                maxd / std::max(maxr, 1e-30), rms, ok ? "PASS" : "FAIL");
    cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(yref); cudaFree(yt);
    return ok;
}

// dense_tc_pair of a qkv and a z case: each output's T columns (T = 1..16) bitwise equal to the 8-column calls on
// columns 0..7 and 8..15, T = 1..8 against the frozen hash, and its error against native_mmvq
bool pair_parity(const Case& a, const Case& b, int index, cudaStream_t s) {
    std::mt19937 rng(4242u + (unsigned) a.type * 7u + (unsigned) b.type);
    const std::vector<uint8_t> wa = make_weights(a, rng), wb = make_weights(b, rng);
    std::vector<float> x((std::size_t) TM * a.n_in);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const std::size_t qcol = basalt::kernels::native_q8_1_bytes(a.n_in, 1);
    void *dwa = nullptr, *dwb = nullptr, *xq = nullptr;
    float *dx = nullptr, *ra = nullptr, *rb = nullptr, *ya = nullptr, *yb = nullptr;
    cudaMalloc(&dwa, wa.size()); cudaMalloc(&dwb, wb.size()); cudaMalloc(&xq, qcol * TM);
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&ra, (std::size_t) TM * a.n_out * 4); cudaMalloc(&rb, (std::size_t) TM * b.n_out * 4);
    cudaMalloc(&ya, (std::size_t) (TM + 2) * TM * a.n_out * 4); cudaMalloc(&yb, (std::size_t) (TM + 2) * TM * b.n_out * 4);
    cudaMemcpy(dwa, wa.data(), wa.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dwb, wb.data(), wb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    quantize_cols(dx, xq, a.n_in, TM, s);
    for (int j = 0; j < TM; ++j) {
        basalt::kernels::native_mmvq(a.type, dwa, (const uint8_t*) xq + j * qcol, ra + (std::size_t) j * a.n_out, a.n_in,
                                     a.n_out, 1, s);
        basalt::kernels::native_mmvq(b.type, dwb, (const uint8_t*) xq + j * qcol, rb + (std::size_t) j * b.n_out, b.n_in,
                                     b.n_out, 1, s);
    }
    const int tmax = g_hashes ? T8 : TM;
    for (int T = 1; T <= tmax; ++T)
        basalt::kernels::dense_tc_pair(a.type, dwa, ya + (std::size_t) T * TM * a.n_out, a.n_out, b.type, dwb,
                                       yb + (std::size_t) T * TM * b.n_out, b.n_out, xq, a.n_in, T, s);
    for (int half = 0; half < 2; ++half)   // the 8-column calls on columns 0..7 and 8..15
        basalt::kernels::dense_tc_pair(a.type, dwa, ya + ((std::size_t) (TM + 1) * TM + half * T8) * a.n_out, a.n_out,
                                       b.type, dwb, yb + ((std::size_t) (TM + 1) * TM + half * T8) * b.n_out, b.n_out,
                                       (const uint8_t*) xq + half * T8 * qcol, a.n_in, T8, s);
    if (!ck(cudaStreamSynchronize(s), "pair run")) return false;
    bool ok = true;
    unsigned long long h = 1469598103934665603ull;
    for (int m = 0; m < 2; ++m) {
        const Case& c = m == 0 ? a : b;
        float* dy = m == 0 ? ya : yb;
        float* dr = m == 0 ? ra : rb;
        std::vector<float> ref((std::size_t) TM * c.n_out), got((std::size_t) (TM + 2) * TM * c.n_out);
        cudaMemcpy(ref.data(), dr, ref.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
        long long bits = 0;
        const float* g8 = got.data() + (std::size_t) (TM + 1) * TM * c.n_out;
        for (int T = 1; T <= tmax; ++T) {
            if (T <= T8) fnv(h, &got[(std::size_t) T * TM * c.n_out], (std::size_t) T * c.n_out);
            for (int j = 0; j < T; ++j)
                for (int r = 0; r < c.n_out; ++r)
                    if (std::memcmp(&got[((std::size_t) T * TM + j) * c.n_out + r], &g8[(std::size_t) j * c.n_out + r], 4))
                        ++bits;
        }
        const float* gw = got.data() + (std::size_t) tmax * TM * c.n_out;
        double maxd = 0, maxr = 0, sd = 0, sr = 0;
        for (std::size_t i = 0; i < (std::size_t) tmax * c.n_out; ++i) {
            const double d = std::fabs((double) gw[i] - (double) ref[i]);
            maxd = std::max(maxd, d);
            maxr = std::max(maxr, std::fabs((double) ref[i]));
            sd += d * d;
            sr += (double) ref[i] * ref[i];
        }
        const double rms = std::sqrt(sd / std::max(sr, 1e-300));
        const bool k = bits == 0 && rms < 1e-3;
        std::printf("pair %-10s + %-10s [%d] T 1..%d columns bitwise: %s (%lld)  vs native: max|d|/max|ref| %.3e  "
                    "rms %.3e  %s\n",
                    a.name, b.name, m, tmax, bits == 0 ? "yes" : "NO", bits, maxd / std::max(maxr, 1e-30), rms,
                    k ? "PASS" : "FAIL");
        ok = ok && k;
    }
    const std::string key = "pair " + std::to_string(index);
    const char* fz = g_hashes ? "-" : frozen_check(key, h, ok);
    if (g_hashes) std::printf("    {\"%s\", 0x%016llxull},\n", key.c_str(), h);
    std::printf("pair %-10s + %-10s T 1..8 hash %016llx (%s)  %s\n", a.name, b.name, h, fz, ok ? "PASS" : "FAIL");
    cudaFree(dwa); cudaFree(dwb); cudaFree(xq); cudaFree(dx); cudaFree(ra); cudaFree(rb); cudaFree(ya); cudaFree(yb);
    return ok;
}

// µs per call: NL calls on NL distinct weight copies (> L2), captured once as a graph, best of `reps` replays
template<typename F>
double time_graph(F&& call, int NL, cudaStream_t s, int reps = 7) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    {
        basalt::kernels::PdlScope ps(true);   // with PDL=1, as the verify graphs launch them
        for (int l = 0; l < NL; ++l) call(l);
    }
    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaGraphLaunch(ge, s);
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float m = 0;
        cudaEventElapsedTime(&m, e0, e1);
        ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return 1000.0 * ms[ms.size() / 2] / NL;
}

// --bench-cases NAMES (comma-separated substrings) picks the cases; --bench-warm replays one weight copy (L2-warm)
void bench(cudaStream_t s, const char* pick, bool warm) {
    std::vector<const Case*> sel;
    for (const Case& c : CASES) {
        bool on = pick == nullptr ? (&c == &CASES[0] || &c == &CASES[1] || &c == &CASES[3] || &c == &CASES[4] ||
                                     &c == &CASES[7] || &c == &CASES[12])
                                  : false;
        if (pick != nullptr) {
            std::string all(pick), name(c.name);
            for (std::size_t a = 0; a <= all.size();) {
                std::size_t b = all.find(',', a);
                if (b == std::string::npos) b = all.size();
                if (b > a && name.find(all.substr(a, b - a)) != std::string::npos) on = true;
                a = b + 1;
            }
        }
        if (on) sel.push_back(&c);
    }
    std::printf("\nbench: us per call (median of 7 graph replays), GB/s of the weight bytes%s\n", warm ? ", L2-warm" : "");
    for (const Case* c : sel) {
        const std::size_t wb = basalt::kernels::native_mmvq_weight_bytes(c->type, c->n_in, c->n_out);
        const int NL = warm ? 16 : (int) std::max<std::size_t>(8, (400u << 20) / wb);
        std::mt19937 rng(99u);
        const std::vector<uint8_t> w = make_weights(*c, rng);
        uint8_t* dw = nullptr;
        void* xq = nullptr;
        float *dx = nullptr, *y = nullptr;
        cudaMalloc(&dw, wb * NL);
        for (int l = 0; l < (warm ? 1 : NL); ++l) cudaMemcpy(dw + wb * l, w.data(), wb, cudaMemcpyHostToDevice);
        cudaMalloc(&xq, basalt::kernels::native_q8_1_bytes(c->n_in, 1) * TM);
        cudaMalloc(&dx, (std::size_t) TM * c->n_in * 4);
        cudaMalloc(&y, (std::size_t) TM * c->n_out * 4);
        std::vector<float> x((std::size_t) TM * c->n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        quantize_cols(dx, xq, c->n_in, TM, s);
        std::printf("%-12s %5d x %5d  %.1f MB  floor %.2f us at 1.79 TB/s\n", c->name, c->n_in, c->n_out, wb / 1e6,
                    wb / 1.79e6);
        std::printf("   T   native us   GB/s   dtc us   GB/s   speedup   (T > 8: dtc vs two 8-column dtc calls)\n");
        for (int T = 1; T <= TM; ++T) {
            const double tn = T <= T8 ? time_graph([&](int l) {
                basalt::kernels::native_mmvq(c->type, dw + wb * (warm ? 0 : l), xq, y, c->n_in, c->n_out, T, s);
            }, NL, s) : time_graph([&](int l) {
                basalt::kernels::dense_tc(c->type, dw + wb * (warm ? 0 : l), xq, y, c->n_in, c->n_out, T8, s);
                basalt::kernels::dense_tc(c->type, dw + wb * (warm ? 0 : l),
                                          (const uint8_t*) xq + T8 * basalt::kernels::native_q8_1_bytes(c->n_in, 1),
                                          y + (std::size_t) T8 * c->n_out, c->n_in, c->n_out, T - T8, s);
            }, NL, s);
            const double td = time_graph([&](int l) {
                basalt::kernels::dense_tc(c->type, dw + wb * (warm ? 0 : l), xq, y, c->n_in, c->n_out, T, s);
            }, NL, s);
            std::printf("  %2d  %8.2f %7.0f %8.2f %7.0f   %.2fx\n", T, tn, wb / tn / 1e3, td, wb / td / 1e3, tn / td);
        }
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(y);
    }
}

// --bench-pair: dense_tc_pair of qkv + z against dense_tc qkv then native z, L2-cold (--bench-warm: L2-warm)
void bench_pair(cudaStream_t s, bool warm) {
    const int pairs[][2] = {{0, 6}, {1, 5}, {3, 5}, {0, 4}, {1, 6}, {3, 6}, {25, 26}};
    std::printf("\npair bench: us per call (median of 7 graph replays)\n");
    for (const auto& pr : pairs) {
        const Case &a = CASES[pr[0]], &b = CASES[pr[1]];
        const std::size_t wba = basalt::kernels::native_mmvq_weight_bytes(a.type, a.n_in, a.n_out);
        const std::size_t wbb = basalt::kernels::native_mmvq_weight_bytes(b.type, b.n_in, b.n_out);
        const std::size_t wb = ((wba + 255) & ~(std::size_t) 255) + ((wbb + 255) & ~(std::size_t) 255);
        const int NL = warm ? 16 : (int) std::max<std::size_t>(8, (400u << 20) / wb), NW = warm ? 1 : NL;
        std::mt19937 rng(5u);
        const std::vector<uint8_t> w0 = make_weights(a, rng), w1 = make_weights(b, rng);
        uint8_t* dw = nullptr;
        void* xq = nullptr;
        float *dx = nullptr, *y0 = nullptr, *y1 = nullptr;
        cudaMalloc(&dw, wb * NW);
        const std::size_t off1 = (wba + 255) & ~(std::size_t) 255;
        for (int l = 0; l < NW; ++l) {
            cudaMemcpy(dw + wb * l, w0.data(), wba, cudaMemcpyHostToDevice);
            cudaMemcpy(dw + wb * l + off1, w1.data(), wbb, cudaMemcpyHostToDevice);
        }
        cudaMalloc(&xq, basalt::kernels::native_q8_1_bytes(a.n_in, 1) * TM);
        cudaMalloc(&dx, (std::size_t) TM * a.n_in * 4);
        cudaMalloc(&y0, (std::size_t) TM * a.n_out * 4);
        cudaMalloc(&y1, (std::size_t) TM * b.n_out * 4);
        std::vector<float> x((std::size_t) TM * a.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        quantize_cols(dx, xq, a.n_in, TM, s);
        std::printf("%-10s + %-10s %.1f MB  floor %.2f us at 1.79 TB/s\n   T   serial us   pair us   GB/s\n", a.name,
                    b.name, (wba + wbb) / 1e6, (wba + wbb) / 1.79e6);
        for (const int T : {1, 2, 3, 4, 8, 12, 16}) {
            const double ts = time_graph([&](int l0) {   // T > 8: native_mmvq of z in two calls
                const int l = warm ? 0 : l0;
                basalt::kernels::dense_tc(a.type, dw + wb * l, xq, y0, a.n_in, a.n_out, T, s);
                for (int t0 = 0; t0 < T; t0 += T8)
                    basalt::kernels::native_mmvq(b.type, dw + wb * l + off1,
                                                 (const uint8_t*) xq + t0 * basalt::kernels::native_q8_1_bytes(b.n_in, 1),
                                                 y1 + (std::size_t) t0 * b.n_out, b.n_in, b.n_out, std::min(T8, T - t0), s);
            }, NL, s);
            const double tp = time_graph([&](int l0) {
                const int l = warm ? 0 : l0;
                basalt::kernels::dense_tc_pair(a.type, dw + wb * l, y0, a.n_out, b.type, dw + wb * l + off1, y1, b.n_out,
                                               xq, a.n_in, T, s);
            }, NL, s);
            std::printf("  %2d  %8.2f  %8.2f  %6.0f\n", T, ts, tp, (wba + wbb) / tp / 1e3);
        }
        cudaFree(dw); cudaFree(xq); cudaFree(dx); cudaFree(y0); cudaFree(y1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 9) {
        std::printf("dense_tc needs thread-block clusters; skipped\n");
        return 77;
    }
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    bool only_bench = false, run_bench = false, pair_only = false, bench_pair_on = false, warm = false;
    const char* pick = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench-only") only_bench = run_bench = true;
        else if (a == "--bench") run_bench = true;
        else if (a == "--pair-only") pair_only = true;          // only the dense_tc_pair checks
        else if (a == "--bench-pair") bench_pair_on = true;
        else if (a == "--bench-warm") warm = true;
        else if (a == "--bench-cases" && i + 1 < argc) pick = argv[++i];
        else if (a == "--hashes") g_hashes = true;
        else {
            std::printf("usage: dense_tc_parity [--bench | --bench-only] [--bench-cases NAMES] [--bench-warm] "
                        "[--bench-pair] [--pair-only] [--hashes]\n");
            return 2;
        }
    }
    bool ok = true;
    if (!only_bench) {
        if (!pair_only)
            for (const Case& c : CASES)
                if (c.type != 21 || c.n_in % 2048 == 0) ok = parity(c, s) && ok;
        const int pairs[][2] = {{0, 6}, {1, 5}, {2, 4}, {3, 6}, {0, 5}, {16, 17}, {3, 17}, {20, 21}, {16, 21}, {1, 17},
                                {20, 17}, {25, 26}, {25, 4}, {0, 26}};
        int index = 0;
        for (const auto& pr : pairs) ok = pair_parity(CASES[pr[0]], CASES[pr[1]], index++, s) && ok;
    }
    if (run_bench) {
        if (bench_pair_on) bench_pair(s, warm);
        else bench(s, pick, warm);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
