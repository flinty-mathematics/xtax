// xbkz: a standalone multithreaded BKZ lattice reducer.
//
// Each worker runs BKZ tours on its own copy of the basis. Every tour picks a
// random block size in [block_start, block], then runs one full pass. Workers
// share a global best under a mutex. When a worker stalls it either deepens the
// best basin with escalating local perturbations (if it holds the frontier) or
// jumps back to the global best and diverges (if it has fallen behind), which
// balances intensification of the best basis against broad exploration. A light
// unimodular nudge after reseed or promotion keeps workers from re-mining the
// same enumeration tree on an identical copy of the frontier basis.
//
// The reduction core is built from scratch: a double-precision Gram-Schmidt, an
// LLL inner reducer, and a pruned Schnorr-Euchner enumeration as the per-block
// SVP oracle. Basis entries are int64 with overflow checks, which suits the
// modest-entry bases the annealer produces. Very large raw entries are out of
// scope, there is no bignum path.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// AVX2 intrinsics for the hand-vectorised sieve reductions. Guarded so the file
// still builds (using the scalar fallback) on targets without AVX2. MSVC defines
// __AVX2__ under /arch:AVX2. GCC and Clang define it under -mavx2.
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "CLI11.hpp"
#include "xbkz_ui.hpp"

// Shared helpers: Lattice, CSV IO, atomic_write, ext_gcd, axpy_overflow,
// norm2_str (mat_io.hpp) and the g_stop / g_interrupted interrupt plumbing
// (stop_signal.hpp). The tuned reduction core below stays local to this file.
#include "mat_io.hpp"
#include "stop_signal.hpp"

// Floating-point type for the reduction core (Gram matrix, Gram-Schmidt mu/r,
// enumeration, sieve). Kept as a single alias so the precision policy lives in
// one place. We use double here: on MSVC the historical long double already
// aliased double, which is the only platform this is built and tuned on, so the
// reducer is known to be numerically fine in 64 bits. Using double everywhere
// also makes results consistent across the big three platforms (the x87 80-bit
// type would otherwise differ) and lets the hot reductions be vectorised with
// AVX2.
using real_t = double;

// The hand-vectorised reductions below operate on double* and are handed
// real_t arrays directly, so real_t must stay double. If the precision policy
// ever changes, those helpers (and their call sites) need revisiting.
static_assert(std::is_same<real_t, double>::value,
              "SIMD reduction helpers assume real_t is double");

using Clock = std::chrono::steady_clock;

// Optional context passed into a BKZ tour for live status reporting.
struct BkzTourContext {
    WorkerStatus* status = nullptr;
};

// Thrown when an int64 basis entry would overflow during reduction. The worker
// catches it, drops that attempt, and keeps the best basis it already shared.
struct ReduceOverflow {};

// Thrown when a bounded local reduction exceeds its work cap. Used for speculative
// block insertions where the caller can roll back and keep the worker moving.
struct ReduceAbort {};

// Computes sum_{t < n} x[t] * y[t] * w[t], the weighted dot product the sieve
// norms and the Gram-Schmidt recurrence both reduce to. Hand vectorised because
// the compiler leaves these reductions scalar under /fp:precise. The AVX2 and
// scalar paths use the same four-lane accumulation and the same final combine,
// so they produce identical results on every platform. Only the lane-parallel
// summation order differs from a plain left-to-right scalar sum. No FMA is used
// so the per-lane rounding matches the scalar fallback exactly.
//
// The AVX-512 path (opt-in XTAX_AVX512 build only, __AVX512F__ defined) sums 8
// doubles per step. Its 8-lane order differs from the 4-lane paths, so its
// result may differ in the last ULP. This is accepted by the build's opt-in
// determinism policy (cross-platform bit-for-bit is a nice-to-have, not a
// requirement, on the fast path). The default build never takes this branch.
static inline double weighted_dot3(const double* x, const double* y,
                                   const double* w, int n) {
#if defined(__AVX512F__)
    __m512d acc = _mm512_setzero_pd();
    int t = 0;
    for (; t + 8 <= n; t += 8) {
        __m512d vx = _mm512_loadu_pd(x + t);
        __m512d vy = _mm512_loadu_pd(y + t);
        __m512d vw = _mm512_loadu_pd(w + t);
        acc = _mm512_add_pd(acc, _mm512_mul_pd(_mm512_mul_pd(vx, vy), vw));
    }
    double s = _mm512_reduce_add_pd(acc);
    for (; t < n; ++t) s += x[t] * y[t] * w[t];
    return s;
#else
    double lane[4] = { 0.0, 0.0, 0.0, 0.0 };
    int t = 0;
#if defined(__AVX2__)
    __m256d acc = _mm256_setzero_pd();
    for (; t + 4 <= n; t += 4) {
        __m256d vx = _mm256_loadu_pd(x + t);
        __m256d vy = _mm256_loadu_pd(y + t);
        __m256d vw = _mm256_loadu_pd(w + t);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_mul_pd(vx, vy), vw));
    }
    _mm256_storeu_pd(lane, acc);
#else
    for (; t + 4 <= n; t += 4) {
        lane[0] += x[t + 0] * y[t + 0] * w[t + 0];
        lane[1] += x[t + 1] * y[t + 1] * w[t + 1];
        lane[2] += x[t + 2] * y[t + 2] * w[t + 2];
        lane[3] += x[t + 3] * y[t + 3] * w[t + 3];
    }
#endif
    double s = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; t < n; ++t) s += x[t] * y[t] * w[t];
    return s;
#endif
}

// The reducer owns one basis and its Gram-Schmidt data. Rows are integer
// vectors of length d. When transform tracking is on it also carries U so that
// B = U * L0 holds for the original basis L0.
struct Reducer {
    int n = 0;   // number of vectors
    int d = 0;   // dimension
    double delta = 0.99;
    bool track_u = true;
    bool u_valid = true;

    std::vector<int64_t> B;   // flat n*d, row-major (row i is [i*d, i*d+d))
    std::vector<int64_t> U;   // flat n*n, row-major (if track_u)
    std::vector<real_t> mu;           // n*n, lower triangle used
    std::vector<real_t> r;            // n, squared norms of b*_i
    std::vector<real_t> G;            // n*n Gram matrix B*B^T, kept in sync

    // Reusable scratch for a single rollback checkpoint (save_state /
    // restore_state). Holding these as members means a checkpoint reuses its
    // allocation instead of allocating on every speculative block insertion or
    // perturbation, and a restore is a set of contiguous copies rather than an
    // O(n^3) Gram-Schmidt rebuild. Only one checkpoint is ever live at a time
    // (the save/restore sites never nest on the same reducer).
    std::vector<int64_t> snap_B;
    std::vector<int64_t> snap_U;
    std::vector<real_t> snap_mu;
    std::vector<real_t> snap_r;
    std::vector<real_t> snap_G;
    bool snap_u_valid = true;

    int64_t* b_row(int i) { return B.data() + (size_t)i * d; }
    const int64_t* b_row(int i) const { return B.data() + (size_t)i * d; }
    int64_t* u_row(int i) { return U.data() + (size_t)i * n; }
    const int64_t* u_row(int i) const { return U.data() + (size_t)i * n; }

    real_t& M(int i, int j) { return mu[(size_t)i * n + j]; }
    real_t  M(int i, int j) const { return mu[(size_t)i * n + j]; }
    real_t& Gx(int i, int j) { return G[(size_t)i * n + j]; }
    real_t  Gx(int i, int j) const { return G[(size_t)i * n + j]; }

    real_t dot(int i, int j) const {
        const int64_t* a = b_row(i);
        const int64_t* b = b_row(j);
        real_t s = 0;
        for (int k = 0; k < d; ++k) s += (real_t)a[k] * (real_t)b[k];
        return s;
    }

    // sum_{t < cnt} M(i, t) * M(j, t) * r[t], the projection term of the
    // Gram-Schmidt recurrence. mu rows and r are contiguous, so this is the
    // weighted dot product the AVX2 helper vectorises.
    real_t gso_sum(int i, int j, int cnt) const {
        return weighted_dot3(mu.data() + (size_t)i * n, mu.data() + (size_t)j * n,
                             r.data(), cnt);
    }

    void init(const Lattice& L, bool track) {
        n = L.m;
        d = L.d;
        track_u = track;
        u_valid = track;
        B.assign((size_t)n * d, 0);
        for (int i = 0; i < n; ++i) {
            int64_t* bi = b_row(i);
            const int64_t* li = L.row(i);
            for (int k = 0; k < d; ++k) bi[k] = li[k];
        }
        if (track_u) {
            U.assign((size_t)n * n, 0);
            for (int i = 0; i < n; ++i) u_row(i)[i] = 1;
        }
        mu.assign((size_t)n * n, 0.0);
        r.assign(n, 0.0);
        G.assign((size_t)n * n, 0.0);
        build_gram();
    }

    // Recompute the whole Gram matrix from the basis. O(n^2 d). Used at startup,
    // after a rollback, and once per tour to bound floating-point drift in the
    // incrementally maintained G.
    void build_gram() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                real_t s = dot(i, j);
                Gx(i, j) = s;
                Gx(j, i) = s;
            }
        }
    }

    // B[dst] += c * B[src] (and the same on U), keeping the Gram matrix in sync.
    // Throws on basis overflow, leaving G untouched so the caller can roll back.
    void axpy(int dst, int src, int64_t c) {
        if (c == 0) return;
        int64_t* bd = b_row(dst);
        const int64_t* bs = b_row(src);
        for (int k = 0; k < d; ++k) {
            if (axpy_overflow(bd[k], c, bs[k])) throw ReduceOverflow{};
        }
        // Gram update for b_dst += c b_src. Off-diagonal first (uses old values),
        // then the diagonal which needs the old cross term and both old norms.
        real_t cc = (real_t)c;
        real_t old_ds = Gx(dst, src);
        real_t old_ss = Gx(src, src);
        real_t old_dd = Gx(dst, dst);
        for (int i = 0; i < n; ++i) {
            if (i == dst) continue;
            real_t v = Gx(dst, i) + cc * Gx(src, i);
            Gx(dst, i) = v;
            Gx(i, dst) = v;
        }
        Gx(dst, dst) = old_dd + 2.0 * cc * old_ds + cc * cc * old_ss;
        if (track_u && u_valid) {
            int64_t* ud = u_row(dst);
            const int64_t* us = u_row(src);
            for (int k = 0; k < n; ++k) {
                if (axpy_overflow(ud[k], c, us[k])) { u_valid = false; break; }
            }
        }
    }

    // Full Gram-Schmidt for rows [from, n), reusing valid data for rows < from.
    // Reads inner products from the Gram cache rather than recomputing them.
    void compute_gso_from(int from) {
        for (int i = from; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[j] > 0) ? s / r[j] : 0.0;
            }
            real_t s = Gx(i, i) - gso_sum(i, i, i);
            r[i] = s;
            M(i, i) = 1.0;
        }
    }

    void compute_gso() { compute_gso_from(0); }

    // Incremental Gram-Schmidt update after a unimodular change confined to the
    // block [k0, k0+h). The orthogonalized vectors b*_i for i >= k0+h are
    // unchanged (so r[i] and mu[i][j] for j outside the block stay valid). Only
    // the block's own GSO rows and the block columns of the tail rows change.
    void update_gso_after_block(int k0, int h) {
        int kend = k0 + h;
        // Recompute the block's own GSO rows in full.
        for (int i = k0; i < kend; ++i) {
            for (int j = 0; j < i; ++j) {
                real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[j] > 0) ? s / r[j] : 0.0;
            }
            real_t s = Gx(i, i) - gso_sum(i, i, i);
            r[i] = s;
            M(i, i) = 1.0;
        }
        // Refresh only the block columns of the tail rows (increasing j so the
        // recurrence can use the just-updated entries of the same row).
        for (int i = kend; i < n; ++i) {
            for (int j = k0; j < kend; ++j) {
                real_t s = Gx(i, j) - gso_sum(i, j, j);
                M(i, j) = (r[j] > 0) ? s / r[j] : 0.0;
            }
        }
    }

    // Size-reduce b_k against b_j using mu[k][j], updating the GSO row of k.
    void size_reduce(int k, int j) {
        real_t m = M(k, j);
        if (m > -0.5 && m < 0.5) return;
        // Nearest integer in the double domain (ties to even under the default
        // rounding mode). Cheaper than the long double llroundl this replaced,
        // which was an out-of-line call on every size-reduction step.
        long long q = std::llrint(m);
        if (q == 0) return;
        axpy(k, j, -(int64_t)q);
        for (int t = 0; t <= j; ++t) M(k, t) -= (real_t)q * M(j, t);
    }

    // Swap rows k-1 and k and update the GSO with the standard formulas.
    void swap_with_prev(int k) {
        std::swap_ranges(b_row(k), b_row(k) + d, b_row(k - 1));
        if (track_u) std::swap_ranges(u_row(k), u_row(k) + n, u_row(k - 1));
        // Keep the Gram matrix consistent: swap rows then columns k-1 and k.
        for (int i = 0; i < n; ++i) std::swap(Gx(k, i), Gx(k - 1, i));
        for (int i = 0; i < n; ++i) std::swap(Gx(i, k), Gx(i, k - 1));

        real_t m = M(k, k - 1);
        real_t Bk = r[k] + m * m * r[k - 1];
        if (Bk < 1e-30) {
            // The incremental formulas below divide by Bk, which is unsafe when
            // the new leading projection collapses. A swap of rows k-1, k only
            // changes the GSO of those two rows and the k-1, k columns of the
            // tail, so recompute just those from the (already swapped) Gram. This
            // is O(n^2) instead of the O(n^3) full rebuild this path used to do,
            // and it reads from the robust Gram rather than the drifting formulas.
            update_gso_after_block(k - 1, 2);
            return;
        }
        M(k, k - 1) = m * r[k - 1] / Bk;
        r[k] = r[k - 1] * r[k] / Bk;
        r[k - 1] = Bk;
        for (int j = 0; j < k - 1; ++j) std::swap(M(k - 1, j), M(k, j));
        for (int i = k + 1; i < n; ++i) {
            real_t t = M(i, k);
            M(i, k) = M(i, k - 1) - m * t;
            M(i, k - 1) = t + M(k, k - 1) * M(i, k);
        }
    }

    // LLL with the Lovasz condition, assuming the GSO is already valid.
    void lll(int start = 1, long long max_steps = 0) {
        int k = std::max(1, start);
        long long steps = 0;
        while (k < n) {
            if (g_stop.load(std::memory_order_relaxed)) return;
            if (max_steps > 0 && ++steps > max_steps) throw ReduceAbort{};
            for (int j = k - 1; j >= 0; --j) size_reduce(k, j);
            real_t lhs = r[k];
            real_t m = M(k, k - 1);
            real_t rhs = ((real_t)delta - m * m) * r[k - 1];
            if (lhs >= rhs) {
                ++k;
            } else {
                swap_with_prev(k);
                k = std::max(k - 1, 1);
            }
        }
    }

    real_t potential() const {
        real_t p = 0;
        for (int i = 0; i < n; ++i) {
            real_t ri = r[i] > 0 ? r[i] : 1e-30;
            p += (real_t)(n - i) * std::log(ri);
        }
        return p;
    }

    // Capture the full reducer state into the reusable scratch buffers so the
    // operation that follows can be rolled back with a plain copy. Only one
    // checkpoint is live at a time (the save/restore sites never nest on the
    // same reducer). Assigning into the pre-sized scratch reuses its storage
    // after the first call, so a checkpoint allocates nothing on the hot path.
    void save_state() {
        snap_B = B;
        if (track_u) snap_U = U;
        snap_mu = mu;
        snap_r = r;
        snap_G = G;
        snap_u_valid = u_valid;
    }

    // Roll back to the last save_state(). This is a set of contiguous copies of
    // the integer basis and the floating-point Gram/Gram-Schmidt state, so it
    // avoids the O(n^3) build_gram()+compute_gso() rebuild the old restore did.
    void restore_state() {
        B = snap_B;
        if (track_u) U = snap_U;
        mu = snap_mu;
        r = snap_r;
        G = snap_G;
        u_valid = snap_u_valid;
    }

    // Exact squared norm of row i (real_t accumulation of integer products).
    real_t row_norm2(int i) const {
        const int64_t* a = b_row(i);
        real_t s = 0;
        for (int k = 0; k < d; ++k) s += (real_t)a[k] * (real_t)a[k];
        return s;
    }

    // Squared norm of the shortest basis row, with its index. After reduction
    // this is normally row 0, but on an unreduced basis any row may be shortest.
    real_t shortest_norm2(int* idx = nullptr) const {
        real_t best = row_norm2(0);
        int bi = 0;
        for (int i = 1; i < n; ++i) {
            real_t v = row_norm2(i);
            if (v < best) { best = v; bi = i; }
        }
        if (idx) *idx = bi;
        return best;
    }
};

// Pruned Schnorr-Euchner enumeration over a block [k0, k0+h). Finds integer
// coefficients x (over basis vectors b_{k0+i}) whose projected vector is
// strictly shorter than the current first block vector, if one exists.
struct Enumerator {
    const Reducer* R = nullptr;
    int k0 = 0, h = 0;
    real_t prune = 0.0;     // 0 = exact, 1 = linear pruning
    real_t bound = 0;        // current best squared norm
    // GS norms at or below deg_eps -- a tiny fraction of the block head norm --
    // are treated as numerically zero. Catastrophic cancellation in the double
    // Gram-Schmidt can leave r[k] a spurious near-zero, which would otherwise
    // make the radius scan at that level explode. See dfs and run.
    static constexpr real_t k_degenerate_rr_rel = 1e-12;
    real_t deg_eps = 0.0;
    std::vector<real_t> rr;  // r values for the block
    // Block mu in natural row-major order: mu_row[k*h + j] = M(k0+k, k0+j) for
    // j < k. The incremental center update at level k reads the contiguous run
    // mu_row[k*h + (0..k-1)], so extending the running partial sums to the child
    // level vectorises as a plain scaled add.
    std::vector<real_t> mu_row;
    // Running center partial sums, one row per recursion level. tsum[k*h + j]
    // holds sum_{i>k} x[i] * M(k0+i, k0+j) for the coordinates already fixed
    // above level k, so center(k) = -tsum[k*h + k] is available in O(1) per node
    // and descending to a child costs only O(k) (a scaled add of the newly fixed
    // coordinate), instead of the old O(h) dot product recomputed at every node.
    std::vector<real_t> tsum;
    std::vector<int> x, best;
    bool found = false;
    bool aborted = false;         // set when a stop is requested mid-search
    long long nodes = 0;          // visited nodes, used to throttle stop polling
    long long node_limit = 0;      // 0 = no per-block cap
    WorkerStatus* status = nullptr;

    real_t prune_bound(int cnt) const {
        if (prune <= 0.0) return bound;
        real_t frac = 1.0 - prune * (1.0 - (real_t)cnt / (real_t)h);
        return bound * frac;
    }

    void dfs(int k, real_t partdist) {
        if (aborted) return;
        ++nodes;
        if (node_limit > 0 && nodes >= node_limit) {
            aborted = true;
            return;
        }
        if ((nodes & 0xFFFF) == 0) {
            if (status)
                status->enum_nodes.store(nodes, std::memory_order_relaxed);
            if (g_stop.load(std::memory_order_relaxed)) {
                aborted = true;
                return;
            }
        }
        int cnt = h - 1 - k;  // coordinates accumulated in partdist
        if (partdist >= prune_bound(cnt)) return;
        if (k < 0) {
            bool nz = false;
            for (int i = 0; i < h; ++i) if (x[i] != 0) { nz = true; break; }
            if (nz && partdist < bound) { bound = partdist; best = x; found = true; }
            return;
        }
        real_t center = -tsum[(size_t)k * h + k];
        long long base = std::llrint(center);
        real_t child_cap = prune_bound(cnt + 1);
        const real_t* mrow = mu_row.data() + (size_t)k * h;  // M(k0+k, k0+0..k-1)
        real_t* child = (k > 0) ? tsum.data() + (size_t)(k - 1) * h : nullptr;
        const real_t* self = tsum.data() + (size_t)k * h;
        for (int radius = 0;; ++radius) {
            bool any = false;
            int signs = (radius == 0) ? 1 : 2;
            for (int s = 0; s < signs; ++s) {
                long long xk = base + (s == 0 ? radius : -radius);
                real_t ck = (real_t)xk - center;
                real_t nd = partdist + ck * ck * rr[k];
                if (nd < child_cap) {
                    any = true;
                    x[k] = (int)xk;
                    // Extend the running center sums to the child level with the
                    // coordinate just fixed: tsum_{k-1}[j] = x[k]*M(k0+k,k0+j) +
                    // tsum_k[j] for j < k. Written fresh from the (stable) parent
                    // row each candidate, so no rounding drift accumulates.
                    if (child) {
                        real_t v = (real_t)xk;
                        for (int j = 0; j < k; ++j)
                            child[j] = v * mrow[j] + self[j];
                    }
                    dfs(k - 1, nd);
                }
            }
            if (radius > 0 && !any) break;
            // A GS norm at or below deg_eps (degenerate, or FP-drifted to a
            // spurious near-zero) barely grows the projected distance with the
            // radius, so the cap test above cannot end the scan in any reasonable
            // number of steps. Such a coordinate does not change the projected
            // distance, so only its nearest integer (radius 0) matters: explore
            // that and stop, instead of expanding to millions of nodes.
            if (rr[k] <= deg_eps) break;
            if (radius > 1 << 20) break;  // safety, never expected
        }
    }

    // Gaussian-heuristic squared radius for a block whose Gram-Schmidt squared
    // norms are rr[0..h-1]: GH^2 = (h / (2*pi*e)) * (prod rr[i])^(1/h), the
    // expected shortest squared norm of a random lattice of that volume. Returns
    // 0 if any rr[i] is non-positive (the estimate is then not meaningful).
    real_t gaussian_heuristic2() const {
        constexpr double two_pi_e = 2.0 * 3.14159265358979323846 * 2.71828182845904523536;
        double log_det = 0.0;
        for (int i = 0; i < h; ++i) {
            if (rr[i] <= 0.0) return 0.0;
            log_det += std::log(rr[i]);
        }
        return (real_t)((double)h / two_pi_e * std::exp(log_det / (double)h));
    }

    // Returns true and fills coeff if a strictly shorter vector is found.
    // gh_factor > 0 caps the search radius at min(rr[0], gh_factor^2 * GH^2),
    // a Gaussian-heuristic prune: it shrinks the tree at the cost of possibly
    // missing a vector between the cap and rr[0], which re-randomization across
    // tours and workers is expected to recover.
    bool run(const Reducer& red, int kappa, int block, real_t prune_amt,
             long long node_limit_, std::vector<int>& coeff,
             WorkerStatus* status_ = nullptr, real_t gh_factor = 0.0) {
        R = &red;
        k0 = kappa;
        h = block;
        prune = prune_amt;
        node_limit = node_limit_;
        status = status_;
        rr.assign(h, 0.0);
        for (int i = 0; i < h; ++i) rr[i] = red.r[k0 + i];
        mu_row.assign((size_t)h * h, 0.0);
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < i; ++j)
                mu_row[(size_t)i * h + j] = red.M(k0 + i, k0 + j);
        tsum.assign((size_t)h * h, 0.0);
        bound = rr[0];
        if (gh_factor > 0.0) {
            real_t gh2 = gaussian_heuristic2();
            real_t cap = gh_factor * gh_factor * gh2;
            if (cap > 0.0 && cap < bound) bound = cap;
        }
        deg_eps = (rr[0] > 0.0) ? rr[0] * k_degenerate_rr_rel : 0.0;
        x.assign(h, 0);
        best.clear();
        found = false;
        aborted = false;
        nodes = 0;
        // The top level has no coordinates fixed above it, so its center partial
        // sums start at zero. dfs then maintains them incrementally.
        dfs(h - 1, 0.0);
        if (found && bound < rr[0] * (1.0 - 1e-9)) {
            coeff = best;
            return true;
        }
        return false;
    }
};

// Primitive-reduce a coefficient vector. Return false if zero or too large.
static bool coeff_make_primitive(std::vector<int64_t>& c) {
    int64_t g = 0;
    for (int64_t v : c) g = std::gcd(g, (int64_t)std::llabs((long long)v));
    if (g == 0) return false;
    if (g > 1) for (int64_t& v : c) v /= g;
    for (int64_t v : c) {
        if (v != 0) {
            if (v < 0) for (int64_t& w : c) w = -w;
            break;
        }
    }
    for (int64_t v : c)
        if (std::llabs((long long)v) > (1LL << 22)) return false;
    return true;
}

// A proper (list-based) Gauss sieve on a single BKZ block. It maintains a list
// of block combinations that is kept pairwise Gauss-reduced: candidates are
// reduced against the whole list to stability, then existing list vectors that
// the new vector can shorten are pulled back out and re-queued. Collisions
// (vectors that reduce to zero) are dropped. When the list is full the longest
// vector is evicted in favour of a shorter newcomer. Any combination shorter
// than the block's first Gram-Schmidt vector r[k0] is returned for BKZ
// insertion.
//
// Each vector caches its projected coordinates a_s (combination expressed over
// b*_{k0+s}). Norms and inner products are then O(h) weighted sums over r, and
// a reduction v -= mu w updates a in O(h). This keeps the list sieve affordable
// to run on every block.
struct BlockSieve {
    const Reducer* R = nullptr;
    int k0 = 0, h = 0;
    // Contiguous copies of the block's r values and the h x h lower triangle of
    // mu (mu_block[i*h+j] = M(k0+i, k0+j)), filled once per run(). Reading these
    // instead of striding by n across the global arrays keeps the hot inner loops
    // cache friendly. rr and the projected coordinates a are spelled double (the
    // type the AVX2 reduction helpers consume). Since real_t is double, this is
    // the same precision as the rest of the reducer.
    std::vector<double> rr;
    std::vector<real_t> mu_block;

    struct Vec {
        std::vector<int64_t> c;  // integer coefficients over the block rows
        std::vector<double> a;   // projected coordinates over b*_{k0+s}
        real_t n2 = 0;           // projected squared norm
    };

    real_t norm_from_a(const std::vector<double>& a) const {
        return (real_t)weighted_dot3(a.data(), a.data(), rr.data(), h);
    }

    real_t dot_from_a(const std::vector<double>& x,
                           const std::vector<double>& y) const {
        return (real_t)weighted_dot3(x.data(), y.data(), rr.data(), h);
    }

    Vec make_vec(std::vector<int64_t> c) const {
        Vec v;
        v.c = std::move(c);
        v.a.assign((size_t)h, 0.0);
        for (int t = 0; t < h; ++t) {
            real_t s = 0;
            for (int i = t; i < h; ++i)
                s += (real_t)v.c[(size_t)i] * mu_block[(size_t)i * h + t];
            v.a[(size_t)t] = (double)s;
        }
        v.n2 = norm_from_a(v.a);
        return v;
    }

    static bool is_zero(const Vec& v) {
        for (int64_t x : v.c) if (x != 0) return false;
        return true;
    }

    // v -= mu w with mu = round(<v,w>/|w|^2). Returns true if v got strictly
    // shorter. Rejects reductions that would overflow the int64 coefficients.
    bool reduce_by(Vec& v, const Vec& w) const {
        if (w.n2 <= 0) return false;
        const real_t dot = dot_from_a(v.a, w.a);
        const long long mu = std::llrint(dot / w.n2);
        if (mu == 0) return false;
        const real_t m = (real_t)mu;
        const real_t newn2 = v.n2 - 2.0 * m * dot + m * m * w.n2;
        if (newn2 >= v.n2 * (1.0 - 1e-12)) return false;
        if (std::llabs(mu) > (1LL << 30)) return false;
        for (int i = 0; i < h; ++i) {
            const real_t nc =
                (real_t)v.c[(size_t)i] - m * (real_t)w.c[(size_t)i];
            if (nc > 9.0e17 || nc < -9.0e17) return false;
        }
        for (int i = 0; i < h; ++i) v.c[(size_t)i] -= mu * w.c[(size_t)i];
        const double md = (double)mu;
        for (int t = 0; t < h; ++t) v.a[(size_t)t] -= md * w.a[(size_t)t];
        v.n2 = norm_from_a(v.a);  // recompute to bound drift
        return true;
    }

    // ui_step_base is the cumulative sieve attempts already made in earlier blocks
    // of this pass. This block reports status->sieve_step as ui_step_base + done so
    // the UI sees one monotonically rising attempt counter for the whole pass.
    // attempts_out receives the number of attempts this block actually made (which
    // may be below the budget if its queue drained early).
    bool run(const Reducer& red, int k0_, int h_, int pool_cap, int sieve_iters,
             std::mt19937_64& rng, std::vector<int64_t>& out_coeff,
             long long ui_step_base = 0, WorkerStatus* status = nullptr,
             long long* attempts_out = nullptr) {
        if (attempts_out) *attempts_out = 0;
        if (pool_cap < 2 || h_ < 2) return false;
        if (g_stop.load(std::memory_order_relaxed)) return false;
        R = &red;
        k0 = k0_;
        h = h_;
        const real_t bound = red.r[k0];
        if (bound <= 0) return false;

        rr.assign((size_t)h, 0.0);
        for (int t = 0; t < h; ++t) rr[(size_t)t] = (double)red.r[k0 + t];
        mu_block.assign((size_t)h * h, 0.0);
        for (int i = 0; i < h; ++i)
            for (int j = 0; j <= i; ++j)
                mu_block[(size_t)i * h + j] = red.M(k0 + i, k0 + j);

        std::vector<Vec> list;
        list.reserve((size_t)pool_cap);
        std::vector<Vec> queue;

        // Seed with the block basis vectors and random small combinations.
        for (int i = 0; i < h; ++i) {
            if (g_stop.load(std::memory_order_relaxed)) return false;
            std::vector<int64_t> c((size_t)h, 0);
            c[(size_t)i] = 1;
            queue.push_back(make_vec(std::move(c)));
        }
        std::uniform_int_distribution<int> coin(0, 1);
        std::uniform_int_distribution<int> trip(-1, 1);
        auto sample_one = [&]() -> Vec {
            std::vector<int64_t> c((size_t)h, 0);
            int nz = std::min(h, 2 + (int)(rng() % (unsigned)std::max(1, h / 2)));
            for (int z = 0; z < nz; ++z) {
                int idx = (int)(rng() % (unsigned)h);
                int64_t v = (int64_t)trip(rng);
                if (v == 0) v = coin(rng) ? 1 : -1;
                c[(size_t)idx] += v;
            }
            return make_vec(std::move(c));
        };
        for (int t = 0; t < pool_cap; ++t) {
            if (g_stop.load(std::memory_order_relaxed)) return false;
            queue.push_back(sample_one());
        }

        bool found = false;
        real_t best_n2 = bound;
        std::vector<int64_t> best;
        auto consider = [&](const Vec& v) {
            if (v.n2 > 0 && v.n2 < best_n2 * (1.0 - 1e-9)) {
                best_n2 = v.n2;
                best = v.c;
                found = true;
            }
        };

        const size_t queue_cap = (size_t)pool_cap * 4;
        const long long budget_init =
            (long long)pool_cap * (long long)sieve_iters + (long long)queue.size();
        long long budget = budget_init;
        if (status)
            status->sieve_step.store(ui_step_base, std::memory_order_relaxed);
        long long done = 0;

        while (budget-- > 0) {
            if (g_stop.load(std::memory_order_relaxed)) break;
            ++done;
            if (status && ((done & 0x3F) == 0))
                status->sieve_step.store(ui_step_base + done,
                                         std::memory_order_relaxed);
            // Draw from the queue while it has work. Once it drains, keep the
            // sieve busy with fresh random samples so the full budget is used
            // (this is what gives sieve-iters its meaning and lets the sieve keep
            // hunting for a shorter vector instead of stopping early).
            Vec v;
            if (!queue.empty()) {
                v = std::move(queue.back());
                queue.pop_back();
            } else {
                v = sample_one();
            }

            // Reduce the candidate against the whole list until it is stable.
            bool changed = true;
            while (changed) {
                if (g_stop.load(std::memory_order_relaxed)) break;
                changed = false;
                for (const Vec& w : list)
                    if (reduce_by(v, w)) changed = true;
            }
            if (g_stop.load(std::memory_order_relaxed)) break;
            if (is_zero(v)) continue;  // collision
            consider(v);

            // Pull out any list vectors the candidate can shorten and re-queue
            // them, keeping the list pairwise reduced. Swap-remove for O(1).
            for (size_t i = 0; i < list.size();) {
                if (g_stop.load(std::memory_order_relaxed)) break;
                if (reduce_by(list[i], v)) {
                    Vec moved = std::move(list[i]);
                    list[i] = std::move(list.back());
                    list.pop_back();
                    if (!is_zero(moved) && queue.size() < queue_cap) {
                        consider(moved);
                        queue.push_back(std::move(moved));
                    }
                } else {
                    ++i;
                }
            }

            // Insert the candidate, evicting the longest vector when full.
            if ((int)list.size() < pool_cap) {
                list.push_back(std::move(v));
            } else {
                size_t worst = 0;
                for (size_t i = 1; i < list.size(); ++i)
                    if (list[i].n2 > list[worst].n2) worst = i;
                if (v.n2 < list[worst].n2)
                    list[worst] = std::move(v);
            }
        }

        if (status)
            status->sieve_step.store(ui_step_base + done, std::memory_order_relaxed);
        if (attempts_out) *attempts_out = done;

        if (!found) return false;
        if (!coeff_make_primitive(best)) return false;
        out_coeff = std::move(best);
        return true;
    }
};

// Build a unimodular h x h matrix whose first row is the primitive vector x.
static std::vector<std::vector<int64_t>> complete_unimodular(std::vector<int64_t> x) {
    int h = (int)x.size();

    // Fast path: if some coordinate is +/-1, complete with standard basis rows.
    // The transform is then tiny (x plus unit rows), which avoids the entry
    // growth the general gcd construction can cause. det = +/- x[unit] = +/- 1.
    int unit = -1;
    for (int i = 0; i < h; ++i) {
        if (x[i] == 1 || x[i] == -1) { unit = i; break; }
    }
    if (unit >= 0) {
        std::vector<std::vector<int64_t>> H(h, std::vector<int64_t>(h, 0));
        H[0] = x;
        int rr = 1;
        for (int j = 0; j < h; ++j) {
            if (j == unit) continue;
            H[rr][j] = 1;
            ++rr;
        }
        return H;
    }

    std::vector<std::vector<int64_t>> Cinv(h, std::vector<int64_t>(h, 0));
    for (int i = 0; i < h; ++i) Cinv[i][i] = 1;
    std::vector<int64_t> y = x;
    for (int j = 1; j < h; ++j) {
        if (y[j] == 0) continue;
        int64_t p = y[0], q = y[j];
        int64_t a, b;
        int64_t g = ext_gcd(p, q, a, b);
        int64_t pg = p / g, qg = q / g;
        y[0] = g; y[j] = 0;
        for (int t = 0; t < h; ++t) {
            int64_t r0 = Cinv[0][t], rj = Cinv[j][t];
            Cinv[0][t] = pg * r0 + qg * rj;
            Cinv[j][t] = -b * r0 + a * rj;
        }
    }
    if (y[0] < 0) {
        for (int t = 0; t < h; ++t) Cinv[0][t] = -Cinv[0][t];
    }
    return Cinv;
}

// Replace block rows [k0, k0+h) of B (and U) by H times the old block.
static void apply_block_transform(Reducer& red, int k0,
                                  const std::vector<std::vector<int64_t>>& H) {
    int h = (int)H.size();
    {
        const int d = red.d;
        std::vector<int64_t> nb((size_t)h * d, 0);
        for (int rIdx = 0; rIdx < h; ++rIdx) {
            int64_t* dst = nb.data() + (size_t)rIdx * d;
            for (int c = 0; c < h; ++c) {
                int64_t coef = H[rIdx][c];
                if (coef == 0) continue;
                const int64_t* src = red.b_row(k0 + c);
                for (int k = 0; k < d; ++k) {
                    if (axpy_overflow(dst[k], coef, src[k])) throw ReduceOverflow{};
                }
            }
        }
        for (int rIdx = 0; rIdx < h; ++rIdx)
            std::copy_n(nb.data() + (size_t)rIdx * d, d, red.b_row(k0 + rIdx));
    }

    // Update the Gram matrix from the old Gram via H, avoiding dot products.
    // T transforms the block rows, then the block columns are transformed too
    // for the block-block submatrix. Old Gram values are still present here.
    {
        int n = red.n;
        std::vector<std::vector<real_t>> oldblk(h, std::vector<real_t>(n));
        for (int a = 0; a < h; ++a)
            for (int j = 0; j < n; ++j) oldblk[a][j] = red.Gx(k0 + a, j);

        std::vector<std::vector<real_t>> T(h, std::vector<real_t>(n, 0.0));
        for (int a = 0; a < h; ++a)
            for (int c = 0; c < h; ++c) {
                real_t hac = (real_t)H[a][c];
                if (hac == 0) continue;
                const std::vector<real_t>& ob = oldblk[c];
                std::vector<real_t>& ta = T[a];
                for (int j = 0; j < n; ++j) ta[j] += hac * ob[j];
            }

        std::vector<std::vector<real_t>> bb(h, std::vector<real_t>(h, 0.0));
        for (int a = 0; a < h; ++a)
            for (int b = 0; b < h; ++b) {
                real_t s = 0;
                for (int e = 0; e < h; ++e) s += (real_t)H[b][e] * T[a][k0 + e];
                bb[a][b] = s;
            }

        for (int a = 0; a < h; ++a)
            for (int j = 0; j < n; ++j) {
                if (j >= k0 && j < k0 + h) continue;
                red.Gx(k0 + a, j) = T[a][j];
                red.Gx(j, k0 + a) = T[a][j];
            }
        for (int a = 0; a < h; ++a)
            for (int b = 0; b < h; ++b) red.Gx(k0 + a, k0 + b) = bb[a][b];
    }

    if (red.track_u && red.u_valid) {
        const int un = red.n;
        std::vector<int64_t> nu((size_t)h * un, 0);
        bool ok = true;
        for (int rIdx = 0; rIdx < h && ok; ++rIdx) {
            int64_t* dst = nu.data() + (size_t)rIdx * un;
            for (int c = 0; c < h && ok; ++c) {
                int64_t coef = H[rIdx][c];
                if (coef == 0) continue;
                const int64_t* src = red.u_row(k0 + c);
                for (int k = 0; k < un; ++k) {
                    if (axpy_overflow(dst[k], coef, src[k])) { ok = false; break; }
                }
            }
        }
        if (ok) {
            for (int rIdx = 0; rIdx < h; ++rIdx)
                std::copy_n(nu.data() + (size_t)rIdx * un, un, red.u_row(k0 + rIdx));
        } else {
            red.u_valid = false;
        }
    }
}

// Insert a shorter block vector via a unimodular block transform + LLL.
static bool try_insert_block_vector(Reducer& red, int kappa, int h,
                                    std::vector<int64_t> coeff) {
    if (g_stop.load(std::memory_order_relaxed)) return false;
    if (!coeff_make_primitive(coeff)) return false;
    const long long lll_step_cap =
        20000LL + 2000LL * (long long)h + 20LL * (long long)red.n;
    std::vector<std::vector<int64_t>> H = complete_unimodular(coeff);
    red.save_state();
    try {
        apply_block_transform(red, kappa, H);
        red.update_gso_after_block(kappa, h);
        red.lll(std::max(1, kappa), lll_step_cap);
        return true;
    } catch (const ReduceOverflow&) {
        red.restore_state();
        return false;
    } catch (const ReduceAbort&) {
        red.restore_state();
        return false;
    }
}

// Local block preprocessing (BKZ 2.0 style). Before the full-beta enumeration
// of a block, run a cheaper pass of smaller-beta (pre_beta) enumerations over
// the sub-blocks inside the window [kappa, kappa+h) so the projected basis is
// better reduced and the subsequent full enumeration tree is far smaller. Uses
// the same enumerate-and-insert machinery, restricted to the window. Any
// improvement it inserts is a valid unimodular change, so this only ever helps
// or leaves the block unchanged.
static void preprocess_block(Reducer& red, int kappa, int h, int pre_beta,
                             real_t prune, long long node_limit,
                             real_t gh_factor) {
    if (pre_beta < 2 || h <= pre_beta) return;
    Enumerator en;
    std::vector<int> coeff;
    std::vector<int64_t> coeff64;
    const int end = kappa + h;
    for (int k = kappa; k + 2 <= end; ++k) {
        if (g_stop.load(std::memory_order_relaxed)) return;
        int hb = std::min(pre_beta, end - k);
        if (hb < 2) break;
        if (!en.run(red, k, hb, prune, node_limit, coeff, nullptr, gh_factor))
            continue;
        coeff64.assign(coeff.begin(), coeff.end());
        if (!coeff_make_primitive(coeff64)) continue;
        try_insert_block_vector(red, k, hb, std::move(coeff64));
    }
}

// One BKZ tour at block size beta. Returns true if anything moved.
// A tour commits to a single SVP oracle for all of its blocks: the block sieve
// when sieving is enabled and beta > sieve_beta, otherwise Schnorr-Euchner
// enumeration. The choice depends only on the tour's block size, so a tour is
// never a mix of sieved and enumerated blocks. sieve_beta <= 0 (or a disabled
// sieve) means enumeration for every tour.
// When refresh_gso is set the Gram and Gram-Schmidt data are rebuilt from the
// basis first, which bounds floating-point drift in the incrementally
// maintained values. Callers do this every few tours rather than every tour.
static bool bkz_tour(Reducer& red, int beta, std::mt19937_64& rng, real_t prune,
                     bool randomize_order, bool refresh_gso, int sieve_pool,
                     int sieve_iters, int sieve_beta, long long enum_node_limit,
                     int preprocess_beta, real_t gh_factor,
                     const BkzTourContext* ctx = nullptr) {
    int n = red.n;
    bool changed = false;
    if (refresh_gso) {
        red.build_gram();
        red.compute_gso();
    }
    std::vector<int> order(std::max(0, n - 1));
    std::iota(order.begin(), order.end(), 0);
    if (randomize_order) std::shuffle(order.begin(), order.end(), rng);

    const int nblocks = (int)order.size();

    // The oracle is chosen once for the whole tour from its block size: the
    // block sieve when sieving is enabled and beta exceeds the threshold,
    // otherwise enumeration. Every block in the tour then uses that single
    // oracle, so a tour is never a mix of the two.
    const bool use_sieve =
        (sieve_beta > 0 && sieve_pool > 0 && sieve_iters > 0 && beta > sieve_beta);

    if (ctx && ctx->status) {
        ctx->status->cur_beta.store(beta, std::memory_order_relaxed);
        ctx->status->block_total.store(nblocks, std::memory_order_relaxed);
        ctx->status->block_idx.store(0, std::memory_order_relaxed);
        ctx->status->phase.store(
            (int)(use_sieve ? WorkerPhase::sieving : WorkerPhase::tour),
            std::memory_order_relaxed);
    }

    Enumerator en;
    BlockSieve sieve;
    std::vector<int> coeff;
    std::vector<int64_t> coeff64;
    int block_idx = 0;

    // Per-block sieve budget cap, matching BlockSieve::run's budget_init (pool *
    // iters plus the initial queue of h block basis vectors and pool seeds). Used
    // as the denominator of the per-block sieve progress shown in the UI.
    auto block_budget = [&](int hh) -> long long {
        return (long long)sieve_pool * (long long)sieve_iters
               + (long long)hh + (long long)sieve_pool;
    };

    for (int kappa : order) {
        if (g_stop.load(std::memory_order_relaxed)) break;
        int k1 = std::min(kappa + beta, n);
        int h = k1 - kappa;
        if (h < 2) continue;
        ++block_idx;
        if (ctx && ctx->status) {
            ctx->status->block_idx.store(block_idx, std::memory_order_relaxed);
            ctx->status->block_kappa.store(kappa, std::memory_order_relaxed);
            ctx->status->block_h.store(h, std::memory_order_relaxed);
        }

        if (use_sieve) {
            // Sieve oracle. Progress is reported per block as the attempts
            // made against this block's budget.
            if (ctx && ctx->status) {
                ctx->status->phase.store((int)WorkerPhase::sieving,
                                         std::memory_order_relaxed);
                ctx->status->sieve_step.store(0, std::memory_order_relaxed);
                ctx->status->sieve_total.store(block_budget(h),
                                               std::memory_order_relaxed);
            }
            std::vector<int64_t> sc;
            long long attempts = 0;
            bool hit = sieve.run(red, kappa, h, sieve_pool, sieve_iters, rng, sc,
                                 0, ctx ? ctx->status : nullptr, &attempts);
            if (ctx && ctx->status)
                ctx->status->sieve_step.store(attempts, std::memory_order_relaxed);
            if (g_stop.load(std::memory_order_relaxed)) break;
            if (hit) {
                if (ctx && ctx->status)
                    ctx->status->blocks_hit.fetch_add(1, std::memory_order_relaxed);
                if (try_insert_block_vector(red, kappa, h, std::move(sc)))
                    changed = true;
            }
        } else {
            // Enumeration oracle (Schnorr-Euchner).
            // Optional local preprocessing shrinks the enumeration tree before
            // the full-beta search on this block.
            if (preprocess_beta > 0 && h > preprocess_beta) {
                preprocess_block(red, kappa, h, preprocess_beta, prune,
                                 enum_node_limit, gh_factor);
                if (g_stop.load(std::memory_order_relaxed)) break;
            }
            if (ctx && ctx->status) {
                ctx->status->phase.store((int)WorkerPhase::tour,
                                         std::memory_order_relaxed);
                ctx->status->enum_nodes.store(0, std::memory_order_relaxed);
            }
            if (!en.run(red, kappa, h, prune, enum_node_limit, coeff,
                        ctx ? ctx->status : nullptr, gh_factor)) {
                if (ctx && ctx->status)
                    ctx->status->enum_nodes.store(en.nodes, std::memory_order_relaxed);
                continue;
            }
            if (ctx && ctx->status)
                ctx->status->enum_nodes.store(en.nodes, std::memory_order_relaxed);

            coeff64.assign(coeff.begin(), coeff.end());
            if (!coeff_make_primitive(coeff64)) continue;

            if (ctx && ctx->status)
                ctx->status->blocks_hit.fetch_add(1, std::memory_order_relaxed);
            if (try_insert_block_vector(red, kappa, h, std::move(coeff64)))
                changed = true;
        }
        if (g_stop.load(std::memory_order_relaxed)) break;
    }

    if (ctx && ctx->status && !g_stop.load(std::memory_order_relaxed))
        ctx->status->phase.store((int)WorkerPhase::tour, std::memory_order_relaxed);

    if (ctx && ctx->status && changed)
        ctx->status->tours_changed.fetch_add(1, std::memory_order_relaxed);
    return changed;
}

// A tiny unimodular perturbation: one or two random row shears b_i += +/- b_j,
// then an optional quick LLL pass. Used after reseed or promotion so workers do
// not keep mining the same enumeration tree on an identical basis. Overflow
// rolls back. Pass lll_max_steps = 0 to shear only (no LLL). The default is a
// bounded reduction suitable for an already-LLL-reduced basis.
static void light_nudge(Reducer& red, std::mt19937_64& rng,
                        long long lll_max_steps = -1) {
    int n = red.n;
    if (n < 2) return;
    red.save_state();
    try {
        std::uniform_int_distribution<int> row(0, n - 1);
        const int shears = 1 + (int)(rng() & 1u);
        int min_row = n;
        for (int t = 0; t < shears; ++t) {
            if (g_stop.load(std::memory_order_relaxed)) return;
            int i = row(rng);
            int j = row(rng);
            if (i == j) j = (j + 1) % n;
            const int64_t s = (rng() & 1u) ? 1 : -1;
            red.axpy(i, j, s);
            min_row = std::min(min_row, std::min(i, j));
        }
        if (g_stop.load(std::memory_order_relaxed)) return;
        // axpy already maintained the Gram matrix incrementally, so build_gram()
        // is redundant. Only the Gram-Schmidt rows from the first perturbed row
        // onward can have changed, so refresh just those instead of all n.
        red.compute_gso_from(std::min(min_row, n));
        if (lll_max_steps < 0)
            lll_max_steps = 200000LL + 1000LL * (long long)n;
        if (lll_max_steps > 0)
            red.lll(1, lll_max_steps);
    } catch (const ReduceOverflow&) {
        red.restore_state();
    } catch (const ReduceAbort&) {
        red.restore_state();
    }
}

// Localized rerandomization used to diversify a worker. A random contiguous
// window of rows is mixed with random unimodular shears, then the basis is
// optionally re-reduced. Confining the shears to a window perturbs the local
// structure enough that LLL does not simply undo them. The strength in (0, 1]
// scales both the window size and the number of shears: small values nudge the
// basis (local intensification near a good point), 1.0 is a full diversifying
// kick. Pass lll_max_steps = 0 to shear only (no LLL). Overflow rolls back.
static void strong_randomize(Reducer& red, std::mt19937_64& rng,
                             double strength = 1.0, long long lll_max_steps = -1) {
    int n = red.n;
    if (n < 4) return;
    strength = std::clamp(strength, 0.05, 1.0);
    int hard_max = std::min(40, n);
    int wmin = std::min(8, hard_max);
    int wmax = std::max(wmin, (int)std::lround(strength * (double)hard_max));
    std::uniform_int_distribution<int> wlen(wmin, wmax);
    int w = wlen(rng);
    std::uniform_int_distribution<int> startpick(0, n - w);
    int a = startpick(rng);
    std::uniform_int_distribution<int> inwin(0, w - 1);

    red.save_state();
    try {
        int moves = std::max(1, (int)std::lround(2.0 * (double)w * strength));
        for (int t = 0; t < moves; ++t) {
            if (g_stop.load(std::memory_order_relaxed)) return;
            int i = a + inwin(rng), j = a + inwin(rng);
            if (i == j) continue;
            int64_t s = (rng() & 1) ? 1 : -1;
            red.axpy(i, j, s);
        }
        if (g_stop.load(std::memory_order_relaxed)) return;
        // The shears are confined to the window [a, a+w) and axpy keeps the Gram
        // matrix in sync, so build_gram() is redundant and only the Gram-Schmidt
        // rows from a onward need recomputing.
        red.compute_gso_from(a);
        if (lll_max_steps < 0)
            lll_max_steps = 200000LL + 1000LL * (long long)n;
        if (lll_max_steps > 0)
            red.lll(1, lll_max_steps);
    } catch (const ReduceOverflow&) {
        red.restore_state();
    } catch (const ReduceAbort&) {
        red.restore_state();
    }
}

// Squared norm of the shortest row, with its index, over a flat row-major
// basis (rows x cols).
static real_t flat_shortest_norm2(const std::vector<int64_t>& B, int rows,
                                  int cols, int* idx = nullptr) {
    real_t best = -1;
    int bi = 0;
    for (int i = 0; i < rows; ++i) {
        const int64_t* row = B.data() + (size_t)i * cols;
        real_t s = 0;
        for (int k = 0; k < cols; ++k) s += (real_t)row[k] * (real_t)row[k];
        if (best < 0 || s < best) { best = s; bi = i; }
    }
    if (idx) *idx = bi;
    return best < 0 ? 0 : best;
}

static bool better(real_t b0, real_t pot,
                   real_t cur_b0, real_t cur_pot) {
    if (b0 < cur_b0 * (1.0 - 1e-12)) return true;
    if (b0 > cur_b0 * (1.0 + 1e-12)) return false;
    return pot < cur_pot;
}

static bool promote(Reducer& local, GlobalBest& best) {
    real_t b0 = local.shortest_norm2();
    // Cheap lock-free reject: only a strictly shorter vector is worth promoting,
    // so skip the lock and the copy when this result cannot beat the current b0.
    if (best.has_relaxed.load(std::memory_order_acquire)) {
        double cur = best.b0_relaxed.load(std::memory_order_relaxed);
        if ((double)b0 >= cur * (1.0 - 1e-12)) return false;
    }
    real_t pot = local.potential();
    // Copy the basis (and transform) into scratch outside the lock, then swap
    // it into place under the mutex. This keeps the lock-held time to O(1)
    // pointer swaps instead of the full O(n*d)/O(n^2) copy other workers wait on.
    std::vector<int64_t> b_copy = local.B;
    std::vector<int64_t> u_copy;
    if (local.track_u) u_copy = local.U;
    std::lock_guard<std::mutex> lk(best.mtx);
    if (!best.has || better(b0, pot, best.b0, best.pot)) {
        best.B.swap(b_copy);
        if (local.track_u) best.U.swap(u_copy);
        best.n = local.n;
        best.d = local.d;
        best.u_valid = local.u_valid;
        best.track_u = local.track_u;
        best.b0 = b0;
        best.pot = pot;
        best.has = true;
        best.b0_relaxed.store((double)b0, std::memory_order_relaxed);
        best.has_relaxed.store(true, std::memory_order_release);
        best.improvements.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

// Copy the integer state of the global best into a local reducer and rebuild its
// Gram-Schmidt data. Returns false if there is no global best yet. Sets n, d and
// resizes the GSO buffers so this works on a default-constructed Reducer.
static bool reseed_from_best(Reducer& local, GlobalBest& best) {
    {
        std::lock_guard<std::mutex> lk(best.mtx);
        if (!best.has) return false;
        local.n = best.n;
        local.d = best.d;
        local.B = best.B;
        if (best.track_u && local.track_u) {
            local.U = best.U;
            local.u_valid = best.u_valid;
        }
    }
    local.mu.assign((size_t)local.n * local.n, 0.0);
    local.r.assign(local.n, 0.0);
    local.G.assign((size_t)local.n * local.n, 0.0);
    local.build_gram();
    local.compute_gso();
    return true;
}

// Install a reducer's basis as the initial global best (not counted as a BKZ
// improvement). Called once in main after the shared initial LLL pass.
static void seed_global_best(const Reducer& red, GlobalBest& best) {
    const real_t b0 = red.shortest_norm2();
    const real_t pot = red.potential();
    std::lock_guard<std::mutex> lk(best.mtx);
    best.B = red.B;
    if (red.track_u) best.U = red.U;
    best.n = red.n;
    best.d = red.d;
    best.u_valid = red.u_valid;
    best.track_u = red.track_u;
    best.b0 = b0;
    best.pot = pot;
    best.has = true;
    best.b0_relaxed.store((double)b0, std::memory_order_relaxed);
    best.has_relaxed.store(true, std::memory_order_release);
}

struct RunParams {
    int threads = 1;
    int block = 20;
    int block_start = 2;
    double delta = 0.99;
    double prune = 0.0;
    double max_seconds = 0.0;
    int reseed_every_k = 10;
    int sieve_pool = 64;
    int sieve_iters = 16;
    int sieve_beta = 0;   // sieve tours whose block size exceeds this (0 = enum only)
    long long enum_node_limit = 10000000; // per block (0 = unlimited)
    uint64_t seed = 0;
    bool track_u = true;
    bool report_status = true;
    bool no_init_lll = false;
    bool pin_threads = true;   // pin each worker to a physical core (Windows)
    bool progressive = true;   // per-worker progressive beta schedule (else random)
    int preprocess_beta = 0;   // local block preprocessing block size (0 = off)
    double gh_factor = 0.0;    // Gaussian-heuristic enum radius cap factor (0 = off)
};

// One shared initial LLL on the loaded basis before workers start. Seeds
// global best with the result (or the raw basis when --no-init-lll is set).
// Returns false on overflow during LLL.
static bool prepare_starting_basis(const Lattice& L0, const RunParams& p,
                                   GlobalBest& best) {
    Reducer red;
    red.init(L0, p.track_u);
    red.delta = p.delta;
    red.build_gram();
    red.compute_gso();
    if (!p.no_init_lll) {
        red.save_state();
        try {
            red.lll(1);
        } catch (const ReduceOverflow&) {
            red.restore_state();
            return false;
        }
    }
    seed_global_best(red, best);
    return true;
}

// Uniform random block size in [min_beta, max_beta] inclusive.
static int random_beta(std::mt19937_64& rng, int min_beta, int max_beta) {
    std::uniform_int_distribution<int> dist(min_beta, max_beta);
    return dist(rng);
}

// Larger block sizes within the schedule get slightly more pruning so enumeration
// stays tractable.
static real_t tour_prune(int beta, int min_beta, double base_prune) {
    real_t pr = (real_t)base_prune;
    if (beta > min_beta) pr += 0.05 * (real_t)(beta - min_beta);
    if (pr > 0.9) pr = 0.9;
    if (pr < 0.0) pr = 0.0;
    return pr;
}

static void worker(int id, const Lattice& L0, RunParams p, GlobalBest& best,
                   WorkerStatus& status) {
    Reducer local;
    local.track_u = p.track_u;
    local.delta = p.delta;
    std::mt19937_64 rng(p.seed + (uint64_t)id * 0x9E3779B97F4A7C15ull);

    const int min_beta = std::max(2, p.block_start);
    const int max_beta = std::max(min_beta, p.block);
    const int beta_span = std::max(1, max_beta - min_beta + 1);
    const bool randomize_order = (id != 0);
    const int gso_refresh = 8;

    // Per-worker diversity: workers run slightly different pruning aggressiveness
    // so the pool explores a spread of speed/quality trade-offs rather than
    // identical searches. Kept small so no worker is starved of exact search.
    const double worker_prune = std::min(0.9, p.prune + 0.05 * (double)(id % 4));

    // A worker's diversified progressive start: different workers begin their
    // beta ramp at different block sizes so the pool covers a range at once.
    auto progressive_start = [&]() {
        return std::min(max_beta, min_beta + (id % beta_span));
    };

    status.target_beta.store(max_beta, std::memory_order_relaxed);
    status.phase.store((int)WorkerPhase::starting, std::memory_order_relaxed);

    auto stop = [&]() { return g_stop.load(std::memory_order_relaxed); };

    auto refresh_local_b0 = [&]() {
        real_t b0 = local.shortest_norm2();
        status.local_b0.store((double)b0, std::memory_order_relaxed);
        return b0;
    };

    BkzTourContext ctx;
    ctx.status = p.report_status ? &status : nullptr;

    // When --no-init-lll is set the input is already reduced. Perturbation
    // helpers still shear to diversify but skip their follow-up LLL pass.
    const long long perturb_lll_cap = p.no_init_lll ? 0LL : -1LL;

    try {
        // Main already ran one shared initial LLL (unless --no-init-lll) and
        // seeded global best. Copy that starting basis instead of repeating LLL
        // in every worker.
        if (!reseed_from_best(local, best)) {
            local.init(L0, p.track_u);
            local.delta = p.delta;
            local.build_gram();
            local.compute_gso();
        }
        refresh_local_b0();

        // Workers 1+ diversify from the shared starting basis with random shears.
        if (id != 0) {
            if (p.report_status)
                status.phase.store((int)WorkerPhase::warmup,
                                   std::memory_order_relaxed);
            strong_randomize(local, rng, 1.0, perturb_lll_cap);
            refresh_local_b0();
        }

        int tour = 0;
        // Progressive schedule: start small (diversified per worker) and ramp
        // beta up over tours so each larger-beta search is preprocessed by the
        // earlier smaller-beta reductions. Without --progressive, fall back to
        // the previous random-beta-per-tour behaviour.
        int beta = p.progressive ? progressive_start()
                                  : random_beta(rng, min_beta, max_beta);
        real_t local_b0 = refresh_local_b0();
        int stale = 0;   // consecutive tours without a local improvement

        while (!stop()) {
            ++tour;
            status.tour.store(tour, std::memory_order_relaxed);
            status.cur_beta.store(beta, std::memory_order_relaxed);
            status.phase.store((int)WorkerPhase::tour, std::memory_order_relaxed);

            const real_t prune = tour_prune(beta, min_beta, worker_prune);
            bkz_tour(local, beta, rng, prune, randomize_order,
                     (tour % gso_refresh) == 0, p.sieve_pool, p.sieve_iters,
                     p.sieve_beta, p.enum_node_limit, p.preprocess_beta,
                     (real_t)p.gh_factor, &ctx);
            if (stop()) {
                status.phase.store((int)WorkerPhase::stopped,
                                   std::memory_order_relaxed);
                break;
            }

            // Assess the tour result on the un-perturbed basis: update the local
            // high-water anchor and the staleness counter, and promote, all from
            // the basis the tour actually produced (before any nudge).
            //
            // In progressive mode staleness is only counted once the schedule has
            // peaked at max_beta. Ramp-up tours run at small/medium block sizes
            // that are not expected to improve an already-reduced basis, so
            // counting them would trip the reseed (which also resets the ramp)
            // before beta ever reaches its productive size.
            const bool schedule_peaked = !p.progressive || beta >= max_beta;
            real_t b0 = refresh_local_b0();
            if (b0 < local_b0 * (1.0 - 1e-12)) {
                local_b0 = b0;
                stale = 0;
            } else if (schedule_peaked) {
                ++stale;
            } else {
                stale = 0;
            }

            // If this tour set (or matched) the frontier, explore around it with
            // a light nudge. local_b0 keeps the frontier value rather than the
            // perturbed one, so the catch-up reseed below does not mistake our own
            // exploratory nudge for having fallen behind and snap us straight back
            // (which would loop on every improvement).
            if (promote(local, best)) {
                light_nudge(local, rng, perturb_lll_cap);
                refresh_local_b0();  // reflect the perturbed basis in the UI only
            }

            // Before starting another round, pick up a strictly better basis
            // another worker published while this worker was busy. This compares
            // the frontier anchor, so an exploratory perturbation does not trip it.
            if (!stop() && best.has_relaxed.load(std::memory_order_acquire)) {
                double gbest = best.b0_relaxed.load(std::memory_order_relaxed);
                if ((double)local_b0 > gbest * (1.0 + 1e-9)) {
                    status.phase.store((int)WorkerPhase::reseed,
                                       std::memory_order_relaxed);
                    if (reseed_from_best(local, best)) {
                        local_b0 = local.shortest_norm2();  // anchor at the frontier
                        light_nudge(local, rng, perturb_lll_cap);
                        refresh_local_b0();  // UI only, local_b0 stays the anchor
                        stale = 0;
                    }
                }
            }

            // Reseed only after reseed_every_k consecutive tours without local
            // improvement. A worker that has fallen behind the global best jumps
            // back to the frontier and diverges. One that is holding (or leading)
            // the frontier instead perturbs in place, since reseeding it from its
            // own basis is a no-op.
            //
            // In progressive mode this is gated on the schedule having peaked at
            // max_beta. The ramp-up tours run at small/medium block sizes that
            // cannot improve an already-reduced basis, so counting them as
            // staleness would let the reseed fire (and reset the ramp to its
            // start) before beta ever reaches the productive large block size.
            // That produced a worker whose block size flipped rapidly while it
            // made no real progress, and made the least-pruned workers re-run
            // their heaviest mid-beta enumerations over and over. We instead let
            // beta climb to the top, then measure staleness there.
            if (!stop() && p.reseed_every_k > 0 && schedule_peaked
                    && stale >= p.reseed_every_k) {
                bool have_global = best.has_relaxed.load(std::memory_order_acquire);
                double gbest = have_global
                    ? best.b0_relaxed.load(std::memory_order_relaxed) : 0.0;
                bool behind =
                    have_global && (double)local_b0 > gbest * (1.0 + 1e-6);

                status.phase.store((int)WorkerPhase::reseed,
                                   std::memory_order_relaxed);
                // Temperature-like kick: the longer a worker has stalled the
                // harder it perturbs, with a per-worker offset for diversity. A
                // worker that has fallen behind the frontier jumps back to it and
                // diversifies strongly. One still holding the frontier intensifies
                // locally with a gentler kick.
                double heat = 0.04 * (double)stale + 0.05 * (double)(id % 4);
                if (behind) {
                    if (reseed_from_best(local, best))
                        local_b0 = local.shortest_norm2();  // anchor at the frontier
                    strong_randomize(local, rng, std::clamp(0.5 + heat, 0.4, 1.0),
                                     perturb_lll_cap);
                } else {
                    strong_randomize(local, rng, std::clamp(0.3 + heat, 0.2, 0.7),
                                     perturb_lll_cap);
                }
                // local_b0 stays anchored to the frontier we are exploring around.
                // refresh only updates the UI with the perturbed basis.
                refresh_local_b0();
                stale = 0;
                // Restart the progressive ramp (diversified) after a reseed so
                // the fresh basis is re-grown through the smaller block sizes.
                if (p.progressive) beta = progressive_start();
            }

            // Advance the schedule: ramp beta toward the maximum block size, or
            // draw a fresh random size when progressive scheduling is disabled.
            if (p.progressive) {
                if (beta < max_beta) ++beta;
            } else {
                beta = random_beta(rng, min_beta, max_beta);
            }
        }
    } catch (const ReduceOverflow&) {
    }
    status.phase.store((int)WorkerPhase::stopped, std::memory_order_relaxed);
}

#if defined(_WIN32)
// Affinity masks for the physical cores in processor group 0 (one mask per
// core, covering that core's logical processors). Enumeration is FP-bound, so
// pinning each worker to its own physical core avoids two workers contending for
// one core's shared execution units via hyperthreading. Cores in other
// processor groups (systems with > 64 logical processors) are skipped. Pinning
// is then simply left off, which is safe. Returns an empty vector if the
// topology cannot be queried.
static std::vector<DWORD_PTR> physical_core_masks() {
    std::vector<DWORD_PTR> masks;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return masks;
    std::vector<char> buf((size_t)len);
    auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len))
        return masks;
    char* ptr = buf.data();
    char* end = buf.data() + len;
    while (ptr < end) {
        auto* cur = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (cur->Relationship == RelationProcessorCore &&
            cur->Processor.GroupCount >= 1 &&
            cur->Processor.GroupMask[0].Group == 0) {
            masks.push_back((DWORD_PTR)cur->Processor.GroupMask[0].Mask);
        }
        ptr += cur->Size;
    }
    return masks;
}
#endif

int main(int argc, char** argv) {
    install_signal_handlers();

    CLI::App app{ "xbkz: multithreaded progressive BKZ lattice reducer" };

    std::string l_csv;
    std::string out_csv = "reduced.csv";
    std::string u_csv = "U.csv";
    std::string short_csv = "shortest.csv";
    RunParams p;
    int logical_cores = (int)std::thread::hardware_concurrency();
    if (logical_cores < 1) logical_cores = 1;
#if defined(_WIN32)
    std::vector<DWORD_PTR> core_masks = physical_core_masks();
    int physical_cores = core_masks.empty() ? logical_cores : (int)core_masks.size();
#else
    int physical_cores = logical_cores;
#endif
    // Enumeration is FP-bound, so the default worker count is the physical core
    // count, not the logical (hyperthreaded) count. --use-hyperthreads restores
    // the old all-logical-processors default.
    p.threads = physical_cores;
    bool use_hyperthreads = false;
    bool no_pin = false;
    bool no_transform = false;
    bool no_gui = false;
    bool no_progressive = false;
    uint64_t seed = std::random_device{}();

    app.add_option("-L,--lattice", l_csv,
                   "CSV lattice basis to reduce (rows are vectors)")->required();
    app.add_option("-o,--out", out_csv, "Output CSV for the reduced basis");
    app.add_option("--transform-out", u_csv,
                   "Output CSV for the unimodular transform U (reduced = U * L)");
    app.add_option("--shortest-out", short_csv,
                   "Output CSV for the shortest vector (first reduced row)");
    app.add_flag("--no-transform", no_transform,
                 "Do not track or write the transform U. Saves memory and time, "
                 "recommended for large n, where each worker otherwise holds a "
                 "full n x n transform in addition to its basis and Gram data.");
    CLI::Option* threads_opt =
        app.add_option("-t,--threads", p.threads,
                       "Number of worker threads (default: physical core count)");
    app.add_flag("--use-hyperthreads", use_hyperthreads,
                 "Default the worker count to all logical processors instead of "
                 "physical cores (ignored if --threads is given)");
    app.add_flag("--no-pin", no_pin,
                 "Do not pin worker threads to physical cores (Windows). By "
                 "default each worker is pinned to its own core.");
    app.add_option("-b,--block", p.block, "Maximum BKZ block size");
    app.add_option("--block-start", p.block_start,
                   "Minimum BKZ block size. Each worker's schedule ramps from "
                   "block-start up to block, progressively by default "
                   "(see --no-progressive)");
    app.add_option("--delta", p.delta, "LLL delta in (0.25, 1.0)");
    app.add_option("--prune", p.prune,
                   "Enumeration pruning in [0, 1]. 0 is exact, higher is faster "
                   "but may miss vectors");
    app.add_option("--enum-node-limit", p.enum_node_limit,
                   "Maximum Schnorr-Euchner nodes per block (0 = unlimited)");
    app.add_flag("--no-progressive", no_progressive,
                 "Disable the per-worker progressive beta schedule and pick a "
                 "random block size per tour instead (the previous behaviour)");
    app.add_option("--preprocess-beta", p.preprocess_beta,
                   "Local block preprocessing block size before each full "
                   "enumeration (0 = off). A cheap smaller-beta pass that shrinks "
                   "the enumeration tree (BKZ 2.0 style)");
    app.add_option("--gh-factor", p.gh_factor,
                   "Gaussian-heuristic enumeration radius cap: search radius is "
                   "capped at gh-factor * GH(block) (0 = off, no cap). Around "
                   "1.1 prunes hard blocks. Missed vectors are recovered by "
                   "re-randomization across tours and workers");
    app.add_option("--sieve-beta", p.sieve_beta,
                   "Use the block sieve instead of enumeration for tours whose "
                   "block size exceeds this (0 = enumeration only). A tour uses a "
                   "single oracle, never a mix");
    app.add_option("--sieve-pool", p.sieve_pool,
                   "Block sieve pool size (used only by sieve tours)");
    app.add_option("--sieve-iters", p.sieve_iters,
                   "Sieve work budget per block is sieve-pool * sieve-iters + seeds");
    app.add_option("--max-seconds", p.max_seconds,
                   "Wall-clock budget. 0 runs until Ctrl-C");
    app.add_option("--reseed-every-k", p.reseed_every_k,
                   "Reseed a worker from the global best after this many "
                   "consecutive tours without local improvement "
                   "(0 disables reseeding)");
    app.add_option("--seed", seed, "Base RNG seed");
    app.add_flag("--no-init-lll", p.no_init_lll,
                 "Skip the shared initial LLL pass (use when the input is already "
                 "reduced). Perturbation shears still run but their follow-up LLL "
                 "is skipped so workers diversify without re-reducing.");
#if defined(_WIN32)
    app.add_flag("--no-gui", no_gui, "Run without the Win32 progress window");
#endif

    CLI11_PARSE(app, argc, argv);

    p.seed = seed;
    p.track_u = !no_transform;
    p.pin_threads = !no_pin;
    p.progressive = !no_progressive;
    // With no explicit --threads, --use-hyperthreads switches the default from
    // physical cores to all logical processors.
    if (threads_opt->count() == 0 && use_hyperthreads)
        p.threads = logical_cores;
#if defined(_WIN32)
    p.report_status = !no_gui;
#else
    p.report_status = false;
#endif
    if (p.threads < 1) p.threads = 1;
    if (p.block_start < 2) p.block_start = 2;
    if (p.block < p.block_start) p.block = p.block_start;
    if (p.delta <= 0.25 || p.delta >= 1.0) {
        std::cerr << "error: --delta must be in (0.25, 1.0)\n";
        return 1;
    }
    if (p.prune < 0.0 || p.prune > 1.0) {
        std::cerr << "error: --prune must be in [0, 1]\n";
        return 1;
    }
    if (p.enum_node_limit < 0) {
        std::cerr << "error: --enum-node-limit must be >= 0 (0 = unlimited)\n";
        return 1;
    }
    if (p.sieve_pool < 0 || p.sieve_pool > 4096) {
        std::cerr << "error: --sieve-pool must be in [0, 4096]\n";
        return 1;
    }
    if (p.sieve_iters < 0 || p.sieve_iters > 100000) {
        std::cerr << "error: --sieve-iters must be in [0, 100000]\n";
        return 1;
    }
    if (p.sieve_beta < 0) {
        std::cerr << "error: --sieve-beta must be >= 0 (0 = enumeration only)\n";
        return 1;
    }
    if (p.preprocess_beta < 0) {
        std::cerr << "error: --preprocess-beta must be >= 0 (0 = off)\n";
        return 1;
    }
    if (p.gh_factor < 0.0) {
        std::cerr << "error: --gh-factor must be >= 0 (0 = off)\n";
        return 1;
    }

    Lattice L0;
    try {
        L0 = read_lattice_csv(l_csv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (L0.m < 2) {
        std::cerr << "error: need at least 2 basis vectors\n";
        return 1;
    }
    if (L0.m > L0.d) {
        std::cerr << "error: basis has more vectors (" << L0.m << ") than the "
                  << "dimension (" << L0.d << "), rows must be independent\n";
        return 1;
    }

    real_t init_short = 0;
    {
        Reducer probe;
        probe.init(L0, false);
        init_short = probe.row_norm2(0);
        for (int i = 1; i < probe.n; ++i)
            init_short = std::min(init_short, probe.row_norm2(i));
    }

    std::cout << "[xbkz] loaded " << L0.m << "x" << L0.d << " basis\n";
    std::cout << "[xbkz] shortest input row norm " << std::sqrt((double)init_short)
              << " (norm^2 = " << norm2_str(init_short) << ")\n";
#if defined(_WIN32)
    std::cout << "[xbkz] cores: " << physical_cores << " physical / "
              << logical_cores << " logical, worker pinning "
              << (p.pin_threads && !core_masks.empty() ? "on" : "off") << "\n";
#endif
    std::cout << "[xbkz] threads " << p.threads << ", block size range "
              << p.block_start << ".." << p.block << ", delta " << p.delta
              << ", prune " << p.prune;
    if (p.sieve_beta > 0 && p.sieve_pool > 0 && p.sieve_iters > 0)
        std::cout << ", sieve tours with beta > " << p.sieve_beta << " (pool "
                  << p.sieve_pool << " x " << p.sieve_iters << " iters)";
    else
        std::cout << ", enumeration only";
    std::cout << "\n";
    std::cout << "[xbkz] schedule "
              << (p.progressive ? "progressive" : "random-beta");
    if (p.preprocess_beta > 0)
        std::cout << ", block preprocessing beta " << p.preprocess_beta;
    if (p.gh_factor > 0.0)
        std::cout << ", GH radius cap x" << p.gh_factor;
    std::cout << "\n";
    if (p.max_seconds > 0.0)
        std::cout << "[xbkz] time budget " << p.max_seconds << " s\n";
    else
        std::cout << "[xbkz] running until Ctrl-C\n";
    std::cout.flush();

    GlobalBest best;
    if (p.no_init_lll)
        std::cout << "[xbkz] skipping initial LLL (--no-init-lll)\n";
    else
        std::cout << "[xbkz] initial LLL reduction...\n";
    std::cout.flush();
    if (!prepare_starting_basis(L0, p, best)) {
        std::cerr << "error: initial LLL failed (basis overflow)\n";
        return 1;
    }
    {
        real_t after = best.b0;
        if (p.no_init_lll)
            std::cout << "[xbkz] starting from input basis, shortest row norm "
                      << std::sqrt((double)after)
                      << " (norm^2 = " << norm2_str(after) << ")\n";
        else
            std::cout << "[xbkz] after initial LLL, shortest row norm "
                      << std::sqrt((double)after)
                      << " (norm^2 = " << norm2_str(after) << ")\n";
    }
    std::cout.flush();

    std::vector<WorkerStatus> worker_status((size_t)p.threads);
    std::atomic<int> workers_alive{ p.threads };
    auto start = Clock::now();

    std::thread monitor;
    if (p.max_seconds > 0.0) {
        monitor = std::thread([&]() {
            auto budget = std::chrono::duration<double>(p.max_seconds);
            while (!g_stop.load(std::memory_order_relaxed)) {
                if (std::chrono::duration<double>(Clock::now() - start) >= budget) {
                    g_stop.store(true, std::memory_order_relaxed);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }

    std::vector<std::thread> pool;
    for (int i = 0; i < p.threads; ++i) {
        pool.emplace_back([&, i]() {
#if defined(_WIN32)
            if (p.pin_threads && !core_masks.empty())
                SetThreadAffinityMask(GetCurrentThread(),
                                      core_masks[(size_t)i % core_masks.size()]);
#endif
            worker(i, L0, p, best, worker_status[(size_t)i]);
            workers_alive.fetch_sub(1, std::memory_order_relaxed);
        });
    }

#if defined(_WIN32)
    const bool use_gui = !no_gui;
#else
    const bool use_gui = false;
#endif

    if (use_gui) {
        BkzUiConfig ui_cfg{};
        ui_cfg.threads = p.threads;
        ui_cfg.block_start = p.block_start;
        ui_cfg.block = p.block;
        ui_cfg.lattice_m = L0.m;
        ui_cfg.lattice_d = L0.d;
        ui_cfg.init_short_norm2 = (double)init_short;
        ui_cfg.max_seconds = p.max_seconds;
        xbkz_ui_run(ui_cfg, start, worker_status, best, g_stop, [&]() {
            return workers_alive.load(std::memory_order_acquire) == 0;
        });
    }

    for (auto& th : pool) th.join();
    if (monitor.joinable()) monitor.join();

    if (!best.has) {
        std::cerr << "error: no result produced\n";
        return 1;
    }

    if (g_interrupted.load(std::memory_order_relaxed))
        std::cout << "[interrupted] writing best basis found so far\n";

    int short_idx = 0;
    real_t final_short = flat_shortest_norm2(best.B, best.n, best.d, &short_idx);
    std::cout << "[xbkz] best shortest vector norm " << std::sqrt((double)final_short)
              << " (norm^2 = " << norm2_str(final_short) << ")\n";

    try {
        write_flat_csv(best.B, best.n, best.d, out_csv);
        std::vector<std::vector<int64_t>> shortest(1);
        shortest[0].assign(best.B.data() + (size_t)short_idx * best.d,
                           best.B.data() + (size_t)short_idx * best.d + best.d);
        write_rows_csv(shortest, short_csv);
        std::cout << "[xbkz] wrote reduced basis to " << out_csv
                  << " and shortest vector to " << short_csv << "\n";
        if (p.track_u) {
            if (best.track_u && best.u_valid) {
                write_flat_csv(best.U, best.n, best.n, u_csv);
                std::cout << "[xbkz] wrote transform U to " << u_csv
                          << " (reduced = U * L)\n";
            } else {
                std::cout << "[xbkz] transform U omitted: its entries exceeded "
                             "int64 during reduction\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
