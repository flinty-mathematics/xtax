// xtax: a multithreaded random-congruence annealer that drives a symmetric
// integer matrix X^T A X toward a diagonal form by minimizing the L1 sparsity
// score 2*sum|A_ij| - sum|A_ii|. See README.md for the full description.
//
// The shared machinery lives in the common headers: mat_io.hpp (Matrix, Lattice,
// CSV IO, gram/transpose/unimodularity helpers), stop_signal.hpp (Ctrl-C / stop
// flags), and congruence_anneal.hpp (the templated simulated-annealing engine).
// This file provides the L1 objective policy the engine anneals, plus the
// deflation outer loop and the A / L mode command-line plumbing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "CLI11.hpp"
#include "mat_io.hpp"
#include "stop_signal.hpp"
#include "congruence_anneal.hpp"

// Largest off-diagonal entry in a +/-1 pivot row that the deflation stage will
// clear. The clearing shears have |s| <= this value, and with matrix entries
// bounded by MAGNITUDE_LIMIT (2^48) the product |s| * entry stays below 2^62, so
// the int64 arithmetic in try_add cannot overflow. A unit pivot worth locking in
// has tiny off-diagonal mass anyway, so this generous cap is rarely the limit.
constexpr int64_t DEFLATE_OFFDIAG_CAP = 1 << 13;

// Evaluate an Add(i,j,s) move without mutating A or Xt.
//
// An Add only changes column j of X (row j of the transposed Xt) and row/column
// j (plus the diagonal jj) of the symmetric A, so everything here is O(n).
// Returns false (the move is infeasible) if any affected entry of X or A would
// exceed MAGNITUDE_LIMIT. On success new_a_row holds the proposed row j of A
// (with new_a_row[j] = new A_jj), delta is the score change, and d_nonzero is
// the change in the number of nonzero off-diagonal pairs.
//
// new_xt_row is optional: when non-null it receives the proposed row j of Xt
// (and that part of X is magnitude-checked). The annealer's hot path evaluates
// the score with new_xt_row = nullptr and only computes the transform row once a
// move is actually accepted, since the vast majority of proposals are rejected.
// The deflation path passes a real buffer because it commits every shear.
static bool try_add(const Matrix& A, const Matrix& Xt, int i, int j, int64_t s,
                    std::vector<int64_t>& new_a_row, std::vector<int64_t>* new_xt_row,
                    int64_t& delta, int64_t& d_nonzero) {
    const int n = (int)A.n;

    if (new_xt_row) {
        const int64_t* xt_row_j = Xt.data.data() + (size_t)j * n;
        const int64_t* xt_row_i = Xt.data.data() + (size_t)i * n;
        for (int r = 0; r < n; ++r) {
            const int64_t v = xt_row_j[r] + s * xt_row_i[r];
            if (v > MAGNITUDE_LIMIT || v < -MAGNITUDE_LIMIT) return false;
            (*new_xt_row)[r] = v;
        }
    }

    const int64_t a_ii = A.at(i, i);
    const int64_t a_ji = A.at(j, i);
    const int64_t* a_row_j = A.data.data() + (size_t)j * n;
    const int64_t* a_row_i = A.data.data() + (size_t)i * n;

    int64_t off_delta = 0;
    int64_t dnz = 0;
    {
        for (int k = 0; k < n; ++k) {
            if (k == j) continue;
            const int64_t old = a_row_j[k];
            const int64_t nv = old + s * a_row_i[k];
            if (nv > MAGNITUDE_LIMIT || nv < -MAGNITUDE_LIMIT) return false;
            new_a_row[k] = nv;
            off_delta += std::llabs(nv) - std::llabs(old);
            if (old == 0) { if (nv != 0) ++dnz; }
            else if (nv == 0) --dnz;
        }
    }

    const int64_t old_jj = a_row_j[j];
    const int64_t new_jj = old_jj + 2 * s * a_ji + s * s * a_ii;
    if (new_jj > MAGNITUDE_LIMIT || new_jj < -MAGNITUDE_LIMIT) return false;
    new_a_row[j] = new_jj;

    // Each off-diagonal change is counted twice in 2*sum|A| (symmetric pair). The
    // diagonal change contributes once to 2*sum|A| and once to -sum|A_ii|.
    delta = 4 * off_delta + (std::llabs(new_jj) - std::llabs(old_jj));
    d_nonzero = dnz;
    return true;
}

// Compute the proposed row j of Xt (column j of X) for an accepted Add(i,j,s),
// magnitude-checking each entry. Returns false if any entry would exceed
// MAGNITUDE_LIMIT, in which case the move must be skipped. Split out of try_add
// so the O(n) transform work is paid only on the accept path.
static bool add_transform_row(const Matrix& Xt, int i, int j, int64_t s,
                              std::vector<int64_t>& new_xt_row) {
    const int n = (int)Xt.n;
    const int64_t* xt_row_j = Xt.data.data() + (size_t)j * n;
    const int64_t* xt_row_i = Xt.data.data() + (size_t)i * n;
    for (int r = 0; r < n; ++r) {
        const int64_t v = xt_row_j[r] + s * xt_row_i[r];
        if (v > MAGNITUDE_LIMIT || v < -MAGNITUDE_LIMIT) return false;
        new_xt_row[r] = v;
    }
    return true;
}

// Commit the precomputed result of try_add into A (row/col j) and Xt (row j),
// and keep row_mass (per-row off-diagonal absolute mass) up to date.
static void commit_add(Matrix& A, Matrix& Xt, int j,
                       const std::vector<int64_t>& new_a_row,
                       const std::vector<int64_t>& new_xt_row,
                       std::vector<int64_t>& row_mass) {
    const int n = (int)A.n;
    int64_t* a_row_j = A.data.data() + (size_t)j * n;
    int64_t mass_j = 0;
    for (int k = 0; k < n; ++k) {
        if (k == j) continue;
        const int64_t old = a_row_j[k];
        const int64_t nv = new_a_row[k];
        a_row_j[k] = nv;             // contiguous row write
        A.at(k, j) = nv;             // strided symmetric mirror
        const int64_t na = std::llabs(nv);
        row_mass[k] += na - std::llabs(old);
        mass_j += na;
    }
    a_row_j[j] = new_a_row[j];
    row_mass[j] = mass_j;
    int64_t* xt_row_j = Xt.data.data() + (size_t)j * n;
    for (int r = 0; r < n; ++r) xt_row_j[r] = new_xt_row[r];
}

// row_mass[r] = sum over k != r of |A_rk|. Recomputed when a worker (re)seeds.
static void compute_row_mass(const Matrix& A, std::vector<int64_t>& row_mass) {
    const int n = (int)A.n;
    for (int r = 0; r < n; ++r) {
        const int64_t* row = A.data.data() + (size_t)r * n;
        int64_t m = 0;
        for (int k = 0; k < n; ++k) if (k != r) m += std::llabs(row[k]);
        row_mass[r] = m;
    }
}

// Change in the L1 score from applying Add(i,j,s), computed directly (no scratch
// written). Used only by best_shear_l1 to score candidate shears. The value
// matches try_add's delta exactly. Overflowing candidates are given a huge
// penalty so they lose the argmin (the engine's evaluate would reject them).
static int64_t score_delta_l1(const Matrix& A, int i, int j, int64_t s) {
    const int n = (int)A.n;
    const int64_t* a_row_j = A.data.data() + (size_t)j * n;
    const int64_t* a_row_i = A.data.data() + (size_t)i * n;
    int64_t off_delta = 0;
    for (int k = 0; k < n; ++k) {
        if (k == j) continue;
        const int64_t old = a_row_j[k];
        const int64_t nv = old + s * a_row_i[k];
        if (nv > MAGNITUDE_LIMIT || nv < -MAGNITUDE_LIMIT) return (1ll << 62);
        off_delta += std::llabs(nv) - std::llabs(old);
    }
    const int64_t a_ii = A.at(i, i);
    const int64_t a_ji = A.at(j, i);
    const int64_t old_jj = a_row_j[j];
    const int64_t new_jj = old_jj + 2 * s * a_ji + s * s * a_ii;
    if (new_jj > MAGNITUDE_LIMIT || new_jj < -MAGNITUDE_LIMIT) return (1ll << 62);
    return 4 * off_delta + (std::llabs(new_jj) - std::llabs(old_jj));
}

// Exact best integer shear s for Add(pivot=i, target=j) under the L1 score.
//
// The score change is
//   f(s) = 4 * sum_{k != j} |a_jk + s a_ik| + |a_jj + 2 s a_ij + s^2 a_ii|.
// The off-diagonal part 4 * sum_k |a_ik| |s - (-a_jk/a_ik)| is a weighted sum of
// absolute values, minimized at the |a_ik|-weighted median of the breakpoints
// -a_jk/a_ik. We take that real minimizer, then evaluate the exact integer
// f(s) at the integer neighbours of the median and of the diagonal-reducing
// quotient -round(a_ji/a_ii) (which minimizes the |diagonal| term), clamped to
// SHEAR_CAP, and return the best strictly-or-equally improving s. Returns 0 when
// no nonzero shear helps (the caller then explores with a random +/-1).
static int64_t best_shear_l1(const Matrix& A, int i, int j) {
    const int n = (int)A.n;
    const int64_t* a_row_i = A.data.data() + (size_t)i * n;
    const int64_t* a_row_j = A.data.data() + (size_t)j * n;

    // Collect the off-diagonal breakpoints b_k = -a_jk / a_ik with weight |a_ik|.
    std::vector<std::pair<double, int64_t>> bp;   // (breakpoint, weight)
    bp.reserve((size_t)n);
    int64_t total_w = 0;
    for (int k = 0; k < n; ++k) {
        if (k == j) continue;
        const int64_t aik = a_row_i[k];
        if (aik == 0) continue;
        const double b = -(double)a_row_j[k] / (double)aik;
        const int64_t w = std::llabs(aik);
        bp.emplace_back(b, w);
        total_w += w;
    }

    double median = 0.0;
    if (!bp.empty()) {
        std::sort(bp.begin(), bp.end(),
                  [](const auto& p, const auto& q) { return p.first < q.first; });
        int64_t acc = 0;
        median = bp.back().first;
        for (const auto& p : bp) {
            acc += p.second;
            if (2 * acc >= total_w) { median = p.first; break; }
        }
    }

    // Candidate integer shears: neighbours of the weighted median and of the
    // diagonal-reducing quotient, plus 0. Deduplicated and clamped to the cap.
    int64_t cands[8];
    int nc = 0;
    auto add_cand = [&](double x) {
        int64_t v = (int64_t)std::llround(x);
        if (v > SHEAR_CAP) v = SHEAR_CAP;
        else if (v < -SHEAR_CAP) v = -SHEAR_CAP;
        for (int t = 0; t < nc; ++t) if (cands[t] == v) return;
        if (nc < 8) cands[nc++] = v;
    };
    add_cand(std::floor(median));
    add_cand(std::ceil(median));
    const int64_t a_ii = A.at(i, i);
    if (a_ii != 0) {
        const int64_t q = -rounded_div(a_row_j[i], a_ii);
        add_cand((double)q);
        add_cand((double)(q + 1));
        add_cand((double)(q - 1));
    }

    int64_t best_s = 0;
    int64_t best_delta = 0;   // s = 0 gives delta 0, only accept a strict/equal improve
    for (int t = 0; t < nc; ++t) {
        const int64_t s = cands[t];
        if (s == 0) continue;
        const int64_t d = score_delta_l1(A, i, j, s);
        if (d < best_delta) { best_delta = d; best_s = s; }
    }
    return best_s;
}

// Reverse Cuthill-McKee ordering of the symmetric matrix A, viewing index i and
// j as adjacent whenever the off-diagonal entry A_ij is nonzero. Returns a
// permutation perm where perm[a] is the OLD index that becomes new index a. The
// symmetric reorder A'[a][b] = A[perm[a]][perm[b]] clusters the nonzeros toward
// the diagonal (a narrower bandwidth) while keeping the matrix symmetric.
static std::vector<int> rcm_order(const Matrix& A) {
    const int n = (int)A.n;
    std::vector<int> deg((size_t)n, 0);
    for (int i = 0; i < n; ++i) {
        const int64_t* row = A.data.data() + (size_t)i * n;
        int d = 0;
        for (int j = 0; j < n; ++j) if (j != i && row[j] != 0) ++d;
        deg[(size_t)i] = d;
    }

    std::vector<char> visited((size_t)n, 0);
    std::vector<int> order;
    order.reserve((size_t)n);
    std::vector<int> nbrs;

    for (;;) {
        int root = -1;
        for (int i = 0; i < n; ++i)
            if (!visited[(size_t)i] && (root < 0 || deg[(size_t)i] < deg[(size_t)root])) root = i;
        if (root < 0) break;

        visited[(size_t)root] = 1;
        order.push_back(root);
        for (size_t head = order.size() - 1; head < order.size(); ++head) {
            const int v = order[head];
            const int64_t* row = A.data.data() + (size_t)v * n;
            nbrs.clear();
            for (int j = 0; j < n; ++j)
                if (j != v && row[j] != 0 && !visited[(size_t)j]) nbrs.push_back(j);
            std::sort(nbrs.begin(), nbrs.end(),
                      [&](int a, int b) { return deg[(size_t)a] < deg[(size_t)b]; });
            for (int j : nbrs) { visited[(size_t)j] = 1; order.push_back(j); }
        }
    }

    std::reverse(order.begin(), order.end());   // CM -> reverse CM
    return order;
}

// Iterative centroid (row centre-of-mass) reordering of the symmetric matrix A.
// Unlike RCM this weights by entry magnitude, so it pulls heavy mass toward the
// diagonal even when the matrix is dense. Returns a permutation perm where
// perm[a] is the OLD index that becomes new index a.
static std::vector<int> centroid_order(const Matrix& A, int max_passes = 8) {
    const int n = (int)A.n;
    std::vector<int> order((size_t)n);
    for (int a = 0; a < n; ++a) order[(size_t)a] = a;
    if (n < 2) return order;

    std::vector<double> key((size_t)n);
    std::vector<int> pos((size_t)n);
    for (int pass = 0; pass < max_passes; ++pass) {
        for (int a = 0; a < n; ++a) {
            const int64_t* row = A.data.data() + (size_t)order[(size_t)a] * n;
            long double num = 0.0L, den = 0.0L;
            for (int b = 0; b < n; ++b) {
                const long double w = (long double)std::llabs(row[order[(size_t)b]]);
                num += (long double)b * w;
                den += w;
            }
            key[(size_t)a] = (den > 0.0L) ? (double)(num / den) : (double)a;
        }
        for (int a = 0; a < n; ++a) pos[(size_t)a] = a;
        std::stable_sort(pos.begin(), pos.end(),
                         [&](int p, int q) { return key[(size_t)p] < key[(size_t)q]; });
        bool changed = false;
        std::vector<int> reordered((size_t)n);
        for (int a = 0; a < n; ++a) {
            reordered[(size_t)a] = order[(size_t)pos[(size_t)a]];
            if (pos[(size_t)a] != a) changed = true;
        }
        order.swap(reordered);
        if (!changed) break;
    }
    return order;
}

// Apply the symmetric permutation perm (new index a <- old index perm[a]) in
// place to the working matrix A, the transposed transform Xt (row a becomes old
// row perm[a]), and the caller's per-row off-diagonal mass. The score and the
// off-diagonal nonzero count are permutation-invariant and stay valid.
static void permute_state(Matrix& A, Matrix& Xt, const std::vector<int>& perm,
                          std::vector<int64_t>& row_mass) {
    const int n = (int)A.n;
    Matrix A2(A.n);
    for (int a = 0; a < n; ++a) {
        const int64_t* src = A.data.data() + (size_t)perm[(size_t)a] * n;
        int64_t* dst = A2.data.data() + (size_t)a * n;
        for (int b = 0; b < n; ++b) dst[b] = src[perm[(size_t)b]];
    }
    Matrix Xt2(Xt.n);
    for (int a = 0; a < n; ++a) {
        const int64_t* src = Xt.data.data() + (size_t)perm[(size_t)a] * n;
        int64_t* dst = Xt2.data.data() + (size_t)a * n;
        for (int k = 0; k < n; ++k) dst[k] = src[k];
    }
    A = std::move(A2);
    Xt = std::move(Xt2);
    std::vector<int64_t> rm2((size_t)n);
    for (int a = 0; a < n; ++a) rm2[(size_t)a] = row_mass[(size_t)perm[(size_t)a]];
    row_mass.swap(rm2);
}

// The L1 objective policy for the shared annealing engine. It owns the working
// matrix A = X^T A0 X, the accumulated transform stored transposed (Xt = X^T),
// the running L1 score, the off-diagonal nonzero count, and the per-row
// off-diagonal mass used to bias shear targets.
struct L1Objective {
    using score_t = int64_t;

    Matrix A;
    Matrix Xt;
    int64_t score_ = 0;
    int64_t off_nonzero_ = 0;
    std::vector<int64_t> row_mass;
    std::vector<int64_t> new_a_row;
    std::vector<int64_t> new_xt_row;
    bool band_sort = false;
    bool centroid_sort = false;

    L1Objective() = default;
    L1Objective(const Matrix& A0, const Matrix& X_start, bool band, bool centroid)
        : A(A0), Xt(transpose(X_start)), score_(A0.score()),
          off_nonzero_(A0.count_offdiag_nonzero()),
          row_mass((size_t)A0.n), new_a_row((size_t)A0.n), new_xt_row((size_t)A0.n),
          band_sort(band), centroid_sort(centroid) {}

    int n() const { return (int)A.n; }
    score_t score() const { return score_; }
    int64_t offdiag_nonzero() const { return off_nonzero_; }
    bool solved() const { return off_nonzero_ == 0; }
    int64_t row_weight(int r) const { return row_mass[(size_t)r]; }
    int64_t pivot_abs(int t, int c) const { return std::llabs(A.at(t, c)); }

    double suggest_t_init() const {
        const int nn = (int)A.n;
        const double avg_entry = (double)score_ / (double)((int64_t)nn * nn);
        return std::max(1.0, 2.0 * avg_entry);
    }

    bool evaluate(int i, int j, int64_t s, score_t& d_score, int64_t& d_nonzero) {
        return try_add(A, Xt, i, j, s, new_a_row, nullptr, d_score, d_nonzero);
    }

    bool commit(int i, int j, int64_t s, score_t d_score, int64_t d_nonzero) {
        if (!add_transform_row(Xt, i, j, s, new_xt_row)) return false;
        commit_add(A, Xt, j, new_a_row, new_xt_row, row_mass);
        score_ += d_score;
        off_nonzero_ += d_nonzero;
        return true;
    }

    int64_t best_shear(int i, int j) const { return best_shear_l1(A, i, j); }

    void refresh_cache() { compute_row_mass(A, row_mass); }
    void periodic_maintenance(uint64_t) {}
    score_t recompute_score() { score_ = A.score(); return score_; }

    void reorder_for_publish(bool enabled) {
        if (!enabled) return;
        if (band_sort) permute_state(A, Xt, rcm_order(A), row_mass);
        if (centroid_sort) permute_state(A, Xt, centroid_order(A), row_mass);
    }

    void publish_files() const {
        write_matrix_csv(transpose(Xt), "best_X.csv");
        write_matrix_csv(A, "best_A.csv");
    }

    std::string best_line() const { return "score=" + std::to_string(score_); }
};

struct Params {
    canneal::EngineParams engine;
    bool deflate = false;          // strict deflation outer loop (needs unimodular A)
    bool deflate_blocks = false;   // relaxed deflation: peel off orthogonal summands
    double deflate_slice = 0.5;    // annealing seconds per slice before checking pivots
    bool band_sort = false;        // reorder toward a band (RCM) on each new best
    bool centroid_sort = false;    // reorder by iterative row centre-of-mass on each best
};

// Clear the off-diagonal entries of row/column i (whose pivot M_ii is +/-1) with
// exact integer shears, updating the working matrix M and the transposed
// transform Xt. Each shear s = -M_ji / M_ii is exact because the pivot is +/-1.
// Returns false and leaves M, Xt unchanged if any shear would push an entry past
// MAGNITUDE_LIMIT. The caller guarantees |M_ii| == 1 and a small pivot row.
static bool deflate_index(Matrix& M, Matrix& Xt, int i, const std::vector<int>& active,
                          std::vector<int64_t>& new_a_row, std::vector<int64_t>& new_xt_row,
                          std::vector<int64_t>& scratch_mass) {
    const int64_t a_ii = M.at(i, i);
    const Matrix M0 = M;     // snapshot for rollback on an out-of-range shear
    const Matrix Xt0 = Xt;
    int64_t delta = 0, dnz = 0;
    for (int j : active) {
        if (j == i) continue;
        const int64_t a_ji = M.at(j, i);
        if (a_ji == 0) continue;
        const int64_t s = (a_ii == 1) ? -a_ji : a_ji;   // s = -a_ji / a_ii, exact for +/-1
        if (!try_add(M, Xt, i, j, s, new_a_row, &new_xt_row, delta, dnz)) {
            M = M0;
            Xt = Xt0;
            return false;
        }
        commit_add(M, Xt, j, new_a_row, new_xt_row, scratch_mass);
    }
    return true;
}

// Deflation solver. Repeatedly anneal the active sub-problem for a short slice,
// then lock in pivots and drop them from the active set, so the working problem
// shrinks monotonically. See README.md for the strict / relaxed rules.
static L1Objective solve_with_deflation(const Matrix& A, const Params& params, bool relaxed) {
    const int n = (int)A.n;
    Matrix M = A;
    Matrix Xt(A.n);
    Xt.fill_identity();                          // Xt = X^T, X = I, so M = X^T A X
    std::vector<int> active((size_t)n);
    for (int i = 0; i < n; ++i) active[i] = i;

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    const bool quiet = params.engine.quiet;
    if (!quiet) {
        std::cout << "[deflate] start mode=" << (relaxed ? "blocks" : "strict")
                  << " score=" << M.score() << " n=" << n
                  << " slice=" << params.deflate_slice << "s threads="
                  << params.engine.threads << "\n";
    }

    canneal::EngineParams sp = params.engine;
    sp.quiet = true;                             // silence per-slice worker output

    std::vector<int64_t> new_a_row((size_t)n), new_xt_row((size_t)n), scratch_mass((size_t)n, 0);
    const double report_every = std::max(params.engine.save_interval, params.deflate_slice);
    double last_report = -1e18;

    while (active.size() >= 2) {
        if (g_interrupted.load(std::memory_order_relaxed)) break;
        const double el = elapsed();
        if (params.engine.max_seconds > 0.0) {
            if (el >= params.engine.max_seconds) break;
            sp.max_seconds = std::min(params.deflate_slice, params.engine.max_seconds - el);
        } else {
            sp.max_seconds = params.deflate_slice;
        }

        L1Objective start(M, transpose(Xt), false, false);
        const L1Objective gb = canneal::run_annealer(start, sp, &active);
        M = gb.A;
        Xt = gb.Xt;

        int locked = 0;
        const std::vector<int> snapshot = active;
        for (int i : snapshot) {
            const int64_t a_ii = M.at(i, i);
            if (relaxed) {
                if (a_ii == 0) continue;
                bool orthogonal = true;          // off-diagonal row already zero?
                for (int j : active) {
                    if (j == i) continue;
                    if (M.at(j, i) != 0) { orthogonal = false; break; }
                }
                if (!orthogonal) continue;
                active.erase(std::find(active.begin(), active.end(), i));
                ++locked;
            } else {
                if (std::llabs(a_ii) != 1) continue;
                int64_t max_off = 0;             // bounds |s| = |M_ji| for a +/-1 pivot
                for (int j : active) {
                    if (j == i) continue;
                    const int64_t v = std::llabs(M.at(j, i));
                    if (v > max_off) max_off = v;
                }
                if (max_off > DEFLATE_OFFDIAG_CAP) continue;
                if (deflate_index(M, Xt, i, active, new_a_row, new_xt_row, scratch_mass)) {
                    active.erase(std::find(active.begin(), active.end(), i));
                    ++locked;
                }
            }
        }

        const bool diagonal = (M.count_offdiag_nonzero() == 0);
        const double now = elapsed();
        if (!quiet && (locked > 0 || diagonal || now - last_report >= report_every)) {
            std::cout << "[deflate] t=" << now << "s score=" << M.score()
                      << " active=" << active.size()
                      << " offdiag=" << M.count_offdiag_nonzero();
            if (locked > 0) std::cout << " (locked " << locked << ")";
            std::cout << "\n";
            write_matrix_csv(transpose(Xt), "best_X.csv");
            write_matrix_csv(M, "best_A.csv");
            last_report = now;
        }

        if (diagonal) break;
    }

    if (!quiet) {
        std::cout << "[deflate] done active=" << active.size() << " score=" << M.score()
                  << " seconds=" << elapsed() << "\n";
    }
    L1Objective result(M, transpose(Xt), false, false);
    return result;
}

int main(int argc, char** argv) {
    // Stop gracefully on Ctrl-C so the best result found so far is still written.
    install_signal_handlers();

    CLI::App app{ "xtax cpu congruence annealer" };

    std::string a_csv;
    std::string l_csv;
    std::string x_csv;
    Params params;

    const int physical = canneal::physical_core_count();
    const int logical = std::max(1, (int)std::thread::hardware_concurrency());
    params.engine.threads = physical;
    bool use_hyperthreads = false;
    bool no_pin = false;
    CLI::Option* threads_opt = nullptr;

    app.add_option("-A", a_csv, "CSV file for A (n x n integers)");
    app.add_option("-L,--lattice", l_csv, "CSV file for a lattice basis (rows are vectors). Anneals the Gram A = L*L^T");
    app.add_option("-X", x_csv, "Path to initial X matrix (A mode). If none, start from identity");
    threads_opt = app.add_option("-t,--threads", params.engine.threads, "Number of worker threads (default: physical core count)");
    app.add_flag("--use-hyperthreads", use_hyperthreads, "Default the worker count to all logical processors instead of physical cores (ignored if --threads is given)");
    app.add_flag("--no-pin", no_pin, "Do not pin worker threads to physical cores (Windows, on by default)");
    app.add_option("--seed", params.engine.seed, "Base RNG seed for reproducibility (0 = seed from random_device)");
    app.add_option("--stuck-threshold", params.engine.stuck_threshold, "Moves without improvement before reheating");
    app.add_option("--t-init", params.engine.t_init, "Initial SA temperature (<= 0 auto-calibrates)");
    app.add_option("--t-min", params.engine.t_min, "Minimum SA temperature");
    app.add_option("--cooling", params.engine.cooling, "Geometric cooling factor per cooling step (0 < alpha < 1)");
    app.add_option("--moves-per-cool", params.engine.moves_per_cool, "Moves between cooling steps");
    app.add_option("--reheat", params.engine.reheat, "Fraction of initial temperature restored when stuck");
    app.add_option("--greedy-fraction", params.engine.greedy_fraction, "Probability an Add move uses the exact best-integer shear");
    app.add_option("--target-fraction", params.engine.target_fraction, "Probability an Add targets a hot row (0 = uniform random)");
    app.add_option("--target-samples", params.engine.target_samples, "Tournament size for hot-row / large-pivot selection");
    app.add_option("--reseed-factor", params.engine.reseed_factor, "Reseed from global best when stuck and score exceeds best by this factor");
    app.add_option("--sweep-fraction", params.engine.sweep_fraction, "Probability of a greedy reduction sweep when a worker stalls (0 = off)");
    app.add_flag("--tempering,!--no-tempering", params.engine.tempering, "Parallel-tempering temperature ladder with replica exchange, on by default with 2+ threads. --no-tempering restores per-worker cooling with reheat/reseed");
    app.add_option("--exchange-interval", params.engine.exchange_interval, "Moves between replica-exchange sweeps (tempering mode)");
    app.add_flag("--adaptive-cooling", params.engine.adaptive_cooling, "Nudge the cooling rate toward the target acceptance ratio");
    app.add_option("--worker-diversity", params.engine.worker_diversity, "Spread of per-worker greedy/target offsets (0 = identical workers)");
    app.add_option("--max-seconds", params.engine.max_seconds, "Wall-clock stop in seconds (<= 0 runs until a diagonal is found)");
    app.add_option("--save-interval", params.engine.save_interval, "Minimum seconds between best_*.csv disk writes");
    app.add_flag("--rcm", params.band_sort, "Reorder the working matrix toward a band (Reverse Cuthill-McKee) on each new best. Symmetric and score-preserving (default off)");
    app.add_flag("--centroid", params.centroid_sort, "Reorder the working matrix by iterative row centre-of-mass on each new best to pull mass toward the diagonal. Symmetric and score-preserving (default off)");
    app.add_flag("--deflate", params.deflate, "Strict deflation: lock +/-1 pivots and shrink the active problem (requires a unimodular matrix, starts from identity)");
    app.add_flag("--deflate-blocks", params.deflate_blocks, "Relaxed deflation: peel off orthogonal summands (works on any Gram, starts from identity)");
    app.add_option("--deflate-slice", params.deflate_slice, "Deflation: annealing seconds per slice before checking for pivots");
    app.add_flag("--verbose", params.engine.verbose, "Also show the inner annealer's progress inside --deflate / --deflate-blocks (does not write per-slice CSVs)");
    CLI11_PARSE(app, argc, argv);

    if (threads_opt->count() == 0 && use_hyperthreads) params.engine.threads = logical;
    if (params.engine.threads < 1) params.engine.threads = 1;
    params.engine.pin_threads = !no_pin;

    const bool have_A = !a_csv.empty();
    const bool have_L = !l_csv.empty();
    if (have_A == have_L) {
        std::cerr << "Error: provide exactly one of -A or -L\n";
        return 1;
    }
    if (params.deflate && params.deflate_blocks) {
        std::cerr << "Error: use at most one of --deflate or --deflate-blocks\n";
        return 1;
    }

    if (have_A) {
        // ----- A mode: diagonalize a given symmetric matrix -----
        Matrix A;
        Matrix X;
        try {
            A = read_matrix_csv(a_csv);
            if (!x_csv.empty()) {
                X = read_matrix_csv(x_csv);
                if (X.n != A.n) throw std::runtime_error("X dimensions do not match A");
            } else {
                X = Matrix(A.n);
                X.fill_identity();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        const int n = (int)A.n;
        if (params.deflate) {
            std::cout << "[deflate] verifying the matrix is unimodular (det = +/-1)...\n";
            if (!is_unimodular(A)) {
                std::cerr << "Error: --deflate requires a unimodular matrix (det = +/-1); this one is not.\n"
                          << "       Use --deflate-blocks for a relaxed (orthogonal-summand) deflation,\n"
                          << "       or drop --deflate to run the plain annealer.\n";
                return 1;
            }
        }
        // With a supplied initial transform the working matrix starts at
        // X^T A X, so the search continues where the previous run stopped and
        // the output invariant X^T A X == best_A holds against the input A.
        const Matrix A_work = x_csv.empty() ? A : congruence_of(A, X);
        const bool any_deflate = params.deflate || params.deflate_blocks;
        L1Objective global_best = any_deflate
            ? solve_with_deflation(A, params, params.deflate_blocks)
            : canneal::run_annealer(L1Objective(A_work, X, params.band_sort, params.centroid_sort),
                                    params.engine);
        std::cout << "Final best score: " << global_best.score() << "\n";
        const Matrix best_X = transpose(global_best.Xt);
        if (n <= 20) {
            std::cout << "A:\n";
            global_best.A.print();
            std::cout << "----\nX:\n";
            best_X.print();
        }
        if (g_interrupted.load(std::memory_order_relaxed))
            std::cout << "[interrupted] writing best_X.csv and best_A.csv before exit\n";
        write_matrix_csv(best_X, "best_X.csv");
        write_matrix_csv(global_best.A, "best_A.csv");
        return 0;
    }

    // ----- L mode: build the Gram matrix A = L*L^T and anneal it -----
    Lattice curL;
    try {
        curL = read_lattice_csv(l_csv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    const int m = curL.m, d = curL.d;
    if (m < 2) {
        std::cerr << "Error: lattice must have at least 2 basis vectors\n";
        return 1;
    }
    if (m > d) {
        std::cerr << "Warning: lattice has more vectors (" << m << ") than dimensions ("
                  << d << "); the basis is linearly dependent and the Gram matrix is singular\n";
    }

    Matrix A = gram_of(curL);
    std::cout << "[lattice] m=" << m << " d=" << d << " score=" << A.score() << "\n";

    Matrix I((size_t)m);
    I.fill_identity();
    if (params.deflate) {
        std::cout << "[deflate] verifying the Gram matrix is unimodular (det = +/-1)...\n";
        if (!is_unimodular(A)) {
            std::cerr << "Error: --deflate requires a unimodular matrix (det = +/-1); this Gram is not\n"
                      << "       (a lattice Gram is unimodular only when the lattice is unimodular).\n"
                      << "       Use --deflate-blocks for a relaxed (orthogonal-summand) deflation,\n"
                      << "       or drop --deflate to run the plain annealer.\n";
            return 1;
        }
    }

    const bool any_deflate = params.deflate || params.deflate_blocks;
    L1Objective gb = any_deflate
        ? solve_with_deflation(A, params, params.deflate_blocks)
        : canneal::run_annealer(L1Objective(A, I, params.band_sort, params.centroid_sort),
                                params.engine);
    const Matrix X = transpose(gb.Xt);
    std::cout << "Final best score: " << gb.score() << "\n";

    // Final basis of the same lattice: L_final = X^T * L, Gram = gb.A.
    Lattice Lfinal = xt_times(X, curL);
    const Matrix g2 = gram_of(Lfinal);
    if (g2.n != gb.A.n || g2.data != gb.A.data) {
        std::cout << "[warn] Gram(X^T L) != annealed Gram (transform mismatch, possible int64 overflow)\n";
    }
    if (g_interrupted.load(std::memory_order_relaxed))
        std::cout << "[interrupted] writing final_L.csv, best_A.csv and best_X.csv before exit\n";
    write_lattice_csv(Lfinal, "final_L.csv");
    write_matrix_csv(gb.A, "best_A.csv");
    write_matrix_csv(X, "best_X.csv");
    if (m <= 20 && d <= 20) {
        std::cout << "Final L:\n";
        for (int i = 0; i < m; ++i) {
            const int64_t* ri = Lfinal.row(i);
            for (int k = 0; k < d; ++k) { std::cout << ri[k]; if (k + 1 < d) std::cout << ','; }
            std::cout << '\n';
        }
        std::cout << "----\nGram:\n";
        gb.A.print();
    }
    return 0;
}
